import jax
import jax.numpy as jnp
from flax import nnx
from ngclearn.components import GaussianErrorCell as ErrorCell,HebbianSynapse,StaticSynapse
from ngclearn.utils import JaxProcess
import ngclearn.utils.weight_distribution as dist
from ngcsimlib.context import Context
from jax import random as jax_random
import optax
from embedding import EmbeddingLayer
from attention import MultiHeadAttention
from mlp import FeedForward
from output import OutputLayer
import torch.jit as jit
import pkg_resources
from ngclearn.utils.io_utils import makedir
import os
import tiktoken
import requests
import numpy as np
from ngcsimlib.compartment import Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngcsimlib.compilers.process import transition

class Transformer:
    def __init__(self, dkey, vocab_size=50257, n_embed=12, block_size=8, num_heads=1, num_layers=1,
                 dropout=0.1, batch_size=1, T=1, eta=0.001, exp_dir="exp", model_name="transformer"):
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.T = T
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_embed = n_embed
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f"{exp_dir}/filters", exist_ok=True)

        dkey, *subkeys = jax.random.split(dkey, 8)
        optim_type = "adam"
        wlb = -0.3
        wub = 0.3

        in_dim = block_size * n_embed
        hid1_dim = block_size * n_embed
        hid2_dim = block_size * n_embed
        out_dim = block_size * vocab_size

        with Context("Circuit") as self.circuit:
            self.emb = EmbeddingLayer("emb", n_embed, vocab_size, block_size, dropout, batch_size, key=subkeys[0])
            self.attn = MultiHeadAttention("attn", num_heads, n_embed // num_heads, n_embed, dropout, block_size, batch_size, key=subkeys[1])
            self.ffwd = FeedForward("ffwd", n_embed, dropout, batch_size, block_size, key=subkeys[2])
            self.output = OutputLayer("output", n_embed, vocab_size, batch_size, block_size, key=subkeys[3])
            self.e1 = ErrorCell("e1", n_units=hid1_dim, batch_size=batch_size)
            self.e2 = ErrorCell("e2", n_units=hid2_dim, batch_size=batch_size)
            self.e3 = ErrorCell("e3", n_units=out_dim, batch_size=batch_size)
            self.W1 = HebbianSynapse("W1", shape=(in_dim, hid1_dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub), bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4], batch_size=batch_size)
            self.W2 = HebbianSynapse("W2", shape=(hid1_dim, hid2_dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub), bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[5], batch_size=batch_size)
            self.W3 = HebbianSynapse("W3", shape=(hid2_dim, out_dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub), bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[6], batch_size=batch_size)
            self.E2 = StaticSynapse("E2", shape=(hid2_dim, hid1_dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4], batch_size=batch_size)
            self.E3 = StaticSynapse("E3", shape=(out_dim, hid2_dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5], batch_size=batch_size)
            
            
            # Store bias_init for evolve calls
            self.W1_bias_init = dist.constant(value=0.)
            self.W2_bias_init = dist.constant(value=0.)
            self.W3_bias_init = dist.constant(value=0.)
            # Wire compartments
            self.W1.inputs << self.emb.z
            self.e1.mu << self.W1.outputs
            self.e1.target << self.attn.z
            self.W2.inputs << self.attn.z
            self.e2.mu << self.W2.outputs
            self.e2.target << self.ffwd.z
            self.W3.inputs << self.ffwd.z
            self.e3.mu << self.W3.outputs
            self.e3.target << self.output.z
            self.E2.inputs << self.e2.dmu
            self.attn.j << self.E2.outputs
            self.attn.j_td << self.e1.dtarget
            self.E3.inputs << self.e3.dmu
            self.ffwd.j << self.E3.outputs
            self.ffwd.j_td << self.e2.dtarget
            self.W1.pre << self.emb.z
            self.W1.post << self.e1.dmu
            self.W2.pre << self.attn.z
            self.W2.post << self.e2.dmu
            self.W3.pre << self.ffwd.z
            self.W3.post << self.e3.dmu

    def process(self, obs, lab, adapt_synapses=True):
        # Reset components
        self.emb.reset()
        self.attn.reset()
        self.ffwd.reset()
        self.output.reset()
       # Pass required arguments to ErrorCell reset
        self.e1.reset(batch_size=self.batch_size, shape=(self.e1.n_units,), sigma_shape=(1, 1))
        self.e2.reset(batch_size=self.batch_size, shape=(self.e2.n_units,), sigma_shape=(1, 1))
        self.e3.reset(batch_size=self.batch_size, shape=(self.e3.n_units,), sigma_shape=(1, 1))
        self.W1.reset(self.batch_size, self.W1.shape)
        self.W2.reset(self.batch_size, self.W2.shape)
        self.W3.reset(self.batch_size, self.W3.shape)
        self.E2.reset(self.batch_size, self.E2.shape)
        self.E3.reset(self.batch_size, self.E2.shape)

        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        # One-hot encode lab to match vocab_size dimension
        _lab_one_hot = jax.nn.one_hot(_lab, self.vocab_size)  # Shape: (batch_size, block_size, vocab_size)
        _lab_flat = _lab_one_hot.reshape((self.batch_size, -1))  # Shape: (1, block_size * vocab_size)

        # Synchronize feedback weights
        self.E2.weights.set(jnp.transpose(self.W2.weights.value))
        self.E3.weights.set(jnp.transpose(self.W3.weights.value))

        # Expectation steps
        self.emb.j.set(obs)
        for ts in range(self.T):
            self.emb.advance_state(t=ts, dt=1.)
            self.attn.advance_state(t=ts, dt=1.)
            self.ffwd.advance_state(t=ts, dt=1.)
            self.output.advance_state(t=ts, dt=1.)
            self.W1.advance_state(Rscale=self.W1.Rscale, inputs=self.W1.inputs.value, weights=self.W1.weights.value, biases=self.W1.biases.value)
            self.W2.advance_state(Rscale=self.W2.Rscale, inputs=self.W2.inputs.value, weights=self.W2.weights.value, biases=self.W2.biases.value)
            self.W3.advance_state(Rscale=self.W3.Rscale, inputs=self.W3.inputs.value, weights=self.W3.weights.value, biases=self.W3.biases.value)
            self.E2.advance_state(Rscale=self.E2.Rscale, inputs=self.E2.inputs.value, weights=self.E2.weights.value, biases=self.E2.biases.value)
            self.E3.advance_state(Rscale=self.E3.Rscale, inputs=self.E3.inputs.value, weights=self.E3.weights.value, biases=self.E3.biases.value)
            self.e1.advance_state(dt=1., mu=self.e1.mu.value, target=self.e1.target.value, Sigma=self.e1.Sigma.value, modulator=self.e1.modulator.value, mask=self.e1.mask.value)
            self.e2.advance_state(dt=1., mu=self.e2.mu.value, target=self.e2.target.value, Sigma=self.e2.Sigma.value, modulator=self.e2.modulator.value, mask=self.e2.mask.value)
            self.e3.advance_state(dt=1., mu=self.e3.mu.value, target=_lab_flat, Sigma=self.e3.Sigma.value, modulator=self.e3.modulator.value, mask=self.e3.mask.value)

        y_mu = self.e3.mu.value.reshape((self.batch_size, self.block_size, self.vocab_size))
        EFE = self.e1.L.value + self.e2.L.value + self.e3.L.value

        # Maximization step
        if adapt_synapses:
            self.W1.evolve(
                opt=self.W1.opt,
                w_bound=self.W1.w_bound,
                is_nonnegative=self.W1.is_nonnegative,
                sign_value=self.W1.sign_value,
                prior_type=self.W1.prior_type,
                prior_lmbda=self.W1.prior_lmbda,
                pre_wght=self.W1.pre_wght,
                post_wght=self.W1.post_wght,
                bias_init=self.W1_bias_init,
                pre=self.W1.pre.value,
                post=self.W1.post.value,
                weights=self.W1.weights.value,
                biases=self.W1.biases.value,
                opt_params=self.W1.opt_params.value
            )
            self.W2.evolve(
                opt=self.W2.opt,
                w_bound=self.W2.w_bound,
                is_nonnegative=self.W2.is_nonnegative,
                sign_value=self.W2.sign_value,
                prior_type=self.W2.prior_type,
                prior_lmbda=self.W2.prior_lmbda,
                pre_wght=self.W2.pre_wght,
                post_wght=self.W2.post_wght,
                bias_init=self.W2_bias_init,
                pre=self.W2.pre.value,
                post=self.W2.post.value,
                weights=self.W2.weights.value,
                biases=self.W2.biases.value,
                opt_params=self.W2.opt_params.value
            )
            self.W3.evolve(
                opt=self.W3.opt,
                w_bound=self.W3.w_bound,
                is_nonnegative=self.W3.is_nonnegative,
                sign_value=self.W3.sign_value,
                prior_type=self.W3.prior_type,
                prior_lmbda=self.W3.prior_lmbda,
                pre_wght=self.W3.pre_wght,
                post_wght=self.W3.post_wght,
                bias_init=self.W3_bias_init,
                pre=self.W3.pre.value,
                post=self.W3.post.value,
                weights=self.W3.weights.value,
                biases=self.W3.biases.value,
                opt_params=self.W3.opt_params.value
            )

        return y_mu, EFE
        

    def save_to_disk(self, params_only=False):
        model_dir = f"{self.exp_dir}/{self.model_name}/custom"
        if params_only:
            os.makedirs(model_dir, exist_ok=True)
            self.W1.save(model_dir)
            self.W2.save(model_dir)
            self.W3.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)

    def load_from_disk(self, model_directory):
        print(f" > Loading model from {model_directory}")
        with Context("Circuit") as self.circuit:
            self.circuit.load_from_dir(model_directory)

# Data preprocessing
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
vocab_size = enc.n_vocab
print(f"vocab size: {vocab_size}")

data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# JAX-based batch function
# JAX-based batch function
def get_batch(split, block_size=8, batch_size=1, key=jax.random.PRNGKey(0)):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    key, subkey = jax.random.split(key)
    ix = jax.random.randint(subkey, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([jnp.array(data[i:i+block_size], dtype=jnp.int32) for i in ix])
    y = jnp.stack([jnp.array(data[i+1:i+1+block_size], dtype=jnp.int32) for i in ix])
    return x, y, key

# Training and evaluation
def train_model(model, num_steps=1000, eval_steps=100, block_size=8, batch_size=1):
    key = jax.random.PRNGKey(0)
    train_losses = []
    val_losses = []

    for step in range(num_steps):
        x, y, key = get_batch('train', block_size=block_size, batch_size=batch_size, key=key)
        y_mu, EFE = model.process(x, y, adapt_synapses=True)
        logits = y_mu
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
        train_losses.append(float(loss))

        if (step + 1) % eval_steps == 0:
            x_val, y_val, key = get_batch('val', block_size=block_size, batch_size=batch_size, key=key)
            y_mu_val, EFE_val = model.process(x_val, y_val, adapt_synapses=False)
            val_loss = optax.softmax_cross_entropy_with_integer_labels(logits=y_mu_val, labels=y_val).mean()
            val_losses.append(float(val_loss))
            print(f"Step {step+1}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}, Val Perplexity = {jnp.exp(val_loss):.2f}")
            model.save_to_disk(params_only=True)

    return train_losses, val_losses

# Generate text
def generate_text(model, enc, prompt, max_length=1000, block_size=8, key=jax.random.PRNGKey(123)):
    # Encode the prompt
    encoded_prompt = enc.encode_ordinary(prompt)
    # Pad or truncate to block_size
    if len(encoded_prompt) < block_size:
        # Pad with a padding token (e.g., 0 or a specific token ID)
        encoded_prompt = encoded_prompt + [0] * (block_size - len(encoded_prompt))
    elif len(encoded_prompt) > block_size:
        # Take the last block_size tokens
        encoded_prompt = encoded_prompt[-block_size:]
    context = jnp.array([encoded_prompt], dtype=jnp.int32)  # Shape: (1, block_size)
    generated = context

    for _ in range(max_length):
        y_mu, _ = model.process(generated, generated, adapt_synapses=False)
        next_token_probs = jax.nn.softmax(y_mu[:, -1, :], axis=-1)
        key, subkey = jax.random.split(key)
        next_token = jax.random.categorical(subkey, next_token_probs, axis=-1)
        generated = jnp.concatenate([generated[:, 1:], next_token[:, None]], axis=1)

    return enc.decode(generated[0].tolist())


# Main execution
if __name__ == "__main__":
    config = {
        'vocab_size': vocab_size,
        'n_embed': 12,
        'block_size': 8,
        'num_heads': 1,
        'num_layers': 1,
        'dropout': 0.1,
        'batch_size': 1,
        'T': 1,
        'eta': 0.001,
        'exp_dir': 'exp',
        'model_name': 'transformer'
    }
    dkey = jax.random.PRNGKey(0)
    model = Transformer(dkey, **config)
    train_losses, val_losses = train_model(model, num_steps=1000, eval_steps=100, block_size=config['block_size'], batch_size=config['batch_size'])
    prompt = "ROMEO:"
    generated_text = generate_text(model, enc, prompt, max_length=1000, block_size=config['block_size'])
    print(f"\nGenerated Text:\n{prompt}{generated_text[len(prompt):]}")