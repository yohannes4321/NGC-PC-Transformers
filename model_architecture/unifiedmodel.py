import jax.numpy as jnp
from flax import nnx
from ngcsimlib.compartment import Compartment
from ngclearn.components.jaxComponent import JaxComponent
import jax
from ngclearn.components import GaussianErrorCell as ErrorCell, HebbianSynapse, StaticSynapse
from ngclearn.utils import JaxProcess
import ngclearn.utils.weight_distribution as dist
from ngcsimlib.context import Context
from jax import random as jax_random
import optax
import os
import tiktoken
import requests
import numpy as np

# ======================
# Dynamical Embedding Layer
# ======================
class DynamicalEmbedding(JaxComponent):
    def __init__(self, name, n_embed, vocab_size, block_size, dropout, batch_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.flat_dim = block_size * n_embed
        self.vocab_size = vocab_size
        self.shape = (batch_size, block_size, n_embed)
        rngs = nnx.Rngs(default=key)
        self.wte = nnx.Embed(vocab_size, n_embed, rngs=rngs)
        self.wpe = nnx.Embed(block_size, n_embed, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.z = Compartment(jnp.zeros(self.shape))  # dynamical state
        self.z_flat = Compartment(jnp.zeros((batch_size, self.flat_dim)))  # <-- new
        self.j = Compartment(jnp.zeros((batch_size, block_size)))

    def advance_state(self, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            B, T = self.j.value.shape
            tok_emb = self.wte(self.j.value)
            pos = jnp.arange(T)[None, :]
            pos_emb = self.wpe(pos)
            drive = self.dropout(tok_emb + pos_emb)
            # Dynamical update: dz/dt = (-z + drive) / tau
            new_z = self.z.value + dt * (-self.z.value + drive) / tau
            self.z.set(new_z)
            self.z_flat.set(new_z.reshape(B, -1))
        return self.z.value

    def reset(self):
        self.z.set(jnp.zeros(self.shape))
        self.j.set(jnp.zeros((self.batch_size, self.block_size)))
        self.z_flat.set(jnp.zeros((self.batch_size,self.flat_dim)))

# ======================
# Dynamical Multi-Head Attention (as a stateful module)
# ======================
class DynamicalAttention(JaxComponent):
    def __init__(self, name, num_heads, head_size, n_embed, dropout, block_size, batch_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.flat_dim = block_size * n_embed
        self.batch_size = batch_size
        self.shape = (batch_size, block_size, n_embed)
        rngs = nnx.Rngs(default=key)
        self.qkv = nnx.Linear(n_embed, 3 * n_embed, rngs=rngs)
        self.proj = nnx.Linear(n_embed, n_embed, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.z = Compartment(jnp.zeros(self.shape))  # state
        self.j = Compartment(jnp.zeros(self.shape))  # bottom-up
        self.j_td = Compartment(jnp.zeros(self.shape))  # top-down error
        self.z_flat = Compartment(jnp.zeros((batch_size, block_size * n_embed)))
        self.j_td_flat = Compartment(jnp.zeros((batch_size, block_size * n_embed)))  # flat error input


    def compute_attn_output(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        scale = 1.0 / jnp.sqrt(self.head_size)
        attn = jnp.einsum('bthd,bThd->bhtT', q, k) * scale
        mask = jnp.tril(jnp.ones((T, T)))
        attn = jnp.where(mask, attn, float('-inf'))
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)
        out = jnp.einsum('bhtT,bThd->bthd', attn, v)
        out = out.reshape(B, T, self.n_embed)
        return self.proj(out)

    def advance_state(self, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            j_td_3d = self.j_td_flat.value.reshape((self.batch_size, self.block_size, self.n_embed))
            drive = self.compute_attn_output(self.j.value) + j_td_3d
            new_z = self.z.value + dt * (-self.z.value + drive) / tau
            self.z.set(new_z)
            self.z_flat.set(new_z.reshape(self.batch_size, -1))
        return self.z.value

    def reset(self):
        self.z.set(jnp.zeros(self.shape))
        self.j.set(jnp.zeros(self.shape))
        self.j_td.set(jnp.zeros(self.shape))
        self.z_flat.set(jnp.zeros((self.batch_size,self.flat_dim)))

# ======================
# Dynamical FeedForward
# ======================
class DynamicalFFN(JaxComponent):
    def __init__(self, name, n_embed, dropout, batch_size, block_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.flat_dim = block_size * n_embed
        self.shape = (batch_size, block_size, n_embed)
        rngs = nnx.Rngs(default=key)
        self.ln = nnx.LayerNorm(n_embed, rngs=rngs)
        self.linear1 = nnx.Linear(n_embed, 4 * n_embed, rngs=rngs)
        self.linear2 = nnx.Linear(4 * n_embed, n_embed, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.z = Compartment(jnp.zeros(self.shape))
        self.j = Compartment(jnp.zeros(self.shape))
        self.j_td = Compartment(jnp.zeros(self.shape))
        self.z_flat = Compartment(jnp.zeros((batch_size, block_size * n_embed)))
        self.j_td_flat = Compartment(jnp.zeros((batch_size, block_size * n_embed)))
         

    def compute_ffn(self, x):
        x = self.ln(x)
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        return self.dropout(x)

    def advance_state(self, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            j_td_3d = self.j_td_flat.value.reshape((self.batch_size, self.block_size, self.n_embed))
            drive = self.compute_ffn(self.j.value) + j_td_3d
            new_z = self.z.value + dt * (-self.z.value + drive) / tau
            self.z.set(new_z)
            self.z_flat.set(new_z.reshape(self.batch_size, -1))
        return self.z.value

    def reset(self):
        self.z.set(jnp.zeros(self.shape))
        self.j.set(jnp.zeros(self.shape))
        self.j_td.set(jnp.zeros(self.shape))
        self.z_flat.set(jnp.zeros((self.batch_size,self.flat_dim)))

# ======================
# Dynamical Output Layer (predicts EMBEDDINGS, not logits)
# ======================
class DynamicalOutput(JaxComponent):
    def __init__(self, name, n_embed, out_embed_dim, batch_size, block_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.flat_dim = block_size * n_embed
        self.out_dim = out_embed_dim
        self.shape = (batch_size, block_size, out_embed_dim)
        rngs = nnx.Rngs(default=key)
        self.ln = nnx.LayerNorm(n_embed, rngs=rngs)
        self.head = nnx.Linear(n_embed, out_embed_dim, rngs=rngs)
        self.z = Compartment(jnp.zeros(self.shape))
        self.j = Compartment(jnp.zeros((batch_size, block_size, n_embed)))
        self.j_td = Compartment(jnp.zeros(self.shape))
        self.z_flat = Compartment(jnp.zeros((batch_size, block_size * n_embed)))

    def compute_output(self, x):
        x = self.ln(x)
        return self.head(x)

    def advance_state(self, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            drive = self.compute_output(self.j.value) + self.j_td.value
            new_z = self.z.value + dt * (-self.z.value + drive) / tau
            self.z.set(new_z)
            self.z_flat.set(new_z.reshape(self.batch_size, -1))
        return self.z.value

    def reset(self):
        self.z.set(jnp.zeros(self.shape))
        self.z_flat.set(jnp.zeros((self.batch_size,self.flat_dim)))
        self.j.set(jnp.zeros((self.batch_size, self.block_size, self.n_embed)))
        self.j_td.set(jnp.zeros(self.shape))

# ======================
# Full PC-Transformer
# ======================
class PC_Transformer:
    def __init__(self, dkey, vocab_size=50257, n_embed=64, block_size=8,
                 num_heads=2, T=20, eta=0.001, batch_size=1, dropout=0.1,
                 exp_dir="exp", model_name="pc_transformer"):
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.batch_size = batch_size
        self.T = T
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)

        dkey, *subkeys = jax.random.split(dkey, 8)
        flat_dim = block_size * n_embed
        out_flat_dim = block_size * n_embed  # predict next-token embeddings

        with Context("Circuit") as self.circuit:
            self.emb = DynamicalEmbedding("emb", n_embed, vocab_size, block_size, dropout, batch_size, key=subkeys[0])
            self.attn = DynamicalAttention("attn", num_heads, n_embed // num_heads, n_embed, dropout, block_size, batch_size, key=subkeys[1])
            self.ffn = DynamicalFFN("ffn", n_embed, dropout, batch_size, block_size, key=subkeys[2])
            self.out = DynamicalOutput("out", n_embed, n_embed, batch_size, block_size, key=subkeys[3])

            # Error cells (flat)
            self.e1 = ErrorCell("e1", n_units=flat_dim)
            self.e2 = ErrorCell("e2", n_units=flat_dim)
            self.e3 = ErrorCell("e3", n_units=out_flat_dim)

            # Synapses
            self.W1 = HebbianSynapse("W1", (flat_dim, flat_dim), eta=eta, weight_init=dist.uniform(-0.3,0.3), bias_init=dist.constant(0.), optim_type="adam", sign_value=-1., key=subkeys[4])
            self.W2 = HebbianSynapse("W2", (flat_dim, flat_dim), eta=eta, weight_init=dist.uniform(-0.3,0.3), bias_init=dist.constant(0.), optim_type="adam", sign_value=-1., key=subkeys[5])
            self.W3 = HebbianSynapse("W3", (flat_dim, out_flat_dim), eta=eta, weight_init=dist.uniform(-0.3,0.3), bias_init=dist.constant(0.), optim_type="adam", sign_value=-1., key=subkeys[6])

            self.E2 = StaticSynapse("E2", (flat_dim, flat_dim), weight_init=dist.uniform(-0.3,0.3), key=subkeys[4])
            self.E3 = StaticSynapse("E3", (out_flat_dim, flat_dim), weight_init=dist.uniform(-0.3,0.3), key=subkeys[5])

            # Wiring
            self.W1.inputs << self.emb.z_flat
            self.e1.mu << self.W1.outputs
            self.e1.target << self.attn.z_flat
            self.W2.inputs << self.attn.z_flat
            self.e2.mu << self.W2.outputs
            self.e2.target << self.ffn.z_flat
            self.W3.inputs << self.ffn.z_flat
            self.e3.mu << self.W3.outputs
            self.e3.target << self.out.z_flat

            # Feedback (reshape flat → 3D)
            self.E2.inputs << self.e2.dmu
            self.attn.j << self.emb.z_flat  # bottom-up from emb
            self.attn.j_td_flat << self.e1.dtarget

            self.E3.inputs << self.e3.dmu
            self.ffn.j << self.attn.z
            self.ffn.j_td_flat << self.e2.dtarget

            self.out.j << self.ffn.z
            # out.j_td not used (top layer)

            # Hebbian signals
            self.W1.pre << self.emb.z_flat
            self.W1.post << self.e1.dmu
            self.W2.pre << self.attn.z_flat
            self.W2.post << self.e2.dmu
            self.W3.pre << self.ffn.z_flat
            self.W3.post << self.e3.dmu

        # Store embedding table for targets & decoding
        rngs = nnx.Rngs(0)
        self.embed_table = nnx.Embed(vocab_size, n_embed, rngs=rngs).embedding.value

    def process(self, obs, lab, adapt_synapses=True):
        B, T = obs.shape
        # Reset
        for comp in [self.emb, self.attn, self.ffn, self.out]:
            comp.reset()
        for e in [self.e1, self.e2, self.e3]:
           e.reset(batch_size=B, shape=(e.n_units,), sigma_shape=e.sigma_shape)

        # Get continuous targets: next-token embeddings
        x_emb = self.embed_table[obs]   # (B, T, D)
        y_emb = self.embed_table[lab]   # (B, T, D)

        # P-step: initialize states with feedforward pass (no feedback)
        self.emb.j.set(obs)
        self.emb.advance_state(tau=0.1)  # fast
        self.attn.j.set(self.emb.z.value)
        self.attn.advance_state(tau=0.1)
        self.ffn.j.set(self.attn.z.value)
        self.ffn.advance_state(tau=0.1)
        self.out.j.set(self.ffn.z.value)
        self.out.advance_state(tau=0.1)

        # Tie feedback weights
        self.E2.weights.set(self.W2.weights.value.T)
        self.E3.weights.set(self.W3.weights.value.T)

        # E-step: run dynamics with error feedback
        for ts in range(self.T):
            self.emb.j.set(obs)
            self.emb.advance_state()

            self.attn.j.set(self.emb.z.value)
            self.attn.j_td.set(self.e1.dtarget.value.reshape((B, T, self.n_embed)))
            self.attn.advance_state()

            self.ffn.j.set(self.attn.z.value)
            self.ffn.j_td.set(self.e2.dtarget.value.reshape((B, T, self.n_embed)))
            self.ffn.advance_state()

            self.out.j.set(self.ffn.z.value)
            self.out.advance_state()

            # Update synapses
            emb_flat = self.emb.z_flat.value   # ← (1, 96)
            attn_flat = self.attn.z_flat.value # ← (1, 96)
            ffn_flat = self.ffn.z_flat.value   # ← (1, 96)

            # self.W1.inputs.set(emb_flat)
            # self.W1.advance_state()
            # self.e1.mu.set(self.W1.outputs.value)
            # self.e1.target.set(attn_flat)
            # self.e1.advance_state(dt=1.mu,target,sigma,modulator=1,mask=1)

            # self.W2.inputs.set(attn_flat)
            # self.W2.advance_state()
            # self.e2.mu.set(self.W2.outputs.value)
            # self.e2.target.set(ffn_flat)
            # self.e2.advance_state(dt=1.mu,target,sigma,modulator=1,mask=1)

            # self.W3.inputs.set(ffn_flat)
            # self.W3.advance_state()
            # self.e3.mu.set(self.W3.outputs.value)
            # self.e3.target.set(y_emb.reshape(B, -1))  # CONTINUOUS TARGET!
            # self.e3.advance_state(dt=1.mu,target,sigma,modulator=1,mask=1)
            W1_out = emb_flat @ self.W1.weights.value + self.W1.biases.value
            self.e1.mu.set(W1_out)
            self.e1.target.set(attn_flat)
            # For e1
            sigma = 1.0  # fixed variance

            # e1
            error1 = self.e1.target.value - self.e1.mu.value
            dmu1 = error1 / sigma
            self.e1.dmu.set(dmu1)
            self.e1.dtarget.set(-dmu1)
            self.e1.L.set(-0.5 * jnp.sum(error1 ** 2) / sigma)
       
            W2_out = attn_flat @ self.W2.weights.value + self.W2.biases.value
            self.e2.mu.set(W2_out)
            self.e2.target.set(ffn_flat)
             # e2
            error2 = self.e2.target.value - self.e2.mu.value
            dmu2 = error2 / sigma
            self.e2.dmu.set(dmu2)
            self.e2.dtarget.set(-dmu2)
            self.e2.L.set(-0.5 * jnp.sum(error2 ** 2) / sigma)

            W3_out = ffn_flat @ self.W3.weights.value + self.W3.biases.value
            self.e3.mu.set(W3_out)
            self.e3.target.set(y_emb.reshape(B, -1))
             # e3
            error3 = self.e3.target.value - self.e3.mu.value
            dmu3 = error3 / sigma
            self.e3.dmu.set(dmu3)
            self.e3.dtarget.set(-dmu3)
            self.e3.L.set(-0.5 * jnp.sum(error3 ** 2) / sigma)
        bias_init = dist.constant(value=0.)
      
        # Evolve synapses
        if adapt_synapses:
            for W in [self.W1, self.W2, self.W3]:
                W.evolve(opt=W.opt, w_bound=0., is_nonnegative=False, sign_value=W.sign_value,
                         prior_type="none",prior_lmbda=W.prior_lmbda, pre_wght=W.pre_wght,
                post_wght=W.post_wght, bias_init=W.bias_init, pre=W.pre.value, post=W.post.value,
                         weights=W.weights.value, biases=W.biases.value, opt_params=W.opt_params.value)

        # Return predicted embeddings
        return self.out.z.value, (self.e1.L.value + self.e2.L.value + self.e3.L.value)

    def save_to_disk(self, params_only=False):
        if params_only:
            model_dir = f"{self.exp_dir}/custom"
            os.makedirs(model_dir, exist_ok=True)
            for W in [self.W1, self.W2, self.W3]:
                W.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name="pc_transformer")

# ======================
# Data & Training (unchanged, but use embedding targets)
# ======================

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

def get_batch(split, block_size=8, batch_size=1, key=jax.random.PRNGKey(0)):
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "data")
    data_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    key, subkey = jax.random.split(key)
    ix = jax.random.randint(subkey, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([jnp.array(data[i:i+block_size], dtype=jnp.int32) for i in ix])
    y = jnp.stack([jnp.array(data[i+1:i+1+block_size], dtype=jnp.int32) for i in ix])
    return x, y, key

def train_model(model, num_steps=1000, eval_steps=100, block_size=8, batch_size=1):
    key = jax.random.PRNGKey(0)
    for step in range(num_steps):
        x, y, key = get_batch('train', block_size, batch_size, key)
        pred_emb, efe = model.process(x, y, adapt_synapses=True)

        # Compute loss for monitoring: nearest-neighbor decoding
        logits = jnp.einsum('btd,vd->btv', pred_emb, model.embed_table)  # (B,T,V)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()

        if (step + 1) % eval_steps == 0:
            x_val, y_val, key = get_batch('val', block_size, batch_size, key)
            pred_emb_val, _ = model.process(x_val, y_val, adapt_synapses=False)
            logits_val = jnp.einsum('btd,vd->btv', pred_emb_val, model.embed_table)
            val_loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits_val, labels=y_val).mean()
            print(f"Step {step+1}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}, Perplexity = {jnp.exp(val_loss):.2f}")

    return loss, val_loss

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
        'vocab_size': 50257,
        'n_embed': 128,
        'block_size':8 ,
        'num_heads': 1,
        # 'num_layers': 1,
        'dropout': 0.1,
        'batch_size': 1,
        'T': 10,
        'eta': 0.001,
        'exp_dir': 'exp',
        'model_name': 'transformer'
    }
    dkey = jax.random.PRNGKey(0)
    model = PC_Transformer(dkey, **config)
    train_losses, val_losses = train_model(model, num_steps=1000, eval_steps=100, block_size=config['block_size'], batch_size=config['batch_size'])
    prompt = "Romeo"
    generated_text = generate_text(model, enc, prompt, max_length=100, block_size=config['block_size'])
    print(f"\nGenerated Text:\n{prompt}{generated_text[len(prompt):]}")
    # print(f"W1.dWeights mean: {model.W1.dWeights.value.mean()}, var: {model.W1.dWeights.value.var()}")
    # print(f"W1.weights mean: {model.W1.weights.value.mean()}, var: {model.W1.weights.value.var()}")

   