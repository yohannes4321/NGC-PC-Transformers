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
from ngclearn.utils.model_utils import drop_out,softmax,gelu,layer_normalize
import optax
import os
import tiktoken
import requests
import numpy as np
from functools import partial


def masked_fill(x: jax.Array, mask: jax.Array, value=0) -> jax.Array:
    return jnp.where(mask, jnp.broadcast_to(value, x.shape), x)


@partial(jax.jit, static_argnums=(5, 6))
def cross_attention(dkey, params, x1, x2, mask, n_heads, dropout_rate):
    B, T, Dq = x1.shape
    _, S, Dkv = x2.shape
    Wq, bq, Wk, bk, Wv, bv, Wout, bout = params
    x1 = jnp.clip(x1, -1e4, 1e4)  # Clip input
    x2 = jnp.clip(x2, -1e4, 1e4)
    q = jnp.clip(x1 @ Wq + bq, -1e4, 1e4)
    k = jnp.clip(x2 @ Wk + bk, -1e4, 1e4)
    v = x2 @ Wv + bv
    hidden = q.shape[-1]
    _hidden = hidden // n_heads
    q = q.reshape((B, T, n_heads, _hidden)).transpose([0, 2, 1, 3])
    k = k.reshape((B, S, n_heads, _hidden)).transpose([0, 2, 1, 3])
    v = v.reshape((B, S, n_heads, _hidden)).transpose([0, 2, 1, 3])
    score = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(_hidden)
    score = jnp.clip(score, -1e4, 1e4)  # Clip scores
    if mask is not None:
        _mask = mask.reshape((B, 1, T, S))
        score = masked_fill(score, _mask, value=-jnp.inf)
    score = score - jnp.max(score, axis=-1, keepdims=True)  # Stable softmax
    score = jax.nn.softmax(score, axis=-1)
    score = score.astype(q.dtype)
    if dropout_rate > 0.:
        score, _ = drop_out(dkey, score, rate=dropout_rate)
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, T, -1))
    return jnp.clip(attention @ Wout + bout, -1e4, 1e4)
# core components
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
        self.z = Compartment(jnp.zeros(self.shape))
        self.z_flat = Compartment(jnp.zeros((batch_size, self.flat_dim)))
        self.j = Compartment(jnp.zeros((batch_size, block_size)))

    def advance_state(self, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            B, T = self.j.value.shape
            tok_emb = self.wte(self.j.value)
            pos = jnp.arange(T)[None, :]
            pos_emb = self.wpe(pos)
            drive = self.dropout(tok_emb + pos_emb)
            new_z = self.z.value + dt * (-self.z.value + drive) / tau
            self.z.set(jnp.clip(new_z, -1e4, 1e4))
            self.z_flat.set(new_z.reshape(B, -1))
        return self.z.value

    def reset(self):
        self.z.set(jnp.zeros(self.shape))
        self.j.set(jnp.zeros((self.batch_size, self.block_size)))
        self.z_flat.set(jnp.zeros((self.batch_size, self.flat_dim)))


class PC_Attention(JaxComponent):
    def __init__(self, name, num_heads, head_size, n_embed, dropout, block_size, batch_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.batch_size = batch_size
        self.shape = (batch_size, block_size, n_embed)
        self.flat_dim = block_size * n_embed
        self.dropout_rate = dropout
        assert n_embed == num_heads * head_size, "n_embed must equal num_heads * head_size"

        keys = jax_random.split(key, 5)

        # Learnable projections via Hebbian synapses
        init = dist.uniform(-0.1, 0.1)
        self.Wq = HebbianSynapse(f"{name}_Wq", (n_embed, n_embed), eta=0.001,
                                weight_init=init, bias_init=dist.constant(0.1),
                                optim_type="adam", sign_value=1.0, key=keys[0])
        self.Wk = HebbianSynapse(f"{name}_Wk", (n_embed, n_embed), eta=0.001,
                                weight_init=init, bias_init=dist.constant(0.1),
                                optim_type="adam", sign_value=1.0, key=keys[1])
        self.Wv = HebbianSynapse(f"{name}_Wv", (n_embed, n_embed), eta=0.001,
                                weight_init=init, bias_init=dist.constant(0.1),
                                optim_type="adam", sign_value=1.0, key=keys[2])
        self.Wout = HebbianSynapse(f"{name}_Wout", (n_embed, n_embed), eta=0.001,
                                  weight_init=init, bias_init=dist.constant(0.1),
                                  optim_type="adam", sign_value=1.0, key=keys[3])
       
        # Causal mask
        self.causal_mask = jnp.tril(jnp.ones((block_size, block_size), dtype=bool))
        self.rng_key = keys[4]

        # Compartments
        self.z = Compartment(jnp.zeros(self.shape))
        self.j = Compartment(jnp.zeros(self.shape))
        self.j_td = Compartment(jnp.zeros(self.shape))
        self.z_flat = Compartment(jnp.zeros((batch_size, self.flat_dim)))
        self.j_td_flat = Compartment(jnp.zeros((batch_size, self.flat_dim)))

    def compute_attn_output(self, x):
        B, T, D = x.shape
        mask = jnp.broadcast_to(self.causal_mask[:T, :T], (B, T, T))
        params = (
            self.Wq.weights.value, self.Wq.biases.value,
            self.Wk.weights.value, self.Wk.biases.value,
            self.Wv.weights.value, self.Wv.biases.value,
            self.Wout.weights.value, self.Wout.biases.value
        )
       
        #print(f"{self.name} j max:", float(jnp.max(jnp.abs(x))), "isnan:", bool(jnp.any(jnp.isnan(x))))
        #print(f"{self.name} Wq max:", float(jnp.max(jnp.abs(self.Wq.weights.value))), "isnan:", bool(jnp.any(jnp.isnan(self.Wq.weights.value))))
        x = jnp.clip(x, -1e2, 1e2)  # Tighter clipping
        q = jnp.clip(x @ self.Wq.weights.value + self.Wq.biases.value, -1e2, 1e2)
        k = jnp.clip(x @ self.Wk.weights.value + self.Wk.biases.value, -1e2, 1e2)
        v = jnp.clip(x @ self.Wv.weights.value + self.Wv.biases.value, -1e2, 1e2)
        #print(f"{self.name} q max:", float(jnp.max(jnp.abs(q))), "isnan:", bool(jnp.any(jnp.isnan(q))))
        #print(f"{self.name} k max:", float(jnp.max(jnp.abs(k))), "isnan:", bool(jnp.any(jnp.isnan(k))))
        hidden = q.shape[-1]
        _hidden = hidden // self.num_heads
        q = q.reshape((B, T, self.num_heads, _hidden)).transpose([0, 2, 1, 3])
        k = k.reshape((B, T, self.num_heads, _hidden)).transpose([0, 2, 1, 3])
        v = v.reshape((B, T, self.num_heads, _hidden)).transpose([0, 2, 1, 3])
        score = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(_hidden)
        score = jnp.clip(score, -50, 50)
        #print(f"{self.name} score max:", float(jnp.max(jnp.abs(score))), "isnan:", bool(jnp.any(jnp.isnan(score))))
        # Debug score distribution
        score_flat = score.flatten()
        #print(f"{self.name} score histogram:", jnp.histogram(score_flat, bins=10, range=(-10, 10))[0].tolist())
        if mask is not None:
            _mask = mask.reshape((B, 1, T, T))
            score = jnp.where(_mask, score, -1e4)  # Use -1e9 instead of -inf
        score = score - jnp.max(score, axis=-1, keepdims=True, where=~jnp.isinf(score),initial=-1e4)  # Stable softmax
        score = jnp.where(jnp.isinf(score), -1e4, score)  # Replace remaining -inf
        score = jax.nn.softmax(score, axis=-1)
        score = jnp.where(jnp.isnan(score), 0.0, score)  # Handle nan in softmax
        #print(f"{self.name} softmax score max:", float(jnp.max(jnp.abs(score))), "isnan:", bool(jnp.any(jnp.isnan(score))))
        if self.dropout_rate > 0.:
            score, _ = drop_out(self.rng_key, score, rate=self.dropout_rate)
        attention = jnp.einsum("BHTS,BHSE->BHTE", score, v)
        attention = attention.transpose([0, 2, 1, 3]).reshape((B, T, -1))
        output = jnp.clip(attention @ self.Wout.weights.value + self.Wout.biases.value, -1e2, 1e2)
        #print(f"{self.name} output max:", float(jnp.max(jnp.abs(output))), "isnan:", bool(jnp.any(jnp.isnan(output))))
        return output

    def advance_state(self, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            
            j_td_3d = self.j_td_flat.value.reshape((self.batch_size, self.block_size, self.n_embed))
            #print(f"{self.name} j_td max:", float(jnp.max(jnp.abs(j_td_3d))), "isnan:", bool(jnp.any(jnp.isnan(j_td_3d))))
            drive = jnp.clip(self.compute_attn_output(self.j.value) + j_td_3d, -1e2, 1e2)
            #print(f"{self.name} drive max:", float(jnp.max(jnp.abs(drive))), "isnan:", bool(jnp.any(jnp.isnan(drive))))
            new_z = self.z.value + dt * (-self.z.value + drive) / tau
            self.z.set(jnp.clip(new_z, -1e2, 1e2))
            self.z_flat.set(new_z.reshape(self.batch_size, -1))
            #print(f"{self.name} z max:", float(jnp.max(jnp.abs(self.z.value))), "isnan:", bool(jnp.any(jnp.isnan(self.z.value))))
        return self.z.value

    def reset(self):
        self.z.set(jnp.zeros(self.shape))
        self.j.set(jnp.zeros(self.shape))
        self.j_td.set(jnp.zeros(self.shape))
        self.z_flat.set(jnp.zeros((self.batch_size, self.flat_dim)))


class DynamicalFFN(JaxComponent):
    def __init__(self, name, n_embed, dropout, batch_size, block_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.flat_dim = block_size * n_embed
        self.shape = (batch_size, block_size, n_embed)
        rngs = nnx.Rngs(default=key)
        self.ln = nnx.LayerNorm(n_embed, rngs=rngs,epsilon=1e-4)
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
        x = jax.nn.gelu(self.linear1(x))
        x = self.linear2(x)
        return self.dropout(x)

    def advance_state(self, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            j_td_3d = self.j_td_flat.value.reshape((self.batch_size, self.block_size, self.n_embed))
            drive = self.compute_ffn(self.j.value) + j_td_3d
            new_z = self.z.value + dt * (-self.z.value + drive) / tau
            self.z.set(jnp.clip(new_z, -1e4, 1e4))
            self.z_flat.set(new_z.reshape(self.batch_size, -1))
        return self.z.value

    def reset(self):
        self.z.set(jnp.zeros(self.shape))
        self.j.set(jnp.zeros(self.shape))
        self.j_td.set(jnp.zeros(self.shape))
        self.z_flat.set(jnp.zeros((self.batch_size, self.flat_dim)))


class DynamicalOutput(JaxComponent):
    def __init__(self, name, n_embed, vocab_size, batch_size, block_size, key, embed_table, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.vocab_size = vocab_size
        self.shape = (batch_size, block_size, vocab_size)
        rngs = nnx.Rngs(default=key)
        self.embed_table = embed_table
        self.ln = nnx.LayerNorm(n_embed, rngs=rngs,epsilon=1e-4)
        self.z = Compartment(jnp.zeros(self.shape))
        self.j = Compartment(jnp.zeros((batch_size, block_size, n_embed)))
        self.j_td = Compartment(jnp.zeros(self.shape))
        self.z_flat = Compartment(jnp.zeros((batch_size, block_size * vocab_size)))

    def compute_output(self, x):
        x = self.ln(x)
        return x @ self.embed_table.T

    def advance_state(self, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            drive = self.compute_output(self.j.value) + self.j_td.value
            new_z = self.z.value + dt * (-self.z.value + drive) / tau
            self.z.set(jnp.clip(new_z, -1e4, 1e4))
            self.z_flat.set(new_z.reshape(self.batch_size, -1))
        return self.z.value

    def reset(self):
        self.z.set(jnp.zeros(self.shape))
        self.z_flat.set(jnp.zeros((self.batch_size, self.block_size * vocab_size)))
        self.j.set(jnp.zeros((self.batch_size, self.block_size, self.n_embed)))
        self.j_td.set(jnp.zeros(self.shape))


class DynamicalTransformerBlock(JaxComponent):
    def __init__(self, name, n_embed, num_heads, head_size, dropout, block_size, batch_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.batch_size = batch_size
        self.shape = (batch_size, block_size, n_embed)
        self.flat_dim = block_size * n_embed

        keys = jax_random.split(key, 2)

        self.attn = PC_Attention(
            f"{name}_attn", num_heads, head_size, n_embed, dropout,
            block_size, batch_size, key=keys[0]
        )
        self.ffn = DynamicalFFN(
            f"{name}_ffn", n_embed, dropout, batch_size, block_size, key=keys[1]
        )

        self.j = Compartment(jnp.zeros(self.shape))
        self.j_td = Compartment(jnp.zeros(self.shape))
        self.z = Compartment(jnp.zeros(self.shape))
        self.z_flat = Compartment(jnp.zeros((batch_size, self.flat_dim)))
        self.j_td_flat = Compartment(jnp.zeros((batch_size, self.flat_dim)))

    def advance_state(self, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            j_td_3d = self.j_td_flat.value.reshape((self.batch_size, self.block_size, self.n_embed))
            self.attn.j.set(self.j.value)
            self.attn.j_td.set(j_td_3d)
            attn_out = self.attn.advance_state(t=t, dt=dt, tau=tau)

            self.ffn.j.set(attn_out)
            ffn_out = self.ffn.advance_state(t=t, dt=dt, tau=tau)

            new_z = self.z.value + dt * (-self.z.value + ffn_out) / tau
            self.z.set(jnp.clip(new_z, -1e4, 1e4))
            self.z_flat.set(new_z.reshape(self.batch_size, -1))
        return self.z.value

    def reset(self):
        self.j.set(jnp.zeros(self.shape))
        self.j_td.set(jnp.zeros(self.shape))
        self.z.set(jnp.zeros(self.shape))
        self.z_flat.set(jnp.zeros((self.batch_size, self.flat_dim)))
        self.attn.reset()
        self.ffn.reset()


class PC_Transformer:
    def __init__(self, dkey, vocab_size=50257, n_embed=64, block_size=128,
                 num_heads=2, num_layers=2, T=20, eta=0.001, batch_size=1, dropout=0.1,
                 exp_dir="exp", model_name="pc_transformer"):
        assert n_embed % num_heads == 0, "n_embed must be divisible by num_heads"
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.T = T
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)

        total_keys = 1 + 2 * num_layers + 1 + (num_layers + 1) + num_layers + 4 * num_layers
        dkey, *subkeys = jax.random.split(dkey, total_keys)

        flat_dim = block_size * n_embed
        out_flat_dim = block_size * vocab_size

        with Context("Circuit") as self.circuit:
            self.emb = DynamicalEmbedding("emb", n_embed, vocab_size, block_size, dropout, batch_size, key=subkeys[0])
            self.embed_table = self.emb.wte.embedding.value

            self.blocks = []
            key_idx = 1
            for i in range(num_layers):
                block = DynamicalTransformerBlock(
                    name=f"block_{i}",
                    n_embed=n_embed,
                    num_heads=num_heads,
                    head_size=n_embed // num_heads,
                    dropout=dropout,
                    block_size=block_size,
                    batch_size=batch_size,
                    key=subkeys[key_idx]
                )
                self.blocks.append(block)
                key_idx += 1

            self.out = DynamicalOutput("out", n_embed, vocab_size, batch_size, block_size,
                                       key=subkeys[key_idx], embed_table=self.embed_table)
            key_idx += 1

            self.error_cells = []
            for i in range(num_layers):
                e = ErrorCell(f"e{i+1}", n_units=flat_dim, batch_size=batch_size)
                self.error_cells.append(e)
            self.e_out = ErrorCell("e_out", n_units=out_flat_dim, batch_size=batch_size)

            self.synapses = []
          
            W0 = HebbianSynapse("W0", (flat_dim, flat_dim), eta=eta,
                    weight_init=dist.uniform(-0.1,0.1), bias_init=dist.constant(0.01),
                    optim_type="adam", sign_value=-1., key=subkeys[key_idx])
            
            
            self.synapses.append(W0)
            key_idx += 1
            for i in range(num_layers - 1):
                W = HebbianSynapse(f"W{i+1}", (flat_dim, flat_dim), eta=eta,
                                   weight_init=dist.uniform(-0.1, 0.1), bias_init=dist.constant(0.1),
                                   optim_type="adam", sign_value=-1., key=subkeys[key_idx])
                self.synapses.append(W)
                key_idx += 1
           
            W_last = HebbianSynapse(f"W{num_layers}", (flat_dim, out_flat_dim), eta=eta,
                        weight_init=dist.uniform(-0.1,0.1), bias_init=dist.constant(0.01),
                        optim_type="adam", sign_value=-1., key=subkeys[key_idx])
            self.synapses.append(W_last)
            key_idx += 1

            self.fb_syns = []
            for i in range(num_layers):
                fb = StaticSynapse(f"fb_{i}", (flat_dim, flat_dim),
                                   weight_init=dist.uniform(-0.1, 0.1), key=subkeys[key_idx])
                self.fb_syns.append(fb)
                key_idx += 1

            # Forward wiring
            self.synapses[0].inputs << self.emb.z_flat
            self.error_cells[0].mu << self.synapses[0].outputs
            self.error_cells[0].target << self.blocks[0].z_flat

            for i in range(1, num_layers):
                self.synapses[i].inputs << self.blocks[i-1].z_flat
                self.error_cells[i].mu << self.synapses[i].outputs
                self.error_cells[i].target << self.blocks[i].z_flat

            self.synapses[-1].inputs << self.blocks[-1].z_flat
            self.e_out.mu << self.synapses[-1].outputs
            self.e_out.target << self.out.z_flat

            for i in range(num_layers):
                self.blocks[i].j_td_flat << self.error_cells[i].dtarget

            self.out.j << self.blocks[-1].z

            self.synapses[0].pre << self.emb.z_flat
            self.synapses[0].post << self.error_cells[0].dmu
            for i in range(1, num_layers):
                self.synapses[i].pre << self.blocks[i-1].z_flat
                self.synapses[i].post << self.error_cells[i].dmu
            self.synapses[-1].pre << self.blocks[-1].z_flat
            self.synapses[-1].post << self.e_out.dmu

        # Collect all learnable synapses (including attention)
        self.all_synapses = self.synapses.copy()
        for block in self.blocks:
            self.all_synapses.extend([block.attn.Wq, block.attn.Wk, block.attn.Wv, block.attn.Wout])

        self.config = {
            'vocab_size': vocab_size, 'n_embed': n_embed, 'block_size': block_size,
            'num_heads': num_heads, 'num_layers': num_layers, 'dropout': dropout
        }

    def process(self, obs, lab, adapt_synapses=True):
        B, T = obs.shape
        self.emb.reset()
        for block in self.blocks:
            block.reset()
        self.out.reset()

        if adapt_synapses:
            for e in self.error_cells:
                e.reset(batch_size=B, shape=(e.n_units,), sigma_shape=e.sigma_shape)
            self.e_out.reset(batch_size=B, shape=(self.e_out.n_units,), sigma_shape=self.e_out.sigma_shape)

        # Continuous targets: next-token embeddings
        y_emb = self.embed_table[lab]
        # Clamp output to true targets during E-step
        target_logits = y_emb @ self.embed_table.T
        self.out.z.set(target_logits)

        # P-step
        self.emb.j.set(obs)
        self.emb.advance_state(tau=0.1)
        x = self.emb.z.value
        for block in self.blocks:
            block.j.set(x)
            block.advance_state(tau=0.1)
            x = block.z.value
        self.out.j.set(x)
        self.out.advance_state(tau=0.1)

        if not adapt_synapses:
            return self.out.z.value, 0.0

        # E-step
        for ts in range(self.T):
            self.emb.j.set(obs)
            self.emb.advance_state()
            x = self.emb.z.value
            for i, block in enumerate(self.blocks):
                block.j.set(x)
                block.advance_state()
                x = block.z.value
            self.out.j.set(x)
            self.out.advance_state()

            sigma = 1.0
            # Layer 0
            mu0 = self.emb.z_flat.value @ self.synapses[0].weights.value + self.synapses[0].biases.value
            target0 = self.blocks[0].z_flat.value
            err0 = target0 - mu0
            self.error_cells[0].mu.set(mu0)
            self.error_cells[0].target.set(target0)
            self.error_cells[0].dmu.set(err0 / sigma)
            self.error_cells[0].dtarget.set(-err0 / sigma)
            self.error_cells[0].L.set(-0.5 * jnp.sum(err0 ** 2) / sigma)

            for i in range(1, self.num_layers):
                mu_i = self.blocks[i-1].z_flat.value @ self.synapses[i].weights.value + self.synapses[i].biases.value
                target_i = self.blocks[i].z_flat.value
                err_i = target_i - mu_i
                self.error_cells[i].mu.set(mu_i)
                self.error_cells[i].target.set(target_i)
                self.error_cells[i].dmu.set(err_i / sigma)
                self.error_cells[i].dtarget.set(-err_i / sigma)
                self.error_cells[i].L.set(-0.5 * jnp.sum(err_i ** 2) / sigma)

            mu_out = self.blocks[-1].z_flat.value @ self.synapses[-1].weights.value + self.synapses[-1].biases.value
            target_out = self.out.z_flat.value
            err_out = target_out - mu_out
            self.e_out.mu.set(mu_out)
            self.e_out.target.set(target_out)
            self.e_out.dmu.set(err_out / sigma)
            self.e_out.dtarget.set(-err_out / sigma)
            self.e_out.L.set(-0.5 * jnp.sum(err_out ** 2) / sigma)

        # Evolve ALL synapses
        if adapt_synapses:
            for W in self.all_synapses:
                W.evolve(
                    opt=W.opt, w_bound=0., is_nonnegative=False, sign_value=W.sign_value,
                    prior_type="none", prior_lmbda=W.prior_lmbda, pre_wght=W.pre_wght,
                    post_wght=W.post_wght, bias_init=W.bias_init,
                    pre=W.pre.value, post=W.post.value,
                    weights=W.weights.value, biases=W.biases.value,
                    opt_params=W.opt_params.value
                )

        total_loss = sum(e.L.value for e in self.error_cells) + self.e_out.L.value
        return target_logits, total_loss  # return clamped logits for loss

    def save_to_disk(self, params_only=False):
        if params_only:
            model_dir = f"{self.exp_dir}/custom"
            os.makedirs(model_dir, exist_ok=True)
            for W in self.all_synapses:
                W.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name="pc_transformer")


#data loading
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt  '
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


def get_batch(split, block_size=128, batch_size=1, key=jax.random.PRNGKey(0)):
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "data")
    data_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    key, subkey = jax.random.split(key)
    ix = jax.random.randint(subkey, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([jnp.array(data[i:i+block_size], dtype=jnp.int32) for i in ix])
    y = jnp.stack([jnp.array(data[i+1:i+1+block_size], dtype=jnp.int32) for i in ix])
    return x, y, key


def train_model(model, num_steps=1000, eval_steps=100, block_size=128, batch_size=1):
    key = jax.random.PRNGKey(0)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config['eta']))
    for step in range(num_steps):
        x, y, key = get_batch('train', block_size, batch_size, key)
        logits, efe = model.process(x, y, adapt_synapses=True)
        train_loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
        # if jnp.isnan(train_loss):
        #     #print(f"NaN loss at step {step+1}")
        #     break
        if (step + 1) % eval_steps == 0:
            x_val, y_val, key = get_batch('val', block_size, batch_size, key)
            logits_val, _ = model.process(x_val, y_val, adapt_synapses=False)
            val_loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits_val, labels=y_val).mean()
            print(f"Step {step+1}: Train Loss = {train_loss:.4f}, Perplexity = {jnp.exp(train_loss):.2f},EFE = {efe:.4f} ")
    return train_loss, val_loss


def generate_text(model, enc, prompt, max_length=1000, block_size=8, key=jax.random.PRNGKey(123)):
    encoded_prompt = enc.encode_ordinary(prompt)
    encoded_prompt = encoded_prompt[-block_size:] if len(encoded_prompt) > block_size else encoded_prompt + [50256] * (block_size - len(encoded_prompt))
    context = jnp.array([encoded_prompt], dtype=jnp.int32)
    generated = context

    for i in range(max_length):
        x_dummy = generated
        logits, _ = model.process(generated, x_dummy, adapt_synapses=False)
        next_token_logits = logits[0, -1, :] / 1.0
      
        # Top-p sampling
        sorted_logits, sorted_indices = jax.lax.sort_key_val(next_token_logits, jnp.arange(next_token_logits.shape[0]))
        sorted_probs = jax.nn.softmax(sorted_logits)
        cumsum_probs = jnp.cumsum(sorted_probs[::-1])[::-1]
        mask = cumsum_probs > 0.9
        top_p_logits = jnp.where(mask, sorted_logits, -jnp.inf)
        probs = jax.nn.softmax(top_p_logits)
        key, subkey = jax.random.split(key)
        next_token = sorted_indices[jax.random.categorical(subkey, jnp.log(probs + 1e-8))]
        next_token = jnp.clip(next_token, 0, model.vocab_size - 1).astype(jnp.int32)
        generated = jnp.concatenate([generated[:, 1:], next_token[None, None]], axis=1)

    tokens = generated[0].tolist()
   
    decoded = enc.decode(tokens)
   
       
    return decoded
# main
if __name__ == "__main__":
    config = {
        'vocab_size': 50257,
        'n_embed': 32,          
        'block_size': 8,       
        'num_heads': 2,
        'num_layers': 2,
        'dropout': 0.1,
        'batch_size': 8,
        'T': 3,
        'eta': 0.001,
        'exp_dir': 'exp',
        'model_name': 'transformer'
    }
    dkey = jax.random.PRNGKey(0)
    model = PC_Transformer(dkey, **config)
    train_losses, val_losses = train_model(
        model, num_steps=1000, eval_steps=100,
        block_size=config['block_size'], batch_size=config['batch_size']
    )
    prompt = "I say unto you "
    generated_text = generate_text(
        model, enc, prompt, max_length=1000,  
        block_size=config['block_size']
    )
    print(f"\nGenerated Text:\n{generated_text}")