import jax.numpy as jnp
from flax import nnx
from ngcsimlib.compartment import Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngcsimlib.compilers.process import transition
# Transformer components with ngclearn JaxComponent

class EmbeddingLayer(JaxComponent):
    def __init__(self, name, n_embed, vocab_size, block_size, dropout, batch_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.vocab_size = vocab_size
        self.shape = (batch_size, block_size, n_embed)
        rngs = nnx.Rngs(default=key)
        self.wte = nnx.Embed(vocab_size, n_embed, rngs=rngs)
        self.wpe = nnx.Embed(block_size, n_embed, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.z = Compartment(jnp.zeros(self.shape), display_name="Rate Activity")
        self.zF = Compartment(jnp.zeros(self.shape), display_name="Transformed Rate Activity")
        self.j = Compartment(jnp.zeros((batch_size, block_size)), display_name="Input Stimulus")

    # @transition(output_compartments=["zF", "z"])
    def advance_state(self):
        if self.j.value is not None:
            B, T = self.j.value.shape
            tok_emb = self.wte(self.j.value)  # (B, T, n_embed)
            pos = jnp.arange(T)[None, :]  # (1, T)
            pos_emb = self.wpe(pos)  # (1, T, n_embed)
            zF = tok_emb + pos_emb
            z = self.dropout(zF)
            self.zF.set(zF)
            self.z.set(z)
            return zF, z
        return self.zF.value, self.z.value

    # @transition(output_compartments=["z", "zF", "j"])
    def reset(self):
        z = jnp.zeros(self.shape)
        zF = jnp.zeros(self.shape)
        j = jnp.zeros((self.batch_size, self.block_size))
        self.z.set(z)
        self.zF.set(zF)
        self.j.set(j)
        return z, zF, j