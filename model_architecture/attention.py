from flax import nnx
import jax.numpy as jnp
from ngcsimlib.compartment import Compartment
import jax
from ngclearn.components.jaxComponent import JaxComponent
from ngcsimlib.compilers.process import transition

class MultiHeadAttention(JaxComponent):
    def __init__(self, name, num_heads, head_size, n_embed, dropout, block_size, batch_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.batch_size = batch_size
        self.shape = (batch_size, block_size, n_embed)
        rngs = nnx.Rngs(default=key)
        self.qkv = nnx.Linear(n_embed, 3 * n_embed, rngs=rngs)
        self.proj = nnx.Linear(n_embed, n_embed, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.z = Compartment(jnp.zeros(self.shape), display_name="Rate Activity")
        self.zF = Compartment(jnp.zeros(self.shape), display_name="Transformed Rate Activity")
        self.j = Compartment(jnp.zeros(self.shape), display_name="Input Stimulus")
        self.j_td = Compartment(jnp.zeros(self.shape), display_name="Modulatory Stimulus")

    # @transition(output_compartments=["zF", "z"])
    def advance_state(self):
        if self.j.value is not None:
            B, T, C = self.j.value.shape
            qkv = self.qkv(self.j.value).reshape(B, T, 3, self.num_heads, self.head_size)
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
            zF = out
            z = self.proj(out)
            self.zF.set(zF)
            self.z.set(z)
            return zF, z
        return self.zF.value, self.z.value

    # @transition(output_compartments=["z", "zF", "j", "j_td"])
    def reset(self):
        z = jnp.zeros(self.shape)
        zF = jnp.zeros(self.shape)
        j = jnp.zeros(self.shape)
        j_td = jnp.zeros(self.shape)
        self.z.set(z)
        self.zF.set(zF)
        self.j.set(j)
        self.j_td.set(j_td)
        return z, zF, j, j_td