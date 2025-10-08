from flax import nnx
from ngcsimlib.compartment import Compartment
import jax.numpy as jnp
from ngclearn.components.jaxComponent import JaxComponent
from ngcsimlib.compilers.process import transition
import jax

class FeedForward(JaxComponent):
    def __init__(self, name, n_embed, dropout, batch_size, block_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.shape = (batch_size, block_size, n_embed)
        rngs = nnx.Rngs(default=key)
        self.linear1 = nnx.Linear(n_embed, 4 * n_embed, rngs=rngs)
        self.linear2 = nnx.Linear(4 * n_embed, n_embed, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.z = Compartment(jnp.zeros(self.shape), display_name="Rate Activity")
        self.zF = Compartment(jnp.zeros(self.shape), display_name="Transformed Rate Activity")
        self.j = Compartment(jnp.zeros(self.shape), display_name="Input Stimulus")
        self.j_td = Compartment(jnp.zeros(self.shape), display_name="Modulatory Stimulus")

    def advance_state(self, deterministic=False):
        if self.j.value is not None:
            zF = self.j.value
            x = self.linear1(zF)  # First linear layer
            x = jax.nn.relu(x)    # ReLU activation
            x = self.linear2(x)   # Second linear layer
            z = self.dropout(x, deterministic=deterministic)  # Dropout
            self.zF.set(zF)
            self.z.set(z)
            return zF, z
        return self.zF.value, self.z.value

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