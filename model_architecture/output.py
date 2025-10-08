from flax import nnx
from ngcsimlib.compartment import Compartment
import jax.numpy as jnp
from ngclearn.components.jaxComponent import JaxComponent
from ngcsimlib.compilers.process import transition

class OutputLayer(JaxComponent):
    def __init__(self, name, n_embed, vocab_size, batch_size, block_size, key, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.vocab_size = vocab_size
        self.shape = (batch_size, block_size, vocab_size)
        rngs = nnx.Rngs(default=key)
        self.ln = nnx.LayerNorm(n_embed, rngs=rngs)
        self.head = nnx.Linear(n_embed, vocab_size, rngs=rngs)
        self.z = Compartment(jnp.zeros(self.shape), display_name="Rate Activity")
        self.zF = Compartment(jnp.zeros((batch_size, block_size, n_embed)), display_name="Transformed Rate Activity")
        self.j = Compartment(jnp.zeros((batch_size, block_size, n_embed)), display_name="Input Stimulus")
        self.j_td = Compartment(jnp.zeros(self.shape), display_name="Modulatory Stimulus")

    # @transition(output_compartments=["zF", "z"])
    def advance_state(self, t=0., dt=1.):
        if self.j.value is not None:
            zF = self.ln(self.j.value)
            z = self.head(zF)
            self.zF.set(zF)
            self.z.set(z)
            return zF, z
        return self.zF.value, self.z.value

    # @transition(output_compartments=["z", "zF", "j", "j_td"])
    def reset(self):
        z = jnp.zeros(self.shape)
        zF = jnp.zeros((self.batch_size, self.block_size, self.n_embed))
        j = jnp.zeros((self.batch_size, self.block_size, self.n_embed))
        j_td = jnp.zeros(self.shape)
        self.z.set(z)
        self.zF.set(zF)
        self.j.set(j)
        self.j_td.set(j_td)
        return z, zF, j, j_td