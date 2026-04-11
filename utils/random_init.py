from jax import random
import jax.numpy as jnp
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable

class RandomInit(JaxComponent):
    """NGC-compatible random state initializer for latent z tensors.
    
    Generates fresh random values each time advance_state() is called.
    Access the compartments directly:
    - z_normal: shape (batch_size, n_embed)
    - z_4x_projection: shape (batch_size, 4 * n_embed)
    """

    def __init__(self, name, batch_size, n_embed, scale=1e-3, key=None, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.n_embed = n_embed
        self.scale = scale
        self.key = Compartment(random.PRNGKey(0) if key is None else key)
        self.z_normal = Compartment(jnp.zeros((batch_size, n_embed)))
        self.z_4x_projection = Compartment(jnp.zeros((batch_size, 4 * n_embed)))

    @compilable
    def advance_state(self):
        """Generate fresh random values for both compartments."""
        key = self.key.get()
        k1, k2 = random.split(key, 2)
        z_normal = random.normal(k1, (self.batch_size, self.n_embed)) * self.scale
        z_4x_projection = random.normal(k2, (self.batch_size, 4 * self.n_embed)) * self.scale
        self.key.set(k2)
        self.z_normal.set(z_normal)
        self.z_4x_projection.set(z_4x_projection)
