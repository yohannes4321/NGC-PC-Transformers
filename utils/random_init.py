from jax import random
import jax.numpy as jnp

from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable


class RandomInit(JaxComponent):
    """
    Simple random initializer for latent z tensors.

    Compartments:
    - z_normal: (batch_size, n_embed)
    - z_4x_projection: (batch_size, 4 * n_embed)
    """

    def __init__(self, name, batch_size, n_embed, scale=1e-3, key=None, **kwargs):
        super().__init__(name, **kwargs)

        self.batch_size = batch_size
        self.n_embed = n_embed
        self.scale = scale

        # Initialize random key
        self.key = random.PRNGKey(0) if key is None else key

        # Compartments
        self.z_normal = Compartment(jnp.zeros((batch_size, n_embed)))
        self.z_4x_projection = Compartment(jnp.zeros((batch_size, 4 * n_embed)))

    @compilable
    def advance_state(self):
        """Generate new random latent states."""

        # Split key
        k0, k1, k2 = random.split(self.key, 3)
        self.key = k0

        # Small random noise
        z_normal = random.normal(k1, (self.batch_size, self.n_embed)) * self.scale
        z_4x_projection = random.normal(k2, (self.batch_size, 4 * self.n_embed)) * self.scale

        # Update compartments
        self.z_normal.set(z_normal)
        self.z_4x_projection.set(z_4x_projection)