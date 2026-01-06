import jax.numpy as jnp
from jax import jit

from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable


@jit
def rms_normalize(x, gamma, eps=1e-6):
    """
    RMS Normalization - Normalizes the input using root mean square.
    """
    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    # compute reciprocal sqrt explicitly, b/c jax.numpy not provide rsqrt in this version
    scale = 1.0 / jnp.sqrt(variance + eps)
    scale = scale.astype(x.dtype)

    return x * scale * gamma


class RMSNorm(JaxComponent):
    """A small ngclearn-compatible RMS normalization component.

    Provides `.inputs` and `.outputs` compartments and implements
    `advance_state` and `reset` so it can be wired like other components
    in `model.py`.

    Parameters
    - n_embed: number of features along last axis
    - prefix: name for the component (kept for backwards compat)
    """

    def __init__(self, name, n_embed, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed = n_embed
        self.gamma = jnp.ones((n_embed,))

        self.inputs = Compartment(jnp.zeros((1, n_embed)))
        self.outputs = Compartment(jnp.zeros((1, n_embed)))

    @compilable
    def advance_state(self):
        x = self.inputs.get()
        # apply RMS normalization across last axis
        out = rms_normalize(x, self.gamma)
        self.outputs.set(out)

    @compilable
    def reset(self):
        x = jnp.zeros((1, self.n_embed))
        self.inputs.set(x)
        self.outputs.set(x)