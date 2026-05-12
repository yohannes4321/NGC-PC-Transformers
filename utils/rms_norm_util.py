import jax.numpy as jnp
from jax import jit

from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable


@jit
def rms_normalize(x, gamma, eps=1e-6):
    """
    RMS Normalization forward pass.

    Formula:
        rms   = sqrt( mean(x^2) + eps )
        y     = (x / rms) * gamma
    """
    x_float  = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    rms      = jnp.sqrt(variance + eps).astype(x.dtype)
    out      = x * (1.0 / rms) * gamma.astype(x.dtype)
    return out, rms


@jit
def rms_norm_grad(x, rms, gamma, v):
    """
    RMSNorm Jacobian-vector product:  dx = J_RMSNorm(x)^T @ v

    
    """
    gamma_r = gamma.reshape((1,) * (v.ndim - 1) + (-1,)).astype(x.dtype)
    x_norm  = x / rms
    # 1. Apply weights (gamma) to the incoming error (v)
    v_weighted = v * gamma_r
    scale   = 1 / rms
    inner   = jnp.mean(x_norm * v_weighted, axis=-1, keepdims=True)
    dx      = scale * (v_weighted - x_norm * inner)
    return dx


class RMSNorm(JaxComponent):
    """
    RMS normalisation — forward pass only.

    Compartments:
        .inputs  : pre-norm tensor   (batch, n_embed)
        .outputs : post-norm tensor  (batch, n_embed)
        .rms     : saved rms value   (batch, 1)  — used by RMSNormGrad
    """

    def __init__(self, name, n_embed, batch_size, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed    = n_embed
        self.batch_size = batch_size
        self.gamma      = jnp.ones((n_embed,))

        self.inputs  = Compartment(jnp.zeros((batch_size, n_embed)))
        self.outputs = Compartment(jnp.zeros((batch_size, n_embed)))
        self.rms     = Compartment(jnp.ones((batch_size, 1)))

    @compilable
    def advance_state(self):
        x        = self.inputs.get()
        out, rms = rms_normalize(x, self.gamma)
        self.outputs.set(out)
        self.rms.set(rms)

    @compilable
    def reset(self):
        zeros = jnp.zeros((self.batch_size, self.n_embed))
        self.inputs.set(zeros)
        self.outputs.set(zeros)
        self.rms.set(jnp.ones((self.batch_size, 1)))


class RMSNormGrad(JaxComponent):
    

    def __init__(self, name, n_embed, batch_size, gamma=None, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed    = n_embed
        self.batch_size = batch_size
        self.gamma      = gamma if gamma is not None else jnp.ones((n_embed,))

        self.z        = Compartment(jnp.zeros((batch_size, n_embed)))
        self.rms       = Compartment(jnp.ones((batch_size, 1)))
        self.dmu_mlp1  = Compartment(jnp.zeros((batch_size, 4 * n_embed)))
        self.dmu_attn  = Compartment(jnp.ones((batch_size, n_embed)))
        self.dmu_out      = Compartment(jnp.zeros((batch_size, n_embed)))
        self.dmu_mlp1_out  = Compartment(jnp.zeros((batch_size, 4 * n_embed)))

    @compilable
    def advance_state(self):
        x   = self.z.get()
        rms = self.rms.get()
        v_attn = self.dmu_attn.get()
        v_mlp1   = self.dmu_mlp1.get()


        # Apply the RMSNorm Jacobian once — derivation applied to v only
        dx_attn = rms_norm_grad(x, rms, self.gamma, v_attn)
        dx_mlp1  = rms_norm_grad(x, rms, self.gamma, v_mlp1)

        self.dmu_out.set(dx_attn)
        self.dmu_mlp1_out.set(dx_mlp1)

    

    @compilable
    def reset(self):
        zeros = jnp.zeros((self.batch_size, self.n_embed))
        self.z.set(zeros)
        self.rms.set(jnp.ones((self.batch_size, 1)))
        self.dmu_attn.set(zeros)
        self.dmu_out.set(zeros)
        self.dmu_mlp1.set(jnp.zeros((self.batch_size, 4 * self.n_embed)))
        self.dmu_mlp1_out.set(jnp.zeros((self.batch_size, 4 * self.n_embed)))