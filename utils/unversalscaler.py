import jax.numpy as jnp
from jax import jit
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable

@jit
def universal_rms_normalize(x, gamma, eps=1e-6):
    """
    Broad RMS Normalization - Normalizes across all dimensions 
    (Batch, Seq, Feature) to stabilize global energy.
    """
    x_float = x.astype(jnp.float32)
    
    # Calculate variance across all axes (global scaling)
    # This prevents energy from scaling linearly with batch_size or seq_len
    variance = jnp.mean(jnp.square(x_float), axis=None, keepdims=True)
    
    scale = 1.0 / jnp.sqrt(variance + eps)
    
    scale = scale.astype(x.dtype)
    gamma_casted = gamma.astype(x.dtype)

    return x * scale * gamma_casted

class UniversalScaler(JaxComponent):
    """
    Scales inputs globally to ensure free energy remains stable regardless 
    of batch size or sequence length.
    
    Parameters:
    - name: Component name
    - n_embed: number of features
    - batch_size: The batch size
    - seq_len: The sequence length (added to track energy across time)
    """

    def __init__(self, name, n_embed, batch_size, seq_len=1, **kwargs):
        super().__init__(name, **kwargs)
        self.n_embed = n_embed
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Learnable gain parameter
        self.gamma = jnp.ones((1,)) # Global gain, or use (n_embed,) for feature-wise

        # Shape updated to include seq_len (B, S, D)
        self.inputs = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))

    @compilable
    def advance_state(self):
        x = self.inputs.get()
        # Apply universal normalization
        out = universal_rms_normalize(x, self.gamma)
        self.outputs.set(out)

    @compilable
    def reset(self):
        # Reset to zero-filled tensors matching the (B, S, D) shape
        x = jnp.zeros((self.batch_size, self.seq_len, self.n_embed))
        self.inputs.set(x)
        self.outputs.set(x)