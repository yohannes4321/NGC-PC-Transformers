import jax.numpy as jnp
from jax import jit
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment, compilable




class UniversalScaler(JaxComponent):
        # Only wire advance_state and reset to MethodProcess, not universal_rms_scale
    def __init__(self, name, input_shape, output_shape, **kwargs):
        super().__init__(name, **kwargs)

        self.input_shape = input_shape
        self.output_shape = output_shape

        # Infer dimensions
        flat_tokens, self.n_embed = input_shape
        self.batch_size, self.seq_len, out_embed = output_shape

        assert self.n_embed == out_embed, \
            "Embedding dimension must match between input and output"

        # Pipe scaling factor (intensive scaling)
        self.volume_scale = 1.0 / jnp.sqrt(self.batch_size * self.seq_len)

        # Learnable scale parameter
        self.gamma = jnp.ones((self.n_embed,))

        # Compartments
        self.inputs = Compartment(jnp.zeros(input_shape))
        self.outputs = Compartment(jnp.zeros(output_shape))

    @compilable
    def advance_state(self):
        x = self.inputs.get()  # (B*S, D)
        # Inline scaling logic
        x_float = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
        x_norm = x * (1.0 / jnp.sqrt(variance + 1e-6))
        out = x_norm * self.volume_scale * self.gamma.astype(x.dtype)
        out = jnp.reshape(out, self.output_shape)
        self.outputs.set(out)

    @compilable
    def reset(self):
        self.inputs.set(jnp.zeros(self.input_shape))
        self.outputs.set(jnp.zeros(self.output_shape))