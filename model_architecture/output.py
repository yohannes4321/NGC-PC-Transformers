import jax
import jax.numpy as jnp
from flax import nnx
from ngclearn.components import DenseSynapse
import ngclearn.utils.weight_distribution as dist
import optax

class OutputLayer(nnx.Module):
    """Output layer mapping transformer embeddings to vocabulary logits."""
    
    def __init__(self, n_embd: int, vocab_size: int, rngs: nnx.Rngs):
        """
        Initialize the output layer.
        
        Args:
            n_embd: Embedding dimension.
            vocab_size: Size of the vocabulary (number of output classes).
            rngs: Random number generator state (required for nnx.Module).
        """
        # Output layer: n_embd -> vocab_size
        self.output = DenseSynapse(
            name="output",
            shape=(n_embd, vocab_size),
            weight_init=dist.uniform(amin=-0.1, amax=0.1),
            bias_init=dist.constant(0.),
            resist_scale=1.,
            p_conn=1.,
            batch_size=1
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the output layer.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd).
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        # Call advance_state as instance method, passing all args explicitly
        x = self.output.advance_state(
            Rscale=self.output.Rscale,
            inputs=x,
            weights=self.output.weights.value,
            biases=self.output.biases.value
        )  # (batch_size, sequence_length, vocab_size)
        return x
