import jax
import jax.numpy as jnp
from flax import nnx
from ngclearn.components import DenseSynapse
import ngclearn.utils.weight_distribution as dist
import optax

class FeedForward(nnx.Module):
    """A simple feedforward layer with two linear transformations, ReLU, and dropout."""
    
    def __init__(self, n_embd: int, dropout: float, rngs: nnx.Rngs):
        """
        Initialize the FeedForward module.
        
        Args:
            n_embd: Input and output embedding dimension.
            dropout: Dropout probability.
            rngs: Random number generator state for dropout.
        """
        # First linear layer: n_embd -> 4 * n_embd
        self.linear1 = DenseSynapse(
            name="ffwd_linear1",
            shape=(n_embd, 4 * n_embd),
            weight_init=dist.uniform(amin=-0.1, amax=0.1),
            bias_init=dist.constant(0.),
            resist_scale=1.,  # Explicitly set to match nn.Linear
            p_conn=1.,        # Dense connections
            batch_size=1      # Default, doesn't affect dynamic batching
        )
        # Second linear layer: 4 * n_embd -> n_embd
        self.linear2 = DenseSynapse(
            name="ffwd_linear2",
            shape=(4 * n_embd, n_embd),
            weight_init=dist.uniform(amin=-0.1, amax=0.1),
            bias_init=dist.constant(0.),
            resist_scale=1.,
            p_conn=1.,
            batch_size=1
        )
        # Dropout
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the feedforward module.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd).
            training: Whether to apply dropout (True for training, False for inference).
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embd).
        """
        # Explicitly pass Rscale, weights, biases to advance_state
        x = self.linear1.advance_state(
            Rscale=self.linear1.Rscale,
            inputs=x,
            weights=self.linear1.weights.value,
            biases=self.linear1.biases.value
        )
        x = nnx.gelu(x)  # Apply 
        x = self.linear2.advance_state(
            Rscale=self.linear2.Rscale,
            inputs=x,
            weights=self.linear2.weights.value,
            biases=self.linear2.biases.value
        )
        x = self.dropout(x)  # Apply dropout
        return x

