import jax
import jax.numpy as jnp
from flax import nnx
from ngclearn.components import DenseSynapse
import ngclearn.utils.weight_distribution as dist
import optax
import torch.nn as nn
class Head(nnx.Module):
    """One head of self-attention."""
    
    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float, rngs: nnx.Rngs):
        """
        Initialize the attention head.
        
        Args:
            head_size: Size of each attention head (n_embd / num_heads).
            n_embd: Embedding dimension.
            block_size: Maximum sequence length (for causal mask).
            dropout: Dropout probability.
            rngs: Random number generator state for dropout.
        """
        # Key, query, value: no biases, shape (n_embd, head_size)
        self.key = DenseSynapse(
            name="key",
            shape=(n_embd, head_size),
            weight_init=dist.uniform(amin=-0.1, amax=0.1),
            bias_init=None,  # No biases
            resist_scale=1.,
            p_conn=1.,
            batch_size=1
        )
        self.query = DenseSynapse(
            name="query",
            shape=(n_embd, head_size),
            weight_init=dist.uniform(amin=-0.1, amax=0.1),
            bias_init=None,
            resist_scale=1.,
            p_conn=1.,
            batch_size=1
        )
        self.value = DenseSynapse(
            name="value",
            shape=(n_embd, head_size),
            weight_init=dist.uniform(amin=-0.1, amax=0.1),
            bias_init=None,
            resist_scale=1.,
            p_conn=1.,
            batch_size=1
        )
        # Causal mask
        self.tril = jnp.tril(jnp.ones((block_size, block_size)))
        # Dropout
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass of the attention head.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd).
            training: Whether to apply dropout.
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, head_size).
        """
        B, T, C = x.shape
        # Compute key, query, value
        k = self.key.advance_state(
            Rscale=self.key.Rscale,
            inputs=x,
            weights=self.key.weights.value,
            biases=self.key.biases.value
        )  # (B, T, head_size)
        q = self.query.advance_state(
            Rscale=self.query.Rscale,
            inputs=x,
            weights=self.query.weights.value,
            biases=self.query.biases.value
        )  # (B, T, head_size)
        v = self.value.advance_state(
            Rscale=self.value.Rscale,
            inputs=x,
            weights=self.value.weights.value,
            biases=self.value.biases.value
        )  # (B, T, head_size)
        # Compute attention scores
        wei = q @ jnp.transpose(k, (-2, -1)) * (C ** -0.5)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = jnp.where(self.tril[:T, :T] == 0, float('-inf'), wei)  # Causal mask
        wei = jax.nn.softmax(wei, axis=-1)  # (B, T, T)
        wei = self.dropout(wei, deterministic=not training)
        # Weighted aggregation
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nnx.Module):
    """Multiple heads of self-attention in parallel."""
    
    def __init__(self, num_heads: int, head_size: int, n_embd: int, dropout: float, block_size: int, rngs: nnx.Rngs):
        """
        Initialize the multi-head attention module.
        
        Args:
            num_heads: Number of attention heads.
            head_size: Size of each head (n_embd / num_heads).
            n_embd: Embedding dimension.
            dropout: Dropout probability.
            block_size: Maximum sequence length.
            rngs: Random number generator state for dropout.
        """
        self.heads = nnx.ModuleList([
            Head(head_size, n_embd, block_size, dropout, rngs) for _ in range(num_heads)
        ])
        self.proj = DenseSynapse(
            name="proj",
            shape=(n_embd, n_embd),
            weight_init=dist.uniform(amin=-0.1, amax=0.1),
            bias_init=dist.constant(0.),
            resist_scale=1.,
            p_conn=1.,
            batch_size=1
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass of the multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd).
            training: Whether to apply dropout.
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embd).
        """
        # Run heads in parallel
        head_outputs = [h(x, training=training) for h in self.heads]  # List of (B, T, head_size)
        out = jnp.concatenate(head_outputs, axis=-1)  # (B, T, num_heads * head_size = n_embd)
        out = self.proj.advance_state(
            Rscale=self.proj.Rscale,
            inputs=out,
            weights=self.proj.weights.value,
            biases=self.proj.biases.value
        )  # (B, T, n_embd)
        out = self.dropout(out, deterministic=not training)
        return out
