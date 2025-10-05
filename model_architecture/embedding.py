import jax.numpy as jnp
from flax import nnx



class EmbeddingLayer(nnx.Module):
    """
    Embedding layer with word and positional embeddings, RMS normalization, and dropout.
    """
    
    def __init__(self, config, rngs: nnx.Rngs):
        """
        Initialize the EmbeddingLayer.
        
        Args:
            config: Configuration object with vocab_size, n_embed, block_size, dropout.
            rngs: Random number generator state for dropout.
        """
        # Word and positional embeddings
        self.word_embeddings = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.n_embed,
            rngs=rngs
        )
        self.position_embeddings = nnx.Embed(
            num_embeddings=config.block_size,
            features=config.n_embed,
            rngs=rngs
        )
        # RMS normalization
        self.rms_norm = nnx.RMSNorm(num_features=config.n_embed, rngs=rngs)
        # Dropout
        self.dropout = nnx.Dropout(rate=config.dropout, rngs=rngs)
    
    def __call__(self, idx: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass of the embedding layer.
        
        Args:
            idx: Input tensor of shape (batch_size, sequence_length) with token indices.
            training: Whether to apply dropout.
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embed).
        """
        B, T = idx.shape
        # Word embeddings
        tok_emb = self.word_embeddings(idx)  # (B, T, n_embed)
        # Positional embeddings
        pos = jnp.arange(T)  # (T,)
        pos_emb = self.position_embeddings(pos)  # (T, n_embed)
        # Combine embeddings
        x = tok_emb + pos_emb  # (B, T, n_embed)
        # RMS normalization
        x = self.rms_norm(x)  # (B, T, n_embed)
        # Dropout
        x = self.dropout(x)  # (B, T, n_embed)
        return x
