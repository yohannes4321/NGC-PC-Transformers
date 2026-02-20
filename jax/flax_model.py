import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any

class FlaxTransformer(nn.Module):
    vocab_size: int
    seq_len: int
    n_embed: int
    n_layers: int
    n_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Embedding
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embed)(x)
        # Positional encoding (learned)
        pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=0.02), (1, self.seq_len, self.n_embed))
        x = x + pos_emb
        # Transformer blocks
        for _ in range(self.n_layers):
            x = nn.SelfAttention(
                num_heads=self.n_heads,
                qkv_features=self.n_embed,
                dropout_rate=self.dropout_rate
            )(x, deterministic=not train)
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.n_embed)(x)
            x = nn.relu(x)
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = -jnp.sum(one_hot * nn.log_softmax(logits), axis=-1)
    return jnp.mean(loss)

def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}
