import jax.numpy as jnp

def precompute_freqs_cis_real(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute RoPE cos/sin of shape [end, dim] for easy broadcasting.
    Args:
        dim: head dimension (must be even)
        end: sequence length
        theta: base frequency (default 10000.0)
    Returns:
        cos: [end, dim] cosine matrix
        sin: [end, dim] sine matrix
    """
    assert dim % 2 == 0, "RoPE head dim must be even."
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
    t = jnp.arange(end).astype(jnp.float32)
    freqs = jnp.outer(t, freqs)  # [end, dim//2]
    cos = jnp.zeros((end, dim), dtype=jnp.float32)
    sin = jnp.zeros((end, dim), dtype=jnp.float32)
    cos = cos.at[:, 0::2].set(jnp.cos(freqs))
    cos = cos.at[:, 1::2].set(jnp.cos(freqs))
    sin = sin.at[:, 0::2].set(jnp.sin(freqs))
    sin = sin.at[:, 1::2].set(jnp.sin(freqs))
    return cos, sin

def rotate_half(x):
    """
    Rotates half the hidden dims of the input (last axis).
    Used for the RoPE 'real' implementation trick.
    Args:
        x: [..., dim]
    Returns:
        Rotated tensor [..., dim]
    """
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2:]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_emb(xq, xk, cos, sin):
    """
    Apply rotary embeddings using the Sine-Cosine rewrite.
    Args:
        xq: [B, n_heads, seq_len, head_dim]
        xk: [B, n_heads, seq_len, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]
    Returns:
        xq_out, xk_out: rotated Q, K
    """
    # Reshape cos/sin for broadcasting: [1, 1, seq_len, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out
