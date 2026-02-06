from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment 
from jax import numpy as jnp, random, jit
from ngclearn import compilable
import jax
from functools import partial
@partial(jit, static_argnums=[4, 5, 6, 7, 8, 10])
def _compute_attention(Q, K, V, mask, n_heads, d_head, dropout_rate, seq_len, batch_size, key, use_cache=False, k_cache=None, v_cache=None, cache_valid=None):
    """
    Compute multi-head attention 
    """
    if Q.ndim == 2:
        # 2D input: (batch_size * seq_len, n_embed) -> reshape to 3D
        M, D = Q.shape
        S = seq_len        
        B = M // S  # batch_size
        Q_3d = Q.reshape(B, S, D)
        K_3d = K.reshape(B, S, D)
        V_3d = V.reshape(B, S, D)
    else:
        # 3D input: (batch_size, seq_len, n_embed)
        B, S, D = Q.shape
        Q_3d, K_3d, V_3d = Q, K, V
    if use_cache:
        if cache_valid is None:
            cache_valid = jnp.array(False)
        if k_cache is None or v_cache is None:
            k_cache = jnp.zeros_like(K_3d)
            v_cache = jnp.zeros_like(V_3d)
        k_all = jnp.concatenate([k_cache, K_3d], axis=1)
        v_all = jnp.concatenate([v_cache, V_3d], axis=1)
        k_all = k_all[:, -seq_len:, :]
        v_all = v_all[:, -seq_len:, :]
    else:
        k_all = K_3d
        v_all = V_3d

    # Reshape for multi-head attention
    q = Q_3d.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    k = k_all.reshape((B, k_all.shape[1], n_heads, d_head)).transpose([0, 2, 1, 3]) 
    v = v_all.reshape((B, v_all.shape[1], n_heads, d_head)).transpose([0, 2, 1, 3])
    # Scaled dot-product attention
    score = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(d_head)
    
    if mask is not None:
        Tq, Tk = q.shape[2], k.shape[2]
        if mask.ndim == 3:
            _mask = mask[:, :Tq, :Tk]
        else:
            _mask = mask
        _mask = _mask.reshape((B, 1, Tq, Tk))
        score = jnp.where(_mask, score, -1e-9)
        
    score = jax.nn.softmax(score, axis=-1)
    score = score.astype(q.dtype)
    
    if dropout_rate > 0.0:
        dkey = random.fold_in(key, 0)
        # dkey = random.PRNGKey(0)
        score = jax.random.bernoulli(dkey, 1 - dropout_rate, score.shape) * score / (1 - dropout_rate)
        
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, S, -1))
    
    if use_cache:
        new_k_cache = k_all
        new_v_cache = v_all
        new_cache_valid = jnp.array(True)
    else:
        new_k_cache = k_cache
        new_v_cache = v_cache
        new_cache_valid = cache_valid
    return attention, new_k_cache, new_v_cache, new_cache_valid

class AttentionBlock(JaxComponent):
    """
    Multi-head attention block for NGC attention.
    
    Takes Q, K, V inputs and computes scaled dot-product attention 
    with optional masking and dropout.
    
    | --- Compartments: ---
    | inputs_q - query inputs
    | inputs_k - key inputs  
    | inputs_v - value inputs
    | mask - attention mask
    | outputs - attention outputs
    | key - JAX PRNG key

    Args:
        name: Component name
        n_heads: Number of attention heads
        n_embed: Embedding dimension
        seq_len: Sequence length
        dropout_rate: Attention dropout rate
        batch_size: Batch size
    """
    
    def __init__(self, name, n_heads, n_embed, seq_len, dropout_rate, batch_size, **kwargs):
        super().__init__(name, **kwargs)

        self.n_heads = n_heads
        self.n_embed = n_embed
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        if n_embed % n_heads != 0:
            raise ValueError(f"n_embed={n_embed} must be divisible by n_heads={n_heads}")
        self.d_head = n_embed // n_heads

        # Input compartments
        self.inputs_q = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_k = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_v = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.mask = Compartment(jnp.zeros((batch_size, seq_len, seq_len), dtype=bool))
        
        self.key = Compartment(random.PRNGKey(0))
        # Output compartment
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.use_cache = False
        self.k_cache = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.v_cache = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.cache_valid = Compartment(jnp.array(False))
    @compilable
    def advance_state(self):
        """
        Compute multi-head attention
        """
        inputs_q=self.inputs_q.get()
        inputs_k=self.inputs_k.get()
        inputs_v=self.inputs_v.get()
        mask=self.mask.get()
        n_heads=self.n_heads
        d_head=self.d_head
        dropout_rate=self.dropout_rate
        key=self.key.get()
        attention, new_k_cache, new_v_cache, new_cache_valid = _compute_attention(
            inputs_q, inputs_k, inputs_v, mask,        
            self.n_heads,        
            self.d_head,        
            self.dropout_rate, 
            self.seq_len,
            self.batch_size,  
            key,
            self.use_cache,
            self.k_cache.get(),
            self.v_cache.get(),
            self.cache_valid.get()
        )
        if self.use_cache:
            self.k_cache.set(new_k_cache)
            self.v_cache.set(new_v_cache)
            self.cache_valid.set(new_cache_valid)
        else:
            self.k_cache.set(jnp.zeros_like(new_k_cache))
            self.v_cache.set(jnp.zeros_like(new_v_cache))
            self.cache_valid.set(jnp.array(False))
        self.outputs.set(attention)
    @compilable
    def reset(self):
        """
        Reset compartments to zeros
        """
        batch_size=self.batch_size
        seq_len=self.seq_len
        n_embed=self.n_embed
        zeros_3d = jnp.zeros((batch_size, seq_len, n_embed))
        mask = jnp.zeros((batch_size, seq_len, seq_len), dtype=bool)
        # return zeros_3d, zeros_3d, zeros_3d, mask, zeros_3d
        self.inputs_q.set(zeros_3d)
        self.inputs_k.set(zeros_3d)
        self.inputs_v.set(zeros_3d)
        self.mask.set(mask)
        self.outputs.set(zeros_3d)
        if not self.use_cache:
            self.k_cache.set(zeros_3d)
            self.v_cache.set(zeros_3d)
            self.cache_valid.set(jnp.array(False))

    def set_use_cache(self, use_cache):
        """Enable or disable KV caching for this block."""
        self.use_cache = bool(use_cache)
        if not self.use_cache:
            self.k_cache.set(jnp.zeros_like(self.k_cache.get()))
            self.v_cache.set(jnp.zeros_like(self.v_cache.get()))
            self.cache_valid.set(jnp.array(False))

    def clear_kv_cache(self):
        """Clear the KV cache."""
        self.k_cache.set(jnp.zeros_like(self.k_cache.get()))
        self.v_cache.set(jnp.zeros_like(self.v_cache.get()))
        self.cache_valid.set(jnp.array(False))

    @classmethod
    def help(cls):
        """Component help function"""
        properties = {
            "component_type": "AttentionBlock - multi-head self-attention mechanism"
        }
        compartment_props = {
            "inputs": 
                {"inputs_q": "Query inputs (batch_size, seq_len, n_embed)",
                 "inputs_k": "Key inputs (batch_size, seq_len, n_embed)", 
                 "inputs_v": "Value inputs (batch_size, seq_len, n_embed)",
                 "mask": "Attention mask (batch_size, seq_len, seq_len)"},
            "outputs":
                {"outputs": "Attention outputs (batch_size, seq_len, n_embed)"},
        }
        hyperparams = {
            "n_heads": "Number of attention heads",
            "n_embed": "Embedding dimension", 
            "seq_len": "Sequence length",
            "dropout_rate": "Attention dropout rate",
            "batch_size": "Batch size dimension"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = MultiHeadAttention(Q, K, V, mask)",
                "hyperparameters": hyperparams}
        return info