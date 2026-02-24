from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment 
from jax import numpy as jnp, random, jit
from ngclearn import compilable
import jax
from functools import partial
import jax.numpy as jnp
from jax import vmap

@partial(jit, static_argnums=[4, 5, 6, 7, 8])
def _compute_attention(Q, K, V, mask, n_heads, d_head, dropout_rate, seq_len, batch_size, key):
    """
    Compute multi-head attention 
    """
    B = batch_size
    S = seq_len
    D = Q.shape[-1]
    
    Q_3d = Q.reshape(B, S, D)
    K_3d = K.reshape(B, S, D)
    V_3d = V.reshape(B, S, D)
    
    # Reshape for multi-head attention
    q = Q_3d.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    k = K_3d.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3]) 
    v = V_3d.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    # Scaled dot-product attention
    s_c = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(d_head)
    
    if mask is not None:
        Tq, Tk = q.shape[2], k.shape[2]
        _mask = mask.reshape((B, 1, Tq, Tk))
        s_c = jnp.where(_mask, s_c, -1e-9)
        
    score = jax.nn.softmax(s_c, axis=-1)
    score = score.astype(q.dtype)
    
    if dropout_rate > 0.0:
        dkey = random.fold_in(key, 0)
        # dkey = random.PRNGKey(0)
        score = jax.random.bernoulli(dkey, 1 - dropout_rate, score.shape) * score / (1 - dropout_rate)
        
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, S, -1))
    
    return attention, s_c, q, k, v

@partial(jit, static_argnums=1)
def d_softmax(x, tau=0.0):
    """
    Returns full Jacobian of softmax applied over last axis.

    Input:
        x: (B, H, T, T)

    Output:
        J: (B, H, T, T, T)
           where last two dims are the (T x T) Jacobian
           for each row.
    """
    if tau > 0.0:
        x = x / tau

    # Compute softmax over last axis
    p = jax.nn.softmax(x, axis=-1)  # (B, H, T, T)

    T = p.shape[-1]

    # Identity matrix for Jacobian construction
    I = jnp.eye(T)

    def row_jacobian(p_row):
        # p_row shape: (T,)
        return jnp.diag(p_row) - jnp.outer(p_row, p_row)

    # Vectorize over last 3 dimensions: (B, H, Tq)
    jacobian = vmap(                      # over B
                    vmap(                  # over H
                        vmap(row_jacobian, # over Tq
                             in_axes=0),
                        in_axes=0),
                    in_axes=0)(p)

    return jacobian

@partial(jit, static_argnums=[6, 7, 8, 9, 10])
def compute_grads(Q, K, V, mask, s_c, e_qkv, n_heads, d_head, dropout_rate, seq_len, batch_size, key):
        """Compute gradients for Q, K, V using d_softmax and attention scores
        """
        J=d_softmax(s_c, tau=0.0)  # Shape: (B, H, S, S, S)
        e_qkv=e_qkv.reshape((batch_size, seq_len, n_heads, d_head)).transpose([0, 2, 1, 3])  # (B, H, S, d_head)
        d_v = jnp.einsum("bhsss,bhsd->bhsd", J, e_qkv)
        da = jnp.matmul(e_qkv, V.transpose([0, 1, 3, 2]))  # (B, H, S, S)
        ds = jnp.einsum("bhqkv,bhqd->bhkv", J, da)        
        d_q = jnp.einsum("bhqv,bhkd->bhqd", ds, K).reshape(batch_size, seq_len, n_heads * d_head)
        d_k = jnp.einsum("bhqv,bhqd->bhqd", ds, Q).reshape(batch_size, seq_len, n_heads * d_head)
        dq= d_q.reshape(batch_size * seq_len, n_heads * d_head)
        dk= d_k.reshape(batch_size * seq_len, n_heads * d_head)
        dv= d_v.reshape(batch_size * seq_len, n_heads * d_head)
        #d_k = ((V.T @ e_qkv)* J).T @ Q
        return dq, dk, dv
    
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
        
        if self.n_embed % self.n_heads != 0:
            raise ValueError(f"n_embed={n_embed} must be divisible by n_heads={n_heads}")
        self.d_head = n_embed // n_heads

        # Input compartments
        self.inputs_q = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_k = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_v = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.J = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.dq = Compartment(jnp.zeros((batch_size*seq_len, n_embed)))
        self.dk = Compartment(jnp.zeros((batch_size*seq_len, n_embed)))
        self.dv = Compartment(jnp.zeros((batch_size*seq_len, n_embed)))
        self.e_qkv = Compartment(jnp.zeros((batch_size * seq_len, n_embed)))
        self.mask = Compartment(jnp.zeros((batch_size, seq_len, seq_len), dtype=bool))
        # self.S = Compartment(jnp.zeros((batch_size, n_heads, seq_len, seq_len)))
        
        self.key = Compartment(random.PRNGKey(0))
        # Output compartment
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))    
    
    @compilable
    def advance_state(self):
        """
        Compute multi-head attention
        and compute gradients for inputs and Jacobian using d_softmax
        """
        inputs_q=self.inputs_q.get()
        inputs_k=self.inputs_k.get()
        inputs_v=self.inputs_v.get()
        mask=self.mask.get()
        # S=self.S.get()
        e_qkv=self.e_qkv.get()
        n_heads=self.n_heads
        d_head=self.d_head
        dropout_rate=self.dropout_rate
        key=self.key.get()
        attention, s_c, q, k, v = _compute_attention(
            inputs_q, inputs_k, inputs_v, mask,        
            self.n_heads,        
            self.d_head,        
            self.dropout_rate, 
            self.seq_len,
            self.batch_size,  
            key                  
        )
        # self.S.set(S)
        dq, dk, dv = compute_grads(q, k, v, mask, s_c, e_qkv, n_heads, d_head, dropout_rate, self.seq_len, self.batch_size, key)
        self.dq.set(dq)
        self.dk.set(dk)
        self.dv.set(dv)
        
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