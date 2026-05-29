import jax
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable
from jax import numpy as jnp, random, jit
from functools import partial
from jax import vmap


def d_softmax_vjp(x, tau=0.0):
    """
    Memory-efficient softmax derivative using JVP (Jacobian-Vector Product).
    
    Returns probabilities and a function that computes J @ v without forming 
    the full Jacobian matrix.
    
    Args:
        x: Input tensor of any shape (attention scores, output logits, etc.)
           Examples:
           - Attention: (batch_size, n_heads, seq_len, seq_len)
           - Output: (batch_size, vocab_size) or (batch_size*seq_len, vocab_size)
        tau: Temperature parameter for softmax
    
    Returns:
        P: Softmax probabilities (same shape as x)
        jvp_fn: Function that computes J @ v for any v of same shape as P
               jvp_fn(v) = p * (v - (p @ v))
    """
    if tau > 0.0:
        x = x / tau
    
    # Compute probabilities once
    P = jax.nn.softmax(x, axis=-1)
    
    def jvp_fn(v):
        """
        Compute J @ v efficiently using the identity:
        J @ v = p * (v - (p @ v))
        
        Args:
            v: Vector to multiply Jacobian with (same shape as P)
               In practice, this is dL/dP from upstream
        
        Returns:
            J @ v with same shape as v (which is dL/dx)
        """
        # p @ v along last dimension (sum over that dimension)
        p_dot_v = jnp.sum(P * v, axis=-1, keepdims=True)
        return P * (v - p_dot_v)
    
    return P, jvp_fn

class ReshapeComponent(JaxComponent):
    """Component that reshapes tensors for ngc-learn wiring"""
    
    def __init__(self, name, input_shape, output_shape, **kwargs):
        super().__init__(name, **kwargs)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.inputs = Compartment(jnp.zeros(input_shape))
        self.outputs = Compartment(jnp.zeros(output_shape))
    
    @compilable
    def advance_state(self):
        output=self.inputs.reshape(self.output_shape)
        self.outputs.set(output)
    
    
    @compilable
    def reset(self):
        self.inputs.set(jnp.zeros(self.input_shape))
        self.outputs.set(jnp.zeros(self.output_shape))

class Outgrad(JaxComponent):
    """Compute the Jacobian matrix multiplication for the logits gradients
    This computes: dL/dmu = J_softmax(mu) @ dL/dP
    where mu are the logits (pre-softmax)
    """
    
    def __init__(self, name, batch_size, seq_len, vocab_size, **kwargs):
        super().__init__(name, **kwargs)

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        self.mu = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        self.dmu = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        self.dmu_ = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
   
    @compilable   
    def advance_state(self):
        """Compute the output gradients: dL/dmu = J_softmax(mu) @ dL/dP"""
        
        mu = self.mu.get()        
        dmu = self.dmu.get()      
        
        P, jvp_fn = d_softmax_vjp(mu, tau=0.0)
        
        dmu_out = jvp_fn(dmu)
        
        self.dmu_.set(dmu_out)
        
    @compilable
    def reset(self):
        """Reset compartments to zeros"""
        zeros = jnp.zeros((self.batch_size * self.seq_len, self.vocab_size))
        self.mu.set(zeros)
        self.dmu.set(zeros)
        self.dmu_.set(zeros)

def stat(name, x):
    if x is None:
        print(f"{name:45s} is None")
        return
    x = jnp.asarray(x)
    print(
        f"{name:45s}"
        f" shape={str(x.shape):15s}"
        f" mean={jnp.mean(x):10.6f}"
        f" std={jnp.std(x):10.6f}"
        f" max={jnp.max(x):10.6f}"
        f" min={jnp.min(x):10.6f}"
    )

def trace_model(model):
    print("\n" + "="*100)
    print(f"EXHAUSTIVE MODEL TRACE (T={model.T})")
    print("="*100)

    # --- 1. EMBEDDING ---
    print("\n[EMBEDDING]")
    c = model.embedding
    stat("  z_embed.z", c.z_embed.z.get())
    stat("  z_embed.zF", c.z_embed.zF.get())
    stat("  W_embed.inputs", c.W_embed.inputs.get())
    stat("  W_embed.outputs", c.W_embed.outputs.get())
    stat("  e_embed.mu", c.e_embed.mu.get())
    stat("  e_embed.target", c.e_embed.target.get())

    # --- 2. BLOCKS ---
    for i, b in enumerate(model.blocks):
        print(f"\n[BLOCK {i} - ATTENTION]")
        stat("  z_qkv.z", b.attention.z_qkv.z.get())
        stat("  z_qkv.zF", b.attention.z_qkv.zF.get())
        stat("  W_q.inputs", b.attention.W_q.inputs.get())
        stat("  W_q.outputs", b.attention.W_q.outputs.get())
        stat("  W_k.inputs", b.attention.W_k.inputs.get())
        stat("  W_k.outputs", b.attention.W_k.outputs.get())
        stat("  W_v.inputs", b.attention.W_v.inputs.get())
        stat("  W_v.outputs", b.attention.W_v.outputs.get())
        stat("  attn_block.inputs_q", b.attention.attn_block.inputs_q.get())
        stat("  attn_block.outputs", b.attention.attn_block.outputs.get())
        stat("  z_attn.z", b.attention.z_attn.z.get())
        stat("  z_attn.zF", b.attention.z_attn.zF.get())
        stat("  W_attn_out.inputs", b.attention.W_attn_out.inputs.get())
        stat("  W_attn_out.outputs", b.attention.W_attn_out.outputs.get())
        stat("  e_qkv.mu", b.attention.e_qkv.mu.get())
        stat("  e_qkv.target", b.attention.e_qkv.target.get())
        stat("  e_attn.mu", b.attention.e_attn.mu.get())
        stat("  e_attn.target", b.attention.e_attn.target.get())

        print(f"\n[BLOCK {i} - MLP]")
        stat("  z_mlp.z", b.mlp.z_mlp.z.get())
        stat("  z_mlp.zF", b.mlp.z_mlp.zF.get())
        stat("  W_mlp1.inputs", b.mlp.W_mlp1.inputs.get())
        stat("  W_mlp1.outputs", b.mlp.W_mlp1.outputs.get())
        stat("  z_mlp2.z", b.mlp.z_mlp2.z.get())
        stat("  z_mlp2.zF", b.mlp.z_mlp2.zF.get())
        stat("  W_mlp2.inputs", b.mlp.W_mlp2.inputs.get())
        stat("  W_mlp2.outputs", b.mlp.W_mlp2.outputs.get())
        stat("  e_mlp1.mu", b.mlp.e_mlp1.mu.get())
        stat("  e_mlp1.target", b.mlp.e_mlp1.target.get())
        stat("  e_mlp.mu", b.mlp.e_mlp.mu.get())
        stat("  e_mlp.target", b.mlp.e_mlp.target.get())

    # --- 3. OUTPUT ---
    print("\n[OUTPUT]")
    stat("  z_out.z", model.output.z_out.z.get())
    stat("  z_out.zF", model.output.z_out.zF.get())
    stat("  W_out.inputs", model.output.W_out.inputs.get())
    stat("  W_out.outputs", model.output.W_out.outputs.get())
    stat("  e_out.mu", model.output.e_out.mu.get())
    stat("  e_out.target", model.output.e_out.target.get())
    stat("  z_target.z", model.z_target.z.get())
    stat("  z_actfx.z", model.z_actfx.z.get())
    stat("  z_actfx.zF", model.z_actfx.zF.get())

    print("\n" + "="*100 + "\n")
