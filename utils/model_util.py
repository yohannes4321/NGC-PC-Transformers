import jax
from jax import numpy as jnp
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable

def d_softmax_vjp(P, v=None, tau=0.0):
    """
    Computes the Vector-Jacobian Product (VJP) for Softmax efficiently.
    This calculates: dL/dmu = J^T @ v = P * (v - sum(P * v, axis=-1))
    
    Backward-compatible: If v is omitted (None) or tau is supplied, 
    it falls back safely to processing raw logits and returning a jvp_fn.
    
    Args:
        P: Softmax probabilities output from the cell OR raw logits if used in attention.
        v: Upstream error vector (dL/dP). Defaults to None for legacy calls.
        tau: Temperature parameter (optional fallback).
    Returns:
        Gradients with respect to the pre-softmax logits (dL/dmu), 
        OR a tuple of (Probabilities, jvp_fn) if in legacy mode.
    """
    # Fallback to legacy JVP format if called from attention_utils without 'v'
    if v is None or tau > 0.0:
        # In this legacy case, P is actually the raw input logits 'x'
        x = P
        if tau > 0.0:
            x = x / tau
        P_probs = jax.nn.softmax(x, axis=-1)
        
        # Return a JVP function to avoid breaking old attention code loops
        def jvp_fn(incoming_v):
            p_dot_v = jnp.sum(P_probs * incoming_v, axis=-1, keepdims=True)
            return P_probs * (incoming_v - p_dot_v)
        return P_probs, jvp_fn

    # Standard, highly optimized VJP calculation for your new Outgrad component
    p_dot_v = jnp.sum(P * v, axis=-1, keepdims=True)
    return P * (v - p_dot_v)


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
        output = self.inputs.get().reshape(self.output_shape)
        self.outputs.set(output)
    
    @compilable
    def reset(self):
        self.inputs.set(jnp.zeros(self.input_shape))
        self.outputs.set(jnp.zeros(self.output_shape))


class Outgrad(JaxComponent):
    """
    Compute the Jacobian matrix multiplication for the logits gradients.
    This computes: dL/dmu = J_softmax(mu)^T @ dL/dP
    where mu are the logits (pre-softmax)
    """
    
    def __init__(self, name, batch_size, seq_len, vocab_size, **kwargs):
        super().__init__(name, **kwargs)

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Receives the POST-activation probabilities (zF) to bypass redundant softmax computation
        self.mu = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        self.target = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        self.dmu = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        self.dmu_ = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
   
    @compilable   
    def advance_state(self):
        """Compute the softmax cross-entropy gradient from logits and targets."""
        logits = self.mu.get()
        target = self.target.get()
        probs = jax.nn.softmax(logits, axis=-1)

        # Gradient of CE wrt logits.
        self.dmu_.set(probs - target)
        
    @compilable
    def reset(self):
        """Reset compartments to zeros"""
        zeros = jnp.zeros((self.batch_size * self.seq_len, self.vocab_size))
        self.mu.set(zeros)
        self.target.set(zeros)
        self.dmu.set(zeros)
        self.dmu_.set(zeros)