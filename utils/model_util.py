

import jax
from jax import numpy as jnp
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable

def d_softmax_vjp(P, v):
    """
    Computes the Vector-Jacobian Product (VJP) for Softmax efficiently.
    This calculates: dL/dmu = J^T @ v = P * (v - sum(P * v, axis=-1))
    
    Args:
        P: Softmax probabilities output from the cell (same shape as v)
        v: Upstream error vector (dL/dP) from the error cell
    Returns:
        Gradients with respect to the pre-softmax logits (dL/dmu)
    """
    # Vectorized dot product sum(p * v) along the vocabulary/class dimension
    p_dot_v = jnp.sum(P * v, axis=-1, keepdims=True)
    
    # Exact VJP identity for categorical distributions
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
        output=self.inputs.reshape(self.output_shape)
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
        self.dmu = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        self.dmu_ = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
   
    @compilable   
    def advance_state(self):
        """Compute the output gradients using the VJP function"""
        P = self.mu.get()        
        dmu = self.dmu.get()      
        
        # Map the upstream error backward through the Softmax manifold
        dmu_out = d_softmax_vjp(P, dmu)
        
        self.dmu_.set(dmu_out)
        
    @compilable
    def reset(self):
        """Reset compartments to zeros"""
        zeros = jnp.zeros((self.batch_size * self.seq_len, self.vocab_size))
        self.mu.set(zeros)
        self.dmu.set(zeros)
        self.dmu_.set(zeros)