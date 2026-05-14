import jax
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable
from jax import numpy as jnp, random, jit
from functools import partial
from jax import vmap
class Outgrad(JaxComponent):
    """
    Output gradient for NGC generative model with cross-entropy + softmax.
    
    In NGC autoregressive setting:
      - z^0 is clamped to input (the data)
      - Model predicts z^0 from latent states via softmax(W_out @ z_hidden)
      - Target is one-hot encoded actual token
    
    Paper Eq.9: e^0 = x - z^0
      where x = one-hot target (clamped input)
            z^0 = softmax probabilities (model prediction)
    
    For cross-entropy + softmax combined, gradient w.r.t. logits is:
      dL/d_mu = P - target  (negated for gradient descent: target - P)
    
    The softmax Jacobian multiplication is NOT needed here because:
      d(CE)/d(logits) = P - target  analytically (Jacobian cancels)
    
    So this component simply computes: dmu_ = target - P
    where P = softmax(mu), matching paper Eq.9: e^0 = x - z^0
    """
    
    def __init__(self, name, batch_size, seq_len, vocab_size, **kwargs):
        super().__init__(name, **kwargs)

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # mu: raw logits (pre-softmax) from W_out
        self.mu = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        # target: one-hot encoded targets (clamped input in generative setting)
        self.target = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        # dmu_: gradient signal = target - P  (matches paper e^0 = x - z^0)
        self.dmu_ = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
   
    @compilable   
    def advance_state(self):
        """
        Compute e^0 = target - P  where P = softmax(mu)
        
        This is:
          - Paper Eq.9: e^0 = x - z^0
          - Analytically correct CE gradient w.r.t. logits (no Jacobian needed)
          - Positive = error signal for NGC (not negated, NGC uses e directly)
        """
        mu     = self.mu.get()       # (B*T, vocab) raw logits
        target = self.target.get()   # (B*T, vocab) one-hot clamped input
        
        P = jax.nn.softmax(mu, axis=-1)  # z^0 = model predictions
        
        # e^0 = x - z^0  (paper Eq.9, NGC error neuron convention)
        dmu_out = target - P
        
        self.dmu_.set(dmu_out)
        
    @compilable
    def reset(self):
        zeros = jnp.zeros((self.batch_size * self.seq_len, self.vocab_size))
        self.mu.set(zeros)
        self.target.set(zeros)
        self.dmu_.set(zeros)