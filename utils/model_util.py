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

    Used for ATTENTION score backpropagation only.
    NOT used for the output layer — see Outgrad below.

    Returns probabilities and a function that computes J @ v without forming
    the full Jacobian matrix.

    Args:
        x: Input tensor of any shape
           Examples:
           - Attention: (batch_size, n_heads, seq_len, seq_len)
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
        output = self.inputs.reshape(self.output_shape)
        self.outputs.set(output)

    @compilable
    def reset(self):
        self.inputs.set(jnp.zeros(self.input_shape))
        self.outputs.set(jnp.zeros(self.output_shape))


class Outgrad(JaxComponent):
    """
    Output error neuron for NGC generative model with cross-entropy + softmax.

    In the NGC autoregressive/generative setting:
      - z^0 is clamped to the input (the data itself)
      - The model predicts z^0 from latent states via softmax(W_out @ z_hidden)
      - target is the one-hot encoded actual token (clamped input)

    From the paper Eq.9:
        e^0 = x - z^0
    where:
        x   = one-hot target (clamped input)
        z^0 = softmax(mu) = model predicted probabilities

    For cross-entropy + softmax combined, the gradient w.r.t. logits simplifies
    analytically to:
        dL/d_mu = P - target
    and the NGC error signal (positive = corrective) is:
        e^0 = target - P

    The softmax Jacobian multiplication is NOT applied here because it cancels
    analytically when cross-entropy is the loss. Applying it again would
    double-transform the gradient and kill the learning signal.

    Wiring:
        W_out.outputs >> Outgrad.mu       (raw logits)
        z_target.z    >> Outgrad.target   (one-hot clamped input)
        Outgrad.dmu_  >> E_out.inputs     (error signal backward)
        Outgrad.dmu_  >> W_out.post       (Hebbian weight update, paper Eq.16)
    """

    def __init__(self, name, batch_size, seq_len, vocab_size, **kwargs):
        super().__init__(name, **kwargs)

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        # mu: raw logits (pre-softmax) from W_out
        self.mu = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        # target: one-hot encoded clamped input (the data)
        self.target = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))
        # dmu_: error signal = target - P  (paper Eq.9: e^0 = x - z^0)
        self.dmu_ = Compartment(jnp.zeros((batch_size * seq_len, vocab_size)))

    @compilable
    def advance_state(self):
        """
        Compute e^0 = target - P  where P = softmax(mu)

        This is simultaneously:
          - Paper Eq.9: e^0 = x - z^0
          - The exact analytic CE gradient w.r.t. logits (no Jacobian needed)
          - The NGC bottom-up error signal for the output layer
        """
        mu     = self.mu.get()      # (B*T, vocab) raw logits
        target = self.target.get()  # (B*T, vocab) one-hot clamped input

        P = jax.nn.softmax(mu, axis=-1)  # z^0 = model predicted probabilities

        # e^0 = x - z^0  (paper Eq.9)
        dmu_out = target - P

        self.dmu_.set(dmu_out)

    @compilable
    def reset(self):
        """Reset all compartments to zeros"""
        zeros = jnp.zeros((self.batch_size * self.seq_len, self.vocab_size))
        self.mu.set(zeros)
        self.target.set(zeros)
        self.dmu_.set(zeros)