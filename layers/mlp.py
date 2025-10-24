import jax
from ngclearn.utils import JaxProcess
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
import ngclearn.utils.weight_distribution as dist
from config import Config as config

class MLP:
    """
    NGC MLP layer with two Hebbian synapses and error cell.

    Minimal working implementation: creates z_mlp RateCell, two Hebbian synapses,
    an ErrorCell and a StaticSynapse for feedback. The constructor accepts
    `target` and `td_error` to match how the model wires layers.
    """

    def __init__(self, dkey,n_embed=config.n_embed, seq_len=config.seq_len,
                 batch_size=config.batch_size, vocab_size=config.vocab_size,
                 act_fx="identity", eta=config.eta, **kwargs):
        dkey, *subkeys = random.split(dkey, 10)
        optim_type = kwargs.get('optim_type', 'adam')
        wlb = -0.3
        wub = 0.3

        self.z_mlp = RateCell("z_mlp", n_units=n_embed, tau_m=1., act_fx="identity", shape=(seq_len, n_embed, 1), batch_size=batch_size)
        self.W_mlp1 = HebbianSynapse(
                    "W_mlp1", shape=(n_embed, 4*n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4])
        self.W_mlp2 = HebbianSynapse(
                    "W_mlp2", shape=(4*n_embed, n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[5])
        self.e_mlp = ErrorCell("e_mlp", n_units=n_embed)
                
                # self.E_mlp1 = StaticSynapse(
                #     "E_mlp1", shape=(n_embed, 4 * n_embed), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4])
        self.E_mlp = StaticSynapse(
                    "E_mlp", shape=(n_embed, 4*n_embed), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4])
    def get_components(self):
        """Return all components for easy access"""
        return {
            'z_mlp': self.z_mlp,
            'W_mlp1': self.W_mlp1,
            'W_mlp2': self.W_mlp2,
            'e_mlp': self.e_mlp,
            'E_mlp': self.E_mlp
        }

                