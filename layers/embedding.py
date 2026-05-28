
from utils.errorcell import GaussianErrorCell as ErrorCell
from utils.ratecell import RateCell
from utils.embed_utils import EmbeddingSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components import StaticSynapse
from config import Config as config
from jax import random

class EMBEDDING:
    """
    Predictive Coding Embedding Layer
    
    ARCHITECTURE:
    - z_embed: Clamped with pre-computed embeddings (batch_size*seq_len, embed_dim)
    - W_embed: EmbeddingSynapse (2D mode) projecting embeddings embed_dim → embed_dim
    - e_embed: ErrorCell for predictive coding error signals
    
    Token IDs are converted to embeddings EXTERNALLY (in generation.py),
    then clamped into z_embed. W_embed then projects to attention layer.
    """
    def __init__(self, dkey, vocab_size, seq_len, embed_dim, batch_size, pos_learnable, eta, optim_type, **kwargs):
        
        dkey, *subkeys = random.split(dkey, 4)
    
        # z_embed: RateCell that holds pre-computed embeddings
        # Shape: (batch_size*seq_len, embed_dim) = (768, 96)
        # This is CLAMPED with embeddings computed from token IDs externally
        self.z_embed = RateCell("z_embed", n_units=embed_dim, tau_m=0., 
                                  act_fx="identity", batch_size=batch_size * seq_len)
        
        # W_embed: EmbeddingSynapse in 2D projection mode
        # Projects embeddings (embed_dim) → projected embeddings (embed_dim)
        # Using learned weight matrix (embed_dim × embed_dim) with Hebbian learning
        self.W_embed = EmbeddingSynapse(
                "W_embed",
                vocab_size=embed_dim,  # In 2D mode, this indicates output dimension
                seq_len=seq_len,
                embed_dim=embed_dim,
                batch_size=batch_size,
                pos_learnable=False,  # No positional encoding in 2D mode
                eta=eta,
                optim_type=optim_type,
                weight_scale=0.02,
                is_2d_mode=True,  # KEY: 2D projection mode
                key=subkeys[0]
        )
            
        # e_embed: ErrorCell receives prediction error from first attention block
        # Shape: (batch_size*seq_len, embed_dim) = (768, 96)
        self.e_embed = ErrorCell("e_embed", n_units=embed_dim, 
                                  batch_size=batch_size * seq_len)
    
            

