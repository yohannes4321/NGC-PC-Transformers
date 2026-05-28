
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
    
        # z_embed: RateCell that holds token IDs (clamped from input)
        # Shape: (batch_size, seq_len) = (12, 64) for token IDs
        # z_embed.zF outputs token IDs with shape (batch_size, seq_len)
        self.z_embed = RateCell("z_embed", n_units=seq_len, tau_m=0., 
                                  act_fx="identity", batch_size=batch_size)
        
        # W_embed: EmbeddingSynapse in 3D mode (token IDs → embeddings)
        # Converts token IDs (batch_size, seq_len) to embeddings (batch_size, seq_len, embed_dim)
        # Using vocab_size x embed_dim word embeddings + learned position embeddings
        self.W_embed = EmbeddingSynapse(
                "W_embed",
                vocab_size=vocab_size,  # Vocabulary size for token→embedding lookup
                seq_len=seq_len,
                embed_dim=embed_dim,
                batch_size=batch_size,
                pos_learnable=pos_learnable,  # Position embeddings are learnable
                eta=eta,
                optim_type=optim_type,
                weight_scale=0.02,
                is_2d_mode=False,  # KEY: 3D mode for token→embedding conversion
                key=subkeys[0]
        )
            
        # e_embed: ErrorCell receives prediction error from first attention block
        # Shape: (batch_size*seq_len, embed_dim) = (768, 96)
        # This receives error signals after z_embed is reshaped
        self.e_embed = ErrorCell("e_embed", n_units=embed_dim, 
                                  batch_size=batch_size * seq_len)
    
            

