
from utils.errorcell import GaussianErrorCell as ErrorCell
from utils.ratecell import RateCell
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from config import Config as config
from utils.embed_utils import EmbeddingSynapse
from jax import random

class EMBEDDING:
    """
   embedding layer using the EmbeddingSynapse
    
    CORRECTED: 
    - z_embed now stores embeddings (n_units=embed_dim, batch_size=batch_size*seq_len)
    - W_embed processes token IDs -> embeddings
    - e_embed receives error signals from predictive coding
    """
    def __init__(self, dkey, vocab_size, seq_len, embed_dim, batch_size, pos_learnable, eta, optim_type, **kwargs):
        
        dkey, *subkeys = random.split(dkey, 4)
    
        # CORRECTED: z_embed stores embeddings, not token IDs
        # Shape should be (batch_size*seq_len, embed_dim) to match output of W_embed
        self.z_embed = RateCell("z_embed", n_units=embed_dim, tau_m=0., 
                                  act_fx="identity", batch_size=batch_size * seq_len)            
            # EmbeddingSynapse (handles both word + position internally)
        self.W_embed = EmbeddingSynapse(
                "W_embed", 
                vocab_size=vocab_size,
                seq_len=seq_len,
                embed_dim=embed_dim, 
                batch_size=batch_size,
                pos_learnable=pos_learnable,
                eta=eta,
                optim_type=optim_type,
                key=subkeys[0])
            
        self.e_embed = ErrorCell("e_embed", n_units=embed_dim, 
                                  batch_size=batch_size * seq_len) # shape=(batch_size*seq_len, embed_dim)
    
            

