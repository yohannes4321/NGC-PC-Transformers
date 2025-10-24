
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
import ngclearn.utils.weight_distribution as dist
from config import Config as config
from utils.embed_utils import EmbeddingSynapse
from jax import random

class EMBEDDING:
    """
   embedding layer using the EmbeddingSynapse
    """
    def __init__(self, dkey, vocab_size=config.vocab_size, seq_len=config.seq_len, embed_dim=config.n_embed, batch_size=config.batch_size, pos_learnable=config.pos_learnable, eta=config.eta, optim_type=config.optim, **kwargs):
        
        dkey, *subkeys = random.split(dkey, 4)
    
        # RateCell expects a 3D shape tuple for image components (seq_len, embed_dim, channels)so here we use the third dim as a placeholder
        self.z_embed = RateCell("z_embed", n_units=embed_dim, tau_m=0, 
                                  act_fx="identity", shape=(seq_len, embed_dim, 1), 
                                  batch_size=batch_size)
            
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
            
        self.e_embed = ErrorCell("e_embed", n_units=embed_dim)
    def get_embedding_weights(self):
        """Get both word and position embeddings"""
        return {
                'word_embeddings': self.W_embed.word_weights.value,
                'position_embeddings': self.W_embed.pos_weights.value
        }   
            

