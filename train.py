import time
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

# ------------------------------------------------------------------
# OPTIMIZATION: JIT-compiled training step
# ------------------------------------------------------------------
@partial(jax.jit, static_argnames=['model', 'adapt_synapses'])
def train_step(model, inputs, targets_flat, adapt_synapses=True):
    """
    JIT-compiled step using one_hot. 
    Returns the inference mean (yMu_inf) and the Expected Free Energy (_EFE).
    """
    yMu_inf, _, _EFE = model.process(
        obs=inputs,
        lab=targets_flat,
        adapt_synapses=adapt_synapses
    )
    # We return EFE so we can print it during training
    return yMu_inf, _EFE

def main():
    total_start_time = time.time()

    # ----------------------------
    # Config & Initialization
    # ----------------------------
    seq_len = config.seq_len
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    num_iter = config.epoch
    dkey = random.PRNGKey(1234)

    print("\n RUNNING OPTIMIZED JIT + ONE-HOT VERSION")
    print(f"Vocab size: {vocab_size} | Epochs: {num_iter}")
    print("-" * 50)

    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    model = NGCTransformer(
        dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1.0, tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.eta, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer",
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    for i in range(num_iter):
        print(f"\nIteration (Epoch) {i}")
        
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            # Data parsing
            inputs = batch[0][1]
            targets = batch[1][1]

            # EFFICIENT: No jnp.eye() matrix created. 
            # jax.nn.one_hot creates the representation lazily/efficiently.
            targets_onehot = jax.nn.one_hot(targets, vocab_size)
            targets_flat = targets_onehot.reshape(-1, vocab_size)
            
            # Tracking memory footprint of the actual batch
            one_hot_mb = targets_onehot.nbytes / (1024**2)

            # --- JIT EXECUTION ---
            yMu_inf, _EFE = train_step(model, inputs, targets_flat)

            # Block to ensure accurate timing
            yMu_inf.block_until_ready()
            step_duration = time.time() - step_start

            # Logging every 10 batches
            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)
                
                # Calculate Cross Entropy (CE)
                batch_nll = measure_CatNLL(y_pred, targets_flat)
                batch_ce = batch_nll.mean()
                
                # Calculate Perplexity (PPL)
                batch_ppl = jnp.exp(batch_ce)

                print(
                    f"Batch {batch_idx:03d} | "
                    f"Time: {step_duration:.4f}s | "
                    f"EFE: {_EFE:.4f} | "
                    f"CE: {batch_ce:.4f} | "
                    f"PPL: {batch_ppl:.2f}"
                )

        # Eval after each epoch
        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
        print(f"--- Iter {i} Summary: Dev CE={dev_ce:.4f}, Dev PPL={dev_ppl:.4f} ---")

    # FINAL REPORTING
    total_duration = time.time() - total_start_time
    print("\n" + "="*50)
    print(" OPTIMIZED TRAINING FINISHED")
    print(f"Total Program Runtime: {total_duration:.2f} seconds")
    print(f"Efficiency: Used jax.nn.one_hot ({one_hot_mb:.2f} MB) instead of jnp.eye.")
    print("="*50)

if __name__ == "__main__":
    main()
