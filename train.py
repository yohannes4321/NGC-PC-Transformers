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
#  OPTIMIZATION: JIT-compiled training step
# ------------------------------------------------------------------
@partial(jax.jit, static_argnames=['adapt_synapses'])
def train_step(model, inputs, targets_flat, adapt_synapses=True):
    yMu_inf, _, _EFE = model.process(
        obs=inputs,
        lab=targets_flat,
        adapt_synapses=adapt_synapses
    )
    return model, yMu_inf, _EFE

def main():
    # START THE CLOCK
    total_start_time = time.time()

    # ----------------------------
    # Config
    # ----------------------------
    seq_len = config.seq_len
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    num_iter = config.num_iter
    dkey = random.PRNGKey(1234)

    print("\n✅ RUNNING OPTIMIZED VERSION (jax.nn.one_hot + JIT)")
    print(f"Vocab size: {vocab_size}")

    # ----------------------------
    # Data & Model
    # ----------------------------
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
        print(f"\nIter {i}:")
        
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time() # Start timer for specific step

            inputs = batch[0][1]
            targets = batch[1][1]

            # ✅ OPTIMIZED: No large identity matrix created
            targets_onehot = jax.nn.one_hot(targets, vocab_size)
            targets_flat = targets_onehot.reshape(-1, vocab_size)
            
            # MEMORY CHECK: Calculate size of the actual encoded batch in MB
            one_hot_mb = targets_onehot.nbytes / (1024**2)

            # JIT EXECUTION
            model, yMu_inf, _EFE = train_step(model, inputs, targets_flat)

            # FORCE SYNC for accurate timing
            jax.block_until_ready(yMu_inf)
            step_duration = time.time() - step_start

            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)
                y_true = jax.nn.one_hot(targets.flatten(), vocab_size)
                
                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce = batch_nll.mean()

                print(
                    f"  Batch {batch_idx:04d} | "
                    f"StepTime: {step_duration:.4f}s | "
                    f"Memory: {one_hot_mb:.2f} MB | "
                    f"CE: {batch_ce:.4f}"
                )

        # Eval
        dev_ce, _ = eval_model(model, valid_loader, vocab_size)
        print(f"--- Iter {i} Summary: Dev CE={dev_ce:.4f} ---")

    # FINAL TOTAL TIME
    total_duration = time.time() - total_start_time
    print("\n" + "="*50)
    print("✅ OPTIMIZED TRAINING FINISHED")
    print(f"Total Program Runtime: {total_duration:.2f} seconds")
    print(f"Efficiency: One-hot only used {one_hot_mb:.2f} MB per batch.")
    print("="*50)

if __name__ == "__main__":
    main()