import time
import jax
import jax.numpy as jnp
from jax import random
import gc

from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

# --- SPEED OPTIMIZATION 1: Enable TensorFloat-32 ---
# This makes matrix multiplication significantly faster on NVIDIA GPUs (Ampere+)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

def main():
    # ----------------------------
    # Config
    # ----------------------------
    seq_len = config.seq_len
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    num_iter = config.epoch
    dkey = random.PRNGKey(1234)

    print("\n RUNNING GPU-OPTIMIZED VERSION (TF32 + bfloat16)")
    print(f"Vocab: {vocab_size} | Epochs: {num_iter} | Batch: {batch_size}")
    print("-" * 50)

    # Load Data
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    # Initialize Model
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
    # Training Loop
    # ----------------------------
    for i in range(num_iter):
        print(f"\nIteration (Epoch) {i}")
        
        train_EFE = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            # --- SPEED OPTIMIZATION 2: Pre-cast to GPU ---
            # Move data to GPU immediately and convert to bfloat16.
            # bfloat16 is 2x faster and uses 50% less RAM than float32.
            inputs = jax.device_put(batch[0][1]).astype(jnp.bfloat16)
            targets = jax.device_put(batch[1][1])

            # One-hot encode
            targets_onehot = jax.nn.one_hot(targets, vocab_size)
            targets_flat = targets_onehot.reshape(-1, vocab_size)
            
            # --- MODEL PROCESS (Standard, no JIT wrapper) ---
            # We don't JIT this wrapper, but bfloat16 makes the internal math fast.
            yMu_inf, _, _EFE = model.process(
                obs=inputs,
                lab=targets_flat,
                adapt_synapses=True
            )

            # Block ensures we wait for GPU to finish before stopping timer
            yMu_inf.block_until_ready()
            
            step_duration = time.time() - step_start
            train_EFE += _EFE
            total_batches += 1

            # Logging every 10 batches
            if batch_idx % 10 == 0:
                # Cast back to float32 for metric calculation (precision matters for stats)
                y_pred = yMu_inf.reshape(-1, vocab_size).astype(jnp.float32)
                
                batch_nll = measure_CatNLL(y_pred, targets_flat)
                batch_ce = batch_nll.mean()
                batch_ppl = jnp.exp(batch_ce)

                print(
                    f"Batch {batch_idx:03d} | "
                    f"Time: {step_duration:.4f}s | "
                    f"EFE: {_EFE:.4f} | "
                    f"CE: {batch_ce:.4f} | "
                    f"PPL: {batch_ppl:.2f}"
                )
                
                # --- MEMORY SAFETY ---
                # Explicitly delete heavy tensors to prevent 'Cannot allocate memory'
                del y_pred, batch_nll
        
        # End of Epoch Metrics
        avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0.0
        
        # Manual Garbage Collection to keep JAX clean
        gc.collect()

        # Validation
        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
        print(f"--- Iter {i} Summary: Dev CE={dev_ce:.4f}, Dev PPL={dev_ppl:.4f}, Avg EFE={avg_train_EFE:.4f} ---")

        if i == num_iter - 1:
            model.save_to_disk(params_only=False)

    print("\nTRAINING FINISHED")

if __name__ == "__main__":
    main()