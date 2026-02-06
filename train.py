import time
import jax
import jax.numpy as jnp
from jax import random
import gc
import os
import psutil  # Requirement: pip install psutil
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

# --- OPTIMIZATION: Enable TensorFloat-32 ---
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

def log_stats(label):
    """Prints memory usage and time marker."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    print(f"[{label}] Memory: {mem:.2f} MB")

def main():
    log_stats("STARTUP")
    total_start_time = time.time() 
    
    seq_len, batch_size, vocab_size = config.seq_len, config.batch_size, config.vocab_size
    dkey = random.PRNGKey(1234)

    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1.0, tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.eta, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer")

    print(f"\n[STARTING GPU-OPTIMIZED TRAINING (TF32 + BF16 + GC)]")
    start_time = time.time()
    
    for i in range(config.epoch):
        train_EFE = 0.
        total_batches = 0
        
        # Log memory before epoch starts to check for leaks between epochs
        log_stats(f"BEGIN EPOCH {i}")
        
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            # --- OPTIMIZATION: Cast to bfloat16 ---
            inputs = jax.device_put(batch[0][1]).astype(jnp.bfloat16)
            targets = jax.device_put(batch[1][1])
            targets_flat = jax.nn.one_hot(targets, vocab_size).reshape(-1, vocab_size)
            
            yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            yMu_inf.block_until_ready() 
            
            step_duration = time.time() - step_start

            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)
                y_true = jnp.eye(vocab_size)[targets.flatten()]
                
                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce_loss = batch_nll.mean()  
                batch_ppl = jnp.exp(batch_ce_loss)
                print(f" Batch {batch_idx} | Time: {step_duration:.4f}s | EFE: {_EFE:.4f} | CE = {batch_ce_loss:.4f}")
        
            # --- OPTIMIZATION: Aggressive Cleanup ---
            del inputs, targets, targets_flat, yMu_inf
            gc.collect() # Force garbage collection
            
        avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0
        
        log_stats(f"END EPOCH {i} (After GC)") # Check memory after epoch cleanup
        
        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
        print(f"Iter {i} Summary: CE = {dev_ce:.4f}, PPL = {dev_ppl:.4f}")

        if i == (config.epoch-1):
          model.save_to_disk(params_only=False)

    total_time = time.time() - start_time
    print(f"\nTraining finished.")
    print(f"Total training time: {total_time:.0f} seconds")

if __name__ == "__main__":
    main()