import jax
import time
import jax.numpy as jnp
from jax import random
import gc
import os
import psutil
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model
import time
import psutil
import os
import gc
import jax
from jax import random, numpy as jnp
# Assume config, DataLoader, NGCTransformer, measure_CatNLL are imported here

# --- OPTIMIZATION: Enable TensorFloat-32 ---
jax.config.update("jax_default_matmul_precision", "tensorfloat32") 

def log_mem(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2) 
    print(f"--- [CLEANUP LOG] {label} | Resident Memory: {mem:.2f} MB ---")

def main():
    log_mem("INITIAL STARTUP")
    seq_len, batch_size, vocab_size = config.seq_len, config.batch_size, config.vocab_size
    dkey = random.PRNGKey(1234)
    
    # Enable prefetch and pin_memory in DataLoader
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size, prefetch_batches=4, pin_memory=True)
    
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data(schedule_seq_len=seq_len)

    # Mixed precision: bfloat16 for model weights and activations
    model = NGCTransformer(
        dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1.0, tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.lr, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer"
    )

    start_time = time.time()
    patience_limit = 5
    patience_counter = 0
    best_loss = float('inf')

    for epoch in range(config.epoch):
        print(f"\n>> Starting Epoch {epoch}")
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            
            # 1. Cast and push data to the GPU immediately
            inputs = jax.device_put(batch[0][1]).astype(jnp.bfloat16)
            targets = jax.device_put(batch[1][1])
            targets_flat = jax.nn.one_hot(targets, vocab_size).astype(jnp.bfloat16).reshape(-1, vocab_size)
            
            # 2. Forward pass and internal Hebbian update (no value_and_grad needed!)
            # ngclearn's evolve.run() handles the optimization natively.
            yMu_inf, _, batch_efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            # 3. Calculate metrics purely for logging
            y_pred = yMu_inf.reshape(-1, vocab_size)
            batch_nll = measure_CatNLL(y_pred, targets_flat)
            loss_val = float(jnp.mean(batch_nll))
            epoch_loss += loss_val
            
            step_duration = time.time() - step_start
            
            if batch_idx % 10 == 0:
                batch_ppl = jnp.exp(loss_val)
                print(
                    f"Step Time: {step_duration:.4f}s | "
                    f"EFE = {float(jnp.mean(batch_efe)):.4f}, CE = {loss_val:.4f}, PPL = {batch_ppl:.4f}"
                )
                log_mem(f"Epoch {epoch} Batch {batch_idx}")

            del inputs, targets, targets_flat
            gc.collect()

        epoch_loss /= (batch_idx + 1)
        
        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            model.save_to_disk(params_only=True) # Save best weights
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break
                
        print(f"Max memory used: {jax.devices()[0].memory_stats()['max_allocated_bytes'] / 1e9:.2f} GB")

    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()