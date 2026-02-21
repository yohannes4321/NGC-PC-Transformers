import time
import os
import gc
import psutil
import jax
import jax.numpy as jnp
from jax import random
import optax

# Import your custom modules
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

# --- OPTIMIZATION: Enable TensorFloat-32 ---
jax.config.update("jax_default_matmul_precision", "tensorfloat32") 

def log_mem(label):
    """Logs the current Resident Set Size (RSS) memory usage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2) 
    print(f"--- [CLEANUP LOG] {label} | Resident Memory: {mem:.2f} MB ---")

def main():
    log_mem("INITIAL STARTUP")
    
    seq_len = config.seq_len
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    dkey = random.PRNGKey(1234)
    
    # 1. Initialize Data Loader
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    schedule_seq_lens = [max(8, seq_len // 4), max(16, seq_len // 2), seq_len]
    
    # 2. Initialize Model
    # Mixed precision: bfloat16 for model weights and activations is recommended
    model = NGCTransformer(
        dkey=dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1.0, tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.eta, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer"
    )

    devices = jax.devices()
    num_devices = len(devices)
    print(f"Using {num_devices} device(s).")

    # 3. Define Train Step (Parallelized across GPUs if available)
    @jax.pmap
    def train_step(params, inputs, targets_flat, opt_state):
        def loss_fn(params, inputs, targets_flat):
            # Note: ngclearn models are inherently stateful. 
            # Ensure model.process works correctly inside a JAX transformation like pmap.
            yMu_inf, _, batch_efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            y_pred = yMu_inf.reshape(-1, vocab_size)
            batch_nll = measure_CatNLL(y_pred, targets_flat)
            return batch_nll.mean(), batch_efe

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, batch_efe), grads = grad_fn(params, inputs, targets_flat)
        
        # Optimizer updates
        grads = optax.clip_by_global_norm(grads, 1.0)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return loss, batch_efe, params, opt_state

    # 4. Training Loop (Curriculum Learning over Sequence Lengths)
    start_time = time.time()
    best_loss = float('inf')
    patience_limit = 5
    
    for scheduled_len in schedule_seq_lens:
        print(f"\n========================================")
        print(f" Training with sequence length: {scheduled_len} ")
        print(f"========================================")
        
        train_loader, valid_loader, _ = data_loader.load_and_prepare_data(schedule_seq_len=scheduled_len)
        num_batches = sum(1 for _ in train_loader)
        
        # Re-initialize loaders after counting batches
        train_loader, valid_loader, _ = data_loader.load_and_prepare_data(schedule_seq_len=scheduled_len)
        
        # Scheduler & Optimizer (re-initialized per sequence length phase, or you can move this outside)
        warmup_steps = max(1, int(0.01 * config.epoch * num_batches))
        total_steps = config.epoch * num_batches
        schedule = optax.join_schedules([
            optax.linear_schedule(init_value=0.1 * config.lr, end_value=config.lr, transition_steps=warmup_steps),
            optax.cosine_decay_schedule(init_value=config.lr, decay_steps=total_steps - warmup_steps)
        ], [warmup_steps])
        
        optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
        
        # Placeholder: Extract actual model parameters if optax is managing them, 
        # otherwise ngclearn manages weights internally.
        params = model 
        opt_state = optimizer.init(params)
        patience_counter = 0

        for epoch in range(config.epoch):
            print(f"\n>> Starting Epoch {epoch}")
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                step_start = time.time()
                
                # Setup data and cast to mixed precision
                inputs = jax.device_put(batch[0][1]).astype(jnp.bfloat16)
                targets = jax.device_put(batch[1][1])
                targets_flat = jax.nn.one_hot(targets, vocab_size).astype(jnp.bfloat16).reshape(-1, vocab_size)
                
                # Reshape for pmap (Batch size must be divisible by num_devices)
                inputs = inputs.reshape(num_devices, -1, inputs.shape[-1])
                targets_flat = targets_flat.reshape(num_devices, -1, targets_flat.shape[-1])
                
                # Execute Forward/Backward pass
                loss, batch_efe, params, opt_state = train_step(params, inputs, targets_flat, opt_state)
                epoch_loss += float(jnp.mean(loss))
                
                step_duration = time.time() - step_start
                
                # Clean up memory
                del inputs, targets, targets_flat
                gc.collect()
                
                # Print Metrics Every 10 Batches
                if batch_idx % 10 == 0:
                    batch_ce_loss = float(jnp.mean(loss))
                    batch_ppl = float(jnp.exp(batch_ce_loss))
                    mean_efe = float(jnp.mean(batch_efe))
                    
                    log_mem(f"Epoch {epoch} Batch {batch_idx}")
                    print(
                        f"Step Time: {step_duration:.4f}s | "
                        f"EFE = {mean_efe:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}"
                    )
            
            # Epoch wrap-up and Early Stopping
            epoch_loss /= num_batches
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print("Early stopping triggered for this sequence length phase.")
                    break
                    
            try:
                peak_mem = jax.devices()[0].memory_stats()['max_allocated_bytes'] / 1e9
                print(f"Max memory used on Device 0: {peak_mem:.2f} GB")
            except AttributeError:
                pass # memory_stats not available on all backends (like CPU)

    print(f"\nTotal Training Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()