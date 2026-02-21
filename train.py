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
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    # Sequence length scheduling: start short, increase
    schedule_seq_lens = [max(8, seq_len // 4), max(16, seq_len // 2), seq_len]
    for scheduled_len in schedule_seq_lens:
        print(f"Training with sequence length: {scheduled_len}")
        train_loader, valid_loader, _ = data_loader.load_and_prepare_data(schedule_seq_len=scheduled_len)
        # ...existing code for training loop...

    # Mixed precision: bfloat16 for model weights and activations
    model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1.0, tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.eta, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer")

    start_time = time.time()
    # JIT-compiled step function
    import optax
    # Gradient clipping and accumulation
    import optax
    # Sequence length scheduling: start short, increase
    schedule_seq_lens = [max(8, seq_len // 4), max(16, seq_len // 2), seq_len]
    for scheduled_len in schedule_seq_lens:
        print(f"Training with sequence length: {scheduled_len}")
        train_loader, valid_loader, _ = data_loader.load_and_prepare_data(schedule_seq_len=scheduled_len)
        # Compute number of batches by iterating through train_loader
        num_batches = sum(1 for _ in train_loader)
        warmup_steps = max(1, int(0.01 * config.epoch * num_batches))
        total_steps = config.epoch * num_batches
        # Reset train_loader iterator after counting
        train_loader, valid_loader, _ = data_loader.load_and_prepare_data(schedule_seq_len=scheduled_len)
        for i in range(config.epoch):
            print(f"\n>> Starting Epoch {i}")
            for batch_idx, batch in enumerate(train_loader):
                step_start = time.time()
                inputs = jax.device_put(batch[0][1]).astype(jnp.bfloat16)
                targets = jax.device_put(batch[1][1])
                targets_flat = jax.nn.one_hot(targets, config.vocab_size).astype(jnp.bfloat16).reshape(-1, config.vocab_size)
                yMu_inf, _, batch_efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
                yMu_inf.block_until_ready()
                y_pred = yMu_inf.reshape(-1, config.vocab_size)
                y_true = targets_flat
                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce_loss = batch_nll.mean()
                batch_ppl = jnp.exp(batch_ce_loss)
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}: EFE = {batch_efe:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}")
    schedule = optax.join_schedules([
        optax.linear_schedule(init_value=0.1 * config.lr, end_value=config.lr, transition_steps=warmup_steps),
        optax.cosine_decay_schedule(init_value=config.lr, decay_steps=total_steps - warmup_steps)
    ], [warmup_steps])
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
    accum_steps = 4
    # Multi-GPU parallelism
    devices = jax.devices()
    num_devices = len(devices)
    @jax.pmap
    def train_step(params, inputs, targets_flat, opt_state):
        def loss_fn(params, inputs, targets_flat):
            yMu_inf, _, batch_efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            y_pred = yMu_inf.reshape(-1, vocab_size)
            y_true = targets_flat
            batch_nll = measure_CatNLL(y_pred, y_true)
            return batch_nll.mean(), batch_efe
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, batch_efe), grads = grad_fn(params, inputs, targets_flat)
        grads = optax.clip_by_global_norm(grads, 1.0)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, batch_efe, params, opt_state

    params = model  # Placeholder: replace with model parameters extraction
    opt_state = optimizer.init(params)
    patience_limit = 5
    patience_counter = 0
    best_loss = float('inf')
    for i in range(config.epoch):
        print(f"\n>> Starting Epoch {i}")
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            # Split batch for devices
            inputs = jax.device_put(batch[0][1]).astype(jnp.bfloat16)
            targets = jax.device_put(batch[1][1])
            targets_flat = jax.nn.one_hot(targets, vocab_size).astype(jnp.bfloat16).reshape(-1, vocab_size)
            # Reshape for pmap
            inputs = inputs.reshape(num_devices, -1, inputs.shape[-1])
            targets_flat = targets_flat.reshape(num_devices, -1, targets_flat.shape[-1])
            loss, batch_efe, params, opt_state = train_step(params, inputs, targets_flat, opt_state)
            epoch_loss += float(jnp.mean(loss))
            step_duration = time.time() - step_start
            if batch_idx % 10 == 0:
                batch_ce_loss = jnp.mean(loss)
                batch_ppl = jnp.exp(batch_ce_loss)
            else:
                batch_ce_loss = None
                batch_ppl = None
            del inputs, targets, targets_flat
          
            if batch_idx % 10 == 0:
                log_mem(f"Epoch {i} Batch {batch_idx}")
                print(
                    f"Step Time: {step_duration:.4f}s | "
                    f"EFE = {jnp.mean(batch_efe):.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}"
                )
        epoch_loss /= (batch_idx + 1)
        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break
        # Log peak memory usage
        print(f"Max memory used: {jax.devices()[0].memory_stats()['max_allocated_bytes'] / 1e9:.2f} GB")
    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()