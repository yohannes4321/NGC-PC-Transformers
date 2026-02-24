import time
import jax
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
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1.0, tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.eta, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer")

    @jax.jit
    def train_step(inputs, targets_flat):
        # All data is already on device, call process directly
        return model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)

    start_time = time.time()
    for i in range(config.epoch):
        print(f"\n>> Starting Epoch {i}")
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            # Move data to device and cast
            inputs = jax.device_put(batch[0][1]).astype(jnp.bfloat16)
            targets = jax.device_put(batch[1][1])
            targets_flat = jax.nn.one_hot(targets, vocab_size).reshape(-1, vocab_size)

            yMu_inf, _, batch_efe = train_step(inputs, targets_flat)

            step_duration = time.time() - step_start

            if batch_idx % 10 == 0:
                # Only block for logging batches
                yMu_inf.block_until_ready()
                y_pred = yMu_inf.reshape(-1, vocab_size)
                y_true = targets_flat
                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce_loss = batch_nll.mean()
                batch_ppl = jnp.exp(batch_ce_loss)
            else:
                batch_ce_loss = None
                batch_ppl = None

            # --- THE CLEANUP LOGIC ---
            del inputs, targets, targets_flat, yMu_inf
            gc.collect()

            if batch_idx % 10 == 0:
                log_mem(f"Epoch {i} Batch {batch_idx}")
                print(
                    f"Step Time: {step_duration:.4f}s | "
                    f"EFE = {batch_efe:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}"
                )

    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()