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

# --- OPTIMIZATION: Enable TensorFloat-32 ---
jax.config.update("jax_default_matmul_precision", "tensorfloat32") 

def main():
    print("--- TRAIN START ---")

    seq_len, batch_size, vocab_size = config.seq_len, config.batch_size, config.vocab_size
    dkey = random.PRNGKey(1234)
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    # Model must be initialized only once, outside the epoch loop
    model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1.0, tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.eta, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer")

    def train_step(inputs, targets_flat):
        # Only one process call per batch, returns yMu_inf, yMu, batch_efe
        return model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)

    start_time = time.time()

    for i in range(config.epoch):
        print(f"\n>> Starting Epoch {i}")
        ten_batch_time = 0.0
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            # Move data to device and prepare batch
            inputs = jax.device_put(batch[0][1]).astype(jnp.bfloat16)
            targets = jax.device_put(batch[1][1])
            targets_flat = jax.nn.one_hot(targets, vocab_size).reshape(-1, vocab_size)

            # Only one process call per batch
            yMu_inf, _, batch_efe = train_step(inputs, targets_flat)

            step_duration = time.time() - step_start
            ten_batch_time += step_duration

            # Only extract metrics for logging every 10 batches
            if batch_idx % 10 == 0:
                yMu_inf.block_until_ready()
                y_pred = yMu_inf.reshape(-1, vocab_size)
                y_true = targets_flat
                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce_loss = batch_nll.mean()
                batch_ppl = jnp.exp(batch_ce_loss)
            else:
                batch_ce_loss = None
                batch_ppl = None

            # Light-weight cleanup only every 50 batches
            if batch_idx % 50 == 0:
                gc.collect()

            # Print/log every 10 batches
            if batch_idx % 10 == 0 and batch_idx != 0:
                print(
                    f"Total Time for last 10 batches: {ten_batch_time:.4f}s | "
                    f" batch {batch_idx} ,EFE = {batch_efe:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}"
                )
                ten_batch_time = 0.0
            elif batch_idx % 10 == 0 and batch_idx == 0:
                print(
                    f"Step Time: {step_duration:.4f}s | "
                    f"EFE = {batch_efe:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}"
                )
        # extra GC once per epoch
        gc.collect()
    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
