import time
import gc
import os
import psutil
import jax
import jax.numpy as jnp
from jax import random
import optax

from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

# --------------------------------------------------
# Enable TensorFloat-32 for faster matmul on Ampere+
# --------------------------------------------------
jax.config.update("jax_default_matmul_precision", "tensorfloat32")


# --------------------------------------------------
# Memory Logging
# --------------------------------------------------
def log_mem(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"--- [MEM LOG] {label} | RAM: {mem:.2f} MB ---")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    log_mem("INITIAL STARTUP")

    seq_len = config.seq_len
    batch_size = config.batch_size
    vocab_size = config.vocab_size

    devices = jax.devices()
    num_devices = len(devices)
    print(f"Using {num_devices} device(s)")

    dkey = random.PRNGKey(1234)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)

    # Sequence length curriculum
    schedule_seq_lens = [
        max(8, seq_len // 4),
        max(16, seq_len // 2),
        seq_len
    ]

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = NGCTransformer(
        dkey,
        batch_size=batch_size,
        seq_len=seq_len,
        n_embed=config.n_embed,
        vocab_size=vocab_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        T=config.n_iter,
        dt=1.0,
        tau_m=config.tau_m,
        act_fx=config.act_fx,
        eta=config.eta,
        dropout_rate=config.dropout_rate,
        exp_dir="exp",
        loadDir=None,
        pos_learnable=config.pos_learnable,
        optim_type=config.optim_type,
        wub=config.wub,
        wlb=config.wlb,
        model_name="ngc_transformer"
    )

    # --------------------------------------------------
    # Extract parameters properly
    # --------------------------------------------------
    params = model.get_params()   # IMPORTANT: must return pytree

    # --------------------------------------------------
    # Optimizer (Warmup + Cosine)
    # --------------------------------------------------
    total_steps = config.epoch * 1000  # safe fallback estimate
    warmup_steps = int(0.01 * total_steps)

    schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0.1 * config.lr,
                end_value=config.lr,
                transition_steps=warmup_steps
            ),
            optax.cosine_decay_schedule(
                init_value=config.lr,
                decay_steps=total_steps - warmup_steps
            )
        ],
        [warmup_steps]
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=0.01)
    )

    opt_state = optimizer.init(params)

    # --------------------------------------------------
    # Training Step (PMAP)
    # --------------------------------------------------
    @jax.pmap
    def train_step(params, inputs, targets_flat, opt_state):

        def loss_fn(params):
            yMu_inf, _, batch_efe = model.process(
                obs=inputs,
                lab=targets_flat,
                adapt_synapses=True,
                params=params
            )

            y_pred = yMu_inf.reshape(-1, vocab_size)
            y_true = targets_flat

            loss = measure_CatNLL(y_pred, y_true).mean()
            return loss, batch_efe

        (loss, batch_efe), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, batch_efe, params, opt_state

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    best_loss = float("inf")
    patience_limit = 5
    patience_counter = 0

    start_time = time.time()

    for scheduled_len in schedule_seq_lens:

        print(f"\n==============================")
        print(f"Training with sequence length: {scheduled_len}")
        print(f"==============================")

        train_loader, valid_loader, _ = \
            data_loader.load_and_prepare_data(schedule_seq_len=scheduled_len)

        for epoch in range(config.epoch):

            print(f"\n>> Epoch {epoch}")
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):

                step_start = time.time()

                inputs = jax.device_put(batch[0][1]).astype(jnp.bfloat16)
                targets = jax.device_put(batch[1][1])

                targets_flat = (
                    jax.nn.one_hot(targets, vocab_size)
                    .astype(jnp.bfloat16)
                    .reshape(-1, vocab_size)
                )

                # reshape for multi-GPU
                inputs = inputs.reshape(num_devices, -1, inputs.shape[-1])
                targets_flat = targets_flat.reshape(
                    num_devices, -1, targets_flat.shape[-1]
                )

                loss, batch_efe, params, opt_state = train_step(
                    params, inputs, targets_flat, opt_state
                )

                batch_loss = float(jnp.mean(loss))
                epoch_loss += batch_loss

                if batch_idx % 10 == 0:
                    batch_ppl = float(jnp.exp(batch_loss))
                    step_time = time.time() - step_start

                    print(
                        f"Batch {batch_idx} | "
                        f"Loss: {batch_loss:.4f} | "
                        f"PPL: {batch_ppl:.4f} | "
                        f"EFE: {float(jnp.mean(batch_efe)):.4f} | "
                        f"Time: {step_time:.3f}s"
                    )

                    log_mem(f"Epoch {epoch} Batch {batch_idx}")

                del inputs, targets, targets_flat
                gc.collect()

            epoch_loss /= (batch_idx + 1)
            print(f"\nEpoch {epoch} Avg Loss: {epoch_loss:.4f}")

            # ---------------- Early Stopping ----------------
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print("Early stopping triggered.")
                    break

        # Validation
        eval_model(model, valid_loader)

    print(f"\nTotal Training Time: {time.time() - start_time:.2f}s")

    # GPU memory usage
    try:
        stats = jax.devices()[0].memory_stats()
        print(
            f"Max device memory used: "
            f"{stats['max_allocated_bytes'] / 1e9:.2f} GB"
        )
    except:
        pass


if __name__ == "__main__":
    main()