import jax
from jax import numpy as jnp, random
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model
import time
import os
import psutil


def log_mem(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"--- [MEM LOG] {label} | Resident Memory: {mem:.2f} MB ---")


def main():
    log_mem("INITIAL STARTUP")
    # ---- config ----
    seq_len = config.seq_len
    batch_size = config.batch_size
    n_embed = config.n_embed
    vocab_size = config.vocab_size
    n_layers = config.n_layers
    n_heads = config.n_heads
    n_iter = config.n_iter
    optim_type = config.optim_type

    pos_learnable = config.pos_learnable
    epoch = config.epoch
    wub = config.wub
    wlb = config.wlb
    eta = config.eta
    T = config.n_iter
    tau_m = config.tau_m
    act_fx = config.act_fx
    dropout_rate = config.dropout_rate

    dkey = random.PRNGKey(1234)

    # ---- data ----
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, test_loader = data_loader.load_and_prepare_data()

    # ---- model ----
    model = NGCTransformer(
        dkey,
        batch_size=batch_size,
        seq_len=seq_len,
        n_embed=n_embed,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        T=T,
        dt=1.0,
        tau_m=tau_m,
        act_fx=act_fx,
        eta=eta,
        dropout_rate=dropout_rate,
        exp_dir="exp",
        loadDir=None,
        pos_learnable=pos_learnable,
        optim_type=optim_type,
        wub=wub,
        wlb=wlb,
        model_name="ngc_transformer",
    )

    # ---- eval-style loss ----
    def train_model(data_loader):
        total_nll = 0.0
        total_tokens = 0

        for batch in data_loader:
            inputs = batch[0][1]
            targets = batch[1][1]  # (B, S)

            targets_onehot = jax.nn.one_hot(targets, vocab_size)  # (B, S, V)
            targets_flat = targets_onehot.reshape(-1, vocab_size)  # (B*S, V)

            yMu_inf, _, _ = model.process(
                obs=inputs,
                lab=targets_flat,
                adapt_synapses=False,
            )

            y_pred = yMu_inf.reshape(-1, vocab_size)  # (B*S, V)
            y_true = targets_flat                     # (B*S, V)

            total_nll += measure_CatNLL(y_pred, y_true) * y_true.shape[0]
            total_tokens += y_true.shape[0]

        ce_loss = total_nll / total_tokens
        return ce_loss, jnp.exp(ce_loss)

    start_time = time.time()

    # ---- training loop ----
    for i in range(epoch):
        train_EFE = 0.0
        total_batches = 0

        print(f"\niter {i}:")

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            inputs = batch[0][1]
            targets = batch[1][1]  # (B, S)

            # one-hot + flatten ONCE
            targets_onehot = jax.nn.one_hot(targets, vocab_size)   # (B, S, V)
            targets_flat = targets_onehot.reshape(-1, vocab_size) # (B*S, V)

            yMu_inf, _, _EFE = model.process(
                obs=inputs,
                lab=targets_flat,
                adapt_synapses=True,
            )

            train_EFE += _EFE
            total_batches += 1

            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)  # (B*S, V)
                y_true = targets_flat                    # (B*S, V)

                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce_loss = batch_nll.mean()
                batch_ppl = jnp.exp(batch_ce_loss)
                step_duration = time.time() - step_start

                print(
                    f"  Batch {batch_idx}: "
                    f"EFE = {_EFE:.4f}, "
                    f"CE = {batch_ce_loss:.4f}, "
                    f"PPL = {batch_ppl:.4f}"
                )
                print(f"  Step Time: {step_duration:.4f}s")
                log_mem(f"Epoch {i} Batch {batch_idx}")

        avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0.0

        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
        print(
            f"Iter {i} Summary: "
            f"CE = {dev_ce:.4f}, "
            f"PPL = {dev_ppl:.4f}, "
            f"Avg EFE = {avg_train_EFE:.4f}"
        )

        if i == epoch - 1:
            model.save_to_disk(params_only=False)

    print(f"Total Time: {time.time() - start_time:.2f}s")
    print("\nTraining finished.")


if __name__ == "__main__":
    main()