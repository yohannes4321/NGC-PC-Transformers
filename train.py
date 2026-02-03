import time
import jax
from jax import numpy as jnp, random

from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model


def main():
    # ----------------------------
    # Config
    # ----------------------------
    seq_len = config.seq_len
    batch_size = config.batch_size
    n_embed = config.n_embed
    vocab_size = config.vocab_size
    n_layers = config.n_layers
    n_heads = config.n_heads
    n_iter = config.n_iter
    optim_type = config.optim_type

    pos_learnable = config.pos_learnable
    epoch = config.num_iter
    wub = config.wub
    wlb = config.wlb
    eta = config.eta
    T = config.n_iter
    tau_m = config.tau_m
    act_fx = config.act_fx
    dropout_rate = config.dropout_rate

    dkey = random.PRNGKey(1234)

    print("\n❌ RUNNING BASELINE (jnp.eye — MEMORY HEAVY)")
    print(f"Vocab size: {vocab_size}")
    print("This version allocates a V×V matrix every step.\n")

    # ----------------------------
    # Data
    # ----------------------------
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    # ----------------------------
    # Model
    # ----------------------------
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

    total_start_time = time.time()

    # ----------------------------
    # Training loop
    # ----------------------------
    for i in range(epoch):
        print(f"\nEpoch {i}")
        train_EFE = 0.0

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[0][1]
            targets = batch[1][1]  # (B, S)

            # ❌ BAD: allocates (V, V) identity matrix
            step_start = time.time()

            targets_onehot = jnp.eye(vocab_size)[targets]   # (B, S, V)
            targets_flat = targets_onehot.reshape(-1, vocab_size)

            yMu_inf, _, _EFE = model.process(
                obs=inputs,
                lab=targets_flat,
                adapt_synapses=True
            )

            # ⛔ FORCE SYNC so timing is REAL
            jax.block_until_ready(yMu_inf)

            step_time = time.time() - step_start
            train_EFE += _EFE

            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)

                # ❌ AGAIN: V×V allocation
                y_true = jnp.eye(vocab_size)[targets.flatten()]

                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce = batch_nll.mean()
                batch_ppl = jnp.exp(batch_ce)

                print(
                    f"[BAD] Batch {batch_idx:04d} | "
                    f"StepTime={step_time:.4f}s | "
                    f"EFE={_EFE:.4f} | "
                    f"CE={batch_ce:.4f} | "
                    f"PPL={batch_ppl:.4f}"
                )

        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)

        print(
            f"Epoch {i} Summary | "
            f"Dev CE={dev_ce:.4f} | "
            f"Dev PPL={dev_ppl:.4f}"
        )

    total_time = time.time() - total_start_time
    print("\n❌ BASELINE FINISHED")
    print(f"Total training time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
