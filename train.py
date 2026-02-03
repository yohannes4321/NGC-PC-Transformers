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
    num_iter = config.num_iter
    wub = config.wub
    wlb = config.wlb
    eta = config.eta
    T = config.n_iter
    tau_m = config.tau_m
    act_fx = config.act_fx
    dropout_rate = config.dropout_rate

    dkey = random.PRNGKey(1234)

    # ----------------------------
    # Data
    # ----------------------------
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, test_loader = data_loader.load_and_prepare_data()

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

    # ----------------------------
    # Train-only CE evaluation
    # ----------------------------
    # def train_model(data_loader):
    #     total_nll = 0.0
    #     total_tokens = 0

    #     for batch in data_loader:
    #         inputs = batch[0][1]
    #         targets = batch[1][1]  # (B, S)

    #         targets_onehot = jax.nn.one_hot(targets, vocab_size)  # (B, S, V)
    #         targets_flat = targets_onehot.reshape(-1, vocab_size)

    #         yMu_inf, _, _ = model.process(
    #             obs=inputs,
    #             lab=targets_flat,
    #             adapt_synapses=False
    #         )

    #         y_pred = yMu_inf.reshape(-1, vocab_size)

    #         nll = measure_CatNLL(y_pred, targets_flat)
    #         total_nll += nll * targets_flat.shape[0]
    #         total_tokens += targets_flat.shape[0]

    #     ce_loss = total_nll / total_tokens
    #     ppl = jnp.exp(ce_loss)
    #     return ce_loss, ppl

    # ----------------------------
    # Training loop
    # ----------------------------
    for i in range(num_iter):
        train_EFE = 0.0
        total_batches = 0

        print(f"\nIter {i}:")

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[0][1]
            targets = batch[1][1]  # (B, S)

            # ✅ one_hot (NO jnp.eye)
            targets_onehot = jax.nn.one_hot(targets, vocab_size)
            targets_flat = targets_onehot.reshape(-1, vocab_size)

            yMu_inf, _, _EFE = model.process(
                obs=inputs,
                lab=targets_flat,
                adapt_synapses=True
            )

            train_EFE += _EFE
            total_batches += 1

            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)
                y_true = jax.nn.one_hot(
                    targets.flatten(), vocab_size
                )

                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce = batch_nll.mean()
                batch_ppl = jnp.exp(batch_ce)

                print(
                    f"  Batch {batch_idx:04d} | "
                    f"EFE={_EFE:.4f} | "
                    f"CE={batch_ce:.4f} | "
                    f"PPL={batch_ppl:.4f}"
                )

        avg_train_EFE = train_EFE / max(total_batches, 1)

        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)

        print(
            f"Iter {i} Summary | "
            f"Dev CE={dev_ce:.4f} | "
            f"Dev PPL={dev_ppl:.4f} | "
            f"Avg EFE={avg_train_EFE:.4f}"
        )

        if i == num_iter - 1:
            model.save_to_disk(params_only=False)


if __name__ == "__main__":
    main()
