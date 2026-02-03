import jax
import jax.numpy as jnp
from jax import random
from functools import partial  # Required for jit static arguments

from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

# ------------------------------------------------------------------
#  OPTIMIZATION: Define the JIT-compiled training step
# ------------------------------------------------------------------
# We compile this function once. 
# 'static_argnames' tells JAX to re-compile only if 'adapt_synapses' changes value.
@partial(jax.jit, static_argnames=['adapt_synapses'])
def train_step(model, inputs, targets_flat, adapt_synapses=True):
    # process() returns the outputs and the internal metric
    yMu_inf, _, _EFE = model.process(
        obs=inputs,
        lab=targets_flat,
        adapt_synapses=adapt_synapses
    )
    # CRITICAL: We must return the 'model' because JAX updates are functional.
    # The 'model' object returned here contains the updated synaptic weights.
    return model, yMu_inf, _EFE

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
    num_iter = config.epoch
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
    print("Initializing Model...")
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
    # Training loop
    # ----------------------------
    print("Starting Training...")
    
    # Optional: Trigger one compilation pass before the loop to separate compile time from train time
    # print("Compiling JIT function...")
    # dummy_in = jnp.zeros((batch_size, seq_len))
    # dummy_tar = jnp.zeros((batch_size * seq_len, vocab_size))
    # train_step(model, dummy_in, dummy_tar, True)
    # print("Compilation complete.")

    for i in range(num_iter):
        train_EFE = 0.0
        total_batches = 0

        print(f"\nIter {i}:")

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[0][1]
            targets = batch[1][1]  # (B, S)

            # ✅ one_hot
            targets_onehot = jax.nn.one_hot(targets, vocab_size)
            targets_flat = targets_onehot.reshape(-1, vocab_size)

            # -------------------------------------------------------
            #  FAST JIT EXECUTION
            # -------------------------------------------------------
            # Note: We overwrite 'model' with the result. 
            # In JAX, the state is immutable, so we must capture the new state.
            model, yMu_inf, _EFE = train_step(
                model, 
                inputs, 
                targets_flat, 
                adapt_synapses=True
            )

            # NOTE: If you get errors about "Tracer" or "Abstract values",
            # ensure your NGCTransformer is registered as a JAX PyTree 
            # (ngclearn models usually are by default).

            train_EFE += _EFE
            total_batches += 1

            if batch_idx % 10 == 0:
                # We need to reshape the output from the JITed function
                y_pred = yMu_inf.reshape(-1, vocab_size)
                y_true = jax.nn.one_hot(
                    targets.flatten(), vocab_size
                )

                # Move calculation to CPU or keep on device? 
                # Keeping it simple here.
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

        # Eval model is usually fast enough without JIT if data is small, 
        # but you can JIT that too if needed.
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