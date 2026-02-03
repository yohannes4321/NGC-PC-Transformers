import time
import jax
from jax import numpy as jnp, random

from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

def main():
    # Start the global clock
    total_start_time = time.time()

    # ----------------------------
    # Config & Setup
    # ----------------------------
    seq_len = config.seq_len
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    epoch = config.num_iter
    dkey = random.PRNGKey(1234)

    # Calculate theoretical "Eye" size once
    eye_temp = jnp.eye(vocab_size)
    eye_mb = eye_temp.nbytes / (1024**2)
    del eye_temp # Free it immediately so we don't crash yet

    print(f"\n❌ RUNNING INEFFICIENT BASELINE")
    print(f"Vocab size: {vocab_size}")
    print(f"Memory waste: Every time you call jnp.eye, JAX allocates {eye_mb:.2f} MB")
    print("-" * 50)

    # ----------------------------
    # Data & Model Init
    # ----------------------------
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    model = NGCTransformer(
        dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1.0, tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.eta, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer"
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    for i in range(epoch):
        print(f"\nEpoch {i}")
        train_EFE=0
        for batch_idx, batch in enumerate(train_loader):
            # START STEP TIMER
            step_start = time.time()
            
            inputs = batch[0][1]
            targets = batch[1][1]

            # ❌ THE BAD OPERATION
            # We create a massive matrix, then index it
            identity = jnp.eye(vocab_size)
            targets_onehot = identity[targets] 
            targets_flat = targets_onehot.reshape(-1, vocab_size)

            yMu_inf, _, _EFE = model.process(
                obs=inputs,
                lab=targets_flat,
                adapt_synapses=True
            )

            # ⛔ SYNC: Must wait for JAX to finish to get a real time measurement
            jax.block_until_ready(yMu_inf)
            
            # END STEP TIMER
            step_time = time.time() - step_start
            train_EFE += _EFE
            total_batches += 1
            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)
                
                # ❌ SECOND WASTE: Re-allocating identity for evaluation
                eval_eye = jnp.eye(vocab_size)
                y_true = eval_eye[targets.flatten()]

                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce = batch_nll.mean()
                
                # Show memory and time usage
                print(
                    f"Batch {batch_idx:03d} | "
                    f"Time: {step_time:.4f}s | "
                    f"Allocated: {eye_mb:.2f} MB | "
                    f"EFE: {_EFE} "
                    f"CE: {batch_ce:.4f}"
                )
        avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0
        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
        print(f"--- Epoch {i} Summary: Dev CE={dev_ce:.4f} ---")

    # Final Time Calculation
    total_duration = time.time() - total_start_time
    print("\n" + "="*50)
    print("❌ BASELINE FINISHED")
    print(f"Total Program Runtime: {total_duration:.2f} seconds")
    print(f"Memory used per jnp.eye call: {eye_mb:.2f} MB")
    print("="*50)

if __name__ == "__main__":
    main()