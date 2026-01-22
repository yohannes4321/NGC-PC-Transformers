import uuid
import gc
import nevergrad as ng
import numpy as np
from jax import random, numpy as jnp
import jax

from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

# -----------------------
# Cleanup function
# -----------------------
def _cleanup_run():
    """Clear JAX caches and trigger GC."""
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()

# -----------------------
# Phase 1 Search Space
# -----------------------
def phase1_space():
    """Discrete Architecture Space for EFE Optimization."""
    return ng.p.Dict(
        n_heads    = ng.p.Choice([2, 4, 8]),
        embed_mult = ng.p.Choice([8, 16, 24]),
        batch_size = ng.p.Choice([16, 32]),
        seq_len    = ng.p.Choice([8, 16]),
        eta        = ng.p.Log(lower=1e-7, upper=5e-5),
        tau_m      = ng.p.Scalar(lower=5, upper=15).set_integer_casting(),
        n_iter     = ng.p.Scalar(lower=1, upper=5).set_integer_casting(),
        wub        = ng.p.Scalar(lower=0.01, upper=0.04),
        wlb        = ng.p.Scalar(lower=-0.04, upper=-0.01),
        optim_type = ng.p.Choice(["adam", "sgd"]),
        act_fx     = ng.p.Choice(["identity", "relu"]),
    )

# -----------------------
# Phase 2 Search Space
# -----------------------
def phase2_space(best_p1):
    """Refined continuous space around best Phase 1 parameters for CE optimization."""
    eta_ref = float(best_p1["eta"])
    return ng.p.Dict(
        n_heads      = ng.p.Choice([best_p1["n_heads"]]),  # keep best
        embed_mult   = ng.p.Choice([best_p1["embed_mult"]]),
        batch_size   = ng.p.Choice([best_p1["batch_size"]]),
        seq_len      = ng.p.Choice([best_p1["seq_len"]]),
        eta          = ng.p.Log(lower=eta_ref * 0.1, upper=min(eta_ref*10, 1e-2)),
        tau_m        = ng.p.Scalar(lower=max(1, best_p1["tau_m"]-3),
                                   upper=best_p1["tau_m"]+3).set_integer_casting(),
        n_iter       = ng.p.Scalar(lower=max(1, best_p1["n_iter"]-1),
                                   upper=best_p1["n_iter"]+1).set_integer_casting(),
        wub          = ng.p.Scalar(lower=0.0001, upper=0.05),
        wlb          = ng.p.Scalar(lower=-0.05, upper=-0.0001),
        dropout_rate = ng.p.Scalar(lower=0.0, upper=0.5),
        optim_type   = ng.p.Choice([best_p1["optim_type"]]),
        act_fx       = ng.p.Choice([best_p1["act_fx"]]),
    )

# -----------------------
# Training & Evaluation
# -----------------------
def train_evaluate_model(params, objective="efe"):
    trial_id = uuid.uuid4().hex[:4]
    _cleanup_run()

    try:
        # --- Extract parameters from Nevergrad ---
        seq_len      = int(params["seq_len"])
        batch_size   = int(params["batch_size"])
        n_heads      = int(params["n_heads"])
        embed_mult   = int(params["embed_mult"])
        n_iter       = int(params["n_iter"])
        eta          = float(params["eta"])
        tau_m        = float(params["tau_m"])
        act_fx       = params["act_fx"]
        optim_type   = params["optim_type"]
        wub          = float(params["wub"])
        wlb          = float(params["wlb"])
        dropout_rate = float(params.get("dropout_rate", 0.0))
        vocab_size   = config.vocab_size
        pos_learnable= config.pos_learnable

        # Random key
        dkey = random.PRNGKey(1234)

        # Data
        data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
        train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

        # Model
        model = NGCTransformer(
            dkey,
            batch_size=batch_size,
            seq_len=seq_len,
            n_embed=embed_mult,
            vocab_size=vocab_size,
            n_layers=config.n_layers,
            n_heads=n_heads,
            T=n_iter,
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
            model_name="ngc_transformer"
        )

        # ---- TRAINING LOOP ----
        total_batches = 0
        train_EFE = 0.0

        for iter_idx in range(n_iter):
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch[0][1]
                targets = batch[1][1]

                # one-hot targets
                targets_onehot = jax.nn.one_hot(targets, vocab_size)
                targets_flat = targets_onehot.reshape(-1, vocab_size)

                # forward
                yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
                train_EFE += _EFE
                total_batches += 1

                if batch_idx % 20 == 0:
                    y_pred = yMu_inf.reshape(-1, vocab_size)
                    y_true = jax.nn.one_hot(targets.flatten(), vocab_size)
                    batch_ce = measure_CatNLL(y_pred, y_true).mean()
                    batch_ppl = jnp.exp(batch_ce)
                    print(f"[Iter {iter_idx} | Batch {batch_idx}] EFE={_EFE:.4f}, CE={batch_ce:.4f}, PPL={batch_ppl:.4f}")

        avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0.0
        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)

        # Determine loss for Nevergrad
        if objective == "efe":
            loss = float(abs(avg_train_EFE))
        else:
            loss = float(dev_ce)

        # Save final model
        model.save_to_disk(params_only=False)

        print(f"[Trial {trial_id}] Finished {objective.upper()}: EFE={avg_train_EFE:.4f}, CE={dev_ce:.4f}, PPL={dev_ppl:.4f}, Loss={loss:.4f}")
        return np.array([[loss]])

    except Exception as e:
        print(f"[Trial {trial_id}] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.array([[1e20]])
    finally:
        _cleanup_run()

# -----------------------
# Nevergrad Two-Phase Tuning
# -----------------------
def two_phase_tuning():
    # --- Phase 1: EFE Optimization ---
    print("=== Phase 1: EFE Optimization ===")
    p1 = phase1_space()
    optimizer1 = ng.optimizers.NGOpt(parametrization=p1, budget=20)  # budget = number of trials
    recommendation1 = optimizer1.minimize(lambda x: train_evaluate_model(x, objective="efe"))
    best_p1 = recommendation1.value
    print("Best Phase 1 Params:", best_p1)

    # --- Phase 2: CE Optimization around best Phase 1 params ---
    print("=== Phase 2: CE Optimization ===")
    p2 = phase2_space(best_p1)
    optimizer2 = ng.optimizers.NGOpt(parametrization=p2, budget=20)
    recommendation2 = optimizer2.minimize(lambda x: train_evaluate_model(x, objective="ce"))
    best_p2 = recommendation2.value
    print("Best Phase 2 Params:", best_p2)

    return best_p1, best_p2

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    best_phase1, best_phase2 = two_phase_tuning()
    print("\n=== Final Results ===")
    print("Best Phase 1 (EFE):", best_phase1)
    print("Best Phase 2 (CE):", best_phase2)
