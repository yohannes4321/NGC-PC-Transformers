import os
import sys
import warnings
import logging
import optuna
import time
import gc

import jax
import jax.numpy as jnp
import jax.random as random
from pathlib import Path

from model import NGCTransformer
from data_preprocess.data_loader import DataLoader
from eval import eval_model
from config import Config as base_config
from ngclearn.utils.metric_utils import measure_CatNLL

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================
# GPU MEMORY SAFETY FLAGS
# =========================
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

EFE_STABILITY_THRESHOLD = 100000

# =========================
# GLOBAL DATA LOADER (IMPORTANT FIX)
# =========================
GLOBAL_LOADER = DataLoader(seq_len=16, batch_size=4)
TRAIN_LOADER, VALID_LOADER, _ = GLOBAL_LOADER.load_and_prepare_data()

eye_vocab_cache = None


# =========================
# SEARCH SPACE (FIXED SHAPES)
# =========================
def define_search_space(trial):

    n_heads = trial.suggest_int("n_heads", 2, 4)

    # FIX: remove dynamic multiplication causing recompilation explosion
    n_embed = trial.suggest_categorical("n_embed", [32, 48, 64, 72])

    return {
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "pos_learnable": trial.suggest_categorical("pos_learnable", [True, False]),

        "eta": trial.suggest_float("eta", 1e-7, 1e-4, log=True),
        "tau_m": trial.suggest_int("tau_m", 10, 30),
        "n_iter": trial.suggest_int("n_iter", 20, 80),

        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.3),

        "wub": trial.suggest_float("wub", 0.005, 0.05),
        "wlb": trial.suggest_float("wlb", -0.05, -0.005),

        "optim_type": trial.suggest_categorical("optim_type", ["adam"]),

        "act_fx": trial.suggest_categorical("act_fx", ["relu", "elu", "gelu", "tanh"]),
        "act_fx_o": trial.suggest_categorical("act_fx_o", ["softmax"]),

        "n_heads": n_heads,
        "n_embed": n_embed,

        "batch_size": trial.suggest_categorical("batch_size", [2, 4, 6]),
        "seq_len": trial.suggest_categorical("seq_len", [8, 16, 24]),

        "embed_mult": 16
    }


# =========================
# MODEL CREATION
# =========================
def create_model_with_all_params(trial_number, params, cfg):
    dkey = random.PRNGKey(trial_number * 1000 + 42)

    model_args = {
        "dkey": dkey,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "n_embed": cfg.n_embed,
        "vocab_size": cfg.vocab_size,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "T": cfg.n_iter,
        "dt": 1.0,
        "tau_m": cfg.tau_m,
        "act_fx": cfg.act_fx,
        "act_fx_o": getattr(cfg, "act_fx_o", cfg.act_fx),
        "eta": cfg.eta,
        "dropout_rate": cfg.dropout_rate,
        "pos_learnable": cfg.pos_learnable,
        "optim_type": cfg.optim_type,
        "wub": cfg.wub,
        "wlb": cfg.wlb,
        "model_name": f"trial_{trial_number}"
    }

    model = NGCTransformer(**model_args)
    return model


# =========================
# CLEAN CONFIG BUILDER
# =========================
def build_cfg(params):
    cfg = type("Config", (), {})()

    for k, v in base_config.__dict__.items():
        if not k.startswith("_"):
            setattr(cfg, k, v)

    for k, v in params.items():
        setattr(cfg, k, v)

    if not hasattr(cfg, "vocab_size"):
        cfg.vocab_size = base_config.vocab_size

    return cfg


# =========================
# PHASE 1: EFE
# =========================
def run_single_trial_efe(trial):
    global eye_vocab_cache

    try:
        params = define_search_space(trial)
        cfg = build_cfg(params)

        print(f"[EFE] Trial {trial.number}")

        model = create_model_with_all_params(trial.number, params, cfg)

        if eye_vocab_cache is None:
            eye_vocab_cache = jnp.eye(cfg.vocab_size)

        total_efe = 0.0
        max_batches = 6

        for i, batch in enumerate(TRAIN_LOADER):
            if i >= max_batches:
                break

            inputs = batch[0][1]
            targets = batch[1][1]

            targets_flat = eye_vocab_cache[targets]

            result = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)

            if isinstance(result, tuple):
                EFE = result[-1]
            else:
                EFE = result

            EFE = abs(float(EFE))

            if jnp.isnan(EFE) or jnp.isinf(EFE) or EFE > EFE_STABILITY_THRESHOLD:
                raise optuna.TrialPruned()

            total_efe += EFE
            avg = total_efe / (i + 1)

            trial.report(avg, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(total_efe / max_batches)

    finally:
        # =========================
        # CRITICAL CLEANUP (FIX)
        # =========================
        gc.collect()
        jax.clear_caches()


# =========================
# PHASE 2: CE
# =========================
def run_phase2_trial(trial, best_params):
    params = best_params.copy()
    params.update({
        "eta": trial.suggest_float("eta", 1e-7, 5e-5, log=True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.2),
        "wub": trial.suggest_float("wub", 0.01, 0.05),
        "wlb": trial.suggest_float("wlb", -0.05, -0.01),
    })

    cfg = build_cfg(params)
    model = create_model_with_all_params(trial.number, params, cfg)

    total_ce = 0.0
    max_batches = 10

    for i, batch in enumerate(TRAIN_LOADER):
        if i >= max_batches:
            break

        inputs = batch[0][1]
        targets = batch[1][1]

        targets_flat = eye_vocab_cache[targets]

        y_pred, _, EFE, *_ = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)

        y_pred = y_pred.reshape(-1, cfg.vocab_size)

        ce = measure_CatNLL(y_pred, targets_flat)

        total_ce += float(ce)

    try:
        val_ce, ppl = eval_model(model, VALID_LOADER, cfg.vocab_size)
    except:
        val_ce = total_ce / max_batches
        ppl = float("inf")

    return float(val_ce)


# =========================
# MAIN
# =========================
def main():

    print("PC TRANSFORMER - SAFE TUNING VERSION")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )

    study.optimize(run_single_trial_efe, n_trials=20)

    print("\nBEST RESULT:", study.best_value)
    print("BEST PARAMS:", study.best_params)


if __name__ == "__main__":
    main()