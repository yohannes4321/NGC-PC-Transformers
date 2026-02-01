import sys
import os
import time
from math import inf
import jax.numpy as jnp
from jax import random

# Project Imports
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class PruningError(Exception):
    """Exception raised when a trial is performing significantly worse than best."""
    pass

def _build_cfg(params_override=None):
    cfg = type("Cfg", (), {})()
    for key, value in config.__dict__.items():
        if not key.startswith("_"):
            setattr(cfg, key, value)
    if isinstance(params_override, dict):
        for k, v in params_override.items():
            setattr(cfg, k, v)
    if not hasattr(cfg, "n_embed"):
        cfg.n_embed = cfg.n_heads * cfg.embed_mult
    return cfg

def run_training(params_override=None, save_model=False, max_train_batches=None, pruning_threshold=None):
    cfg = _build_cfg(params_override)
    max_train_batches = 10 if max_train_batches is None else int(max_train_batches)

    dkey = random.PRNGKey(1234)
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    model = NGCTransformer(
        dkey,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        n_embed=cfg.n_embed,
        vocab_size=config.vocab_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        T=cfg.n_iter,
        dt=1.0,
        tau_m=cfg.tau_m,
        act_fx=cfg.act_fx,
        eta=cfg.eta,
        dropout_rate=getattr(cfg, "dropout_rate", 0.0),
        exp_dir="exp",
        pos_learnable=cfg.pos_learnable,
        optim_type=config.optim_type,   
        wub=cfg.wub,
        wlb=cfg.wlb,
        model_name="ngc_transformer",
    )

    total_batches = 0
    sum_efe = 0.0
    start_time = time.time()

    for epoch in range(cfg.epoch):
        for batch_idx, batch in enumerate(train_loader):
            if total_batches >= max_train_batches:
                break

            inputs, targets = batch[0][1], batch[1][1]
            targets_flat = jnp.eye(config.vocab_size)[targets].reshape(-1, config.vocab_size)

            # FORWARD
            _, _, efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            current_efe = float(abs(efe))

            # --- DYNAMIC PRUNING ---
            # We start checking after 10 batches to let the model settle
            if pruning_threshold is not None and total_batches > 10:
                if current_efe > (pruning_threshold * 3.0):
                    print(f"[PRUNE] Current EFE {current_efe:.2f} > 3x Best ({pruning_threshold:.2f})")
                    raise PruningError(f"Trial EFE {current_efe} exceeded threshold.")

            # --- SAFETY PRUNING ---
            if not jnp.isfinite(efe) or current_efe > 1e7:
                print("[PRUNE-HARD] Instability detected")
                raise PruningError("Numerical instability.")

            sum_efe += float(efe)
            total_batches += 1

            if batch_idx % 10 == 0:
                y_pred = _.reshape(-1, config.vocab_size)
                y_true = jnp.eye(config.vocab_size)[targets.flatten()]
                batch_ce = measure_CatNLL(y_pred, y_true).mean()
                print(f"B:{batch_idx} | EFE:{efe:.2f} | CE:{batch_ce:.4f}")

        avg_train_EFE = sum_efe / max(total_batches, 1)
        dev_ce, dev_ppl = eval_model(model, valid_loader, config.vocab_size)

    total_time = time.time() - start_time
    print("time",total_time)
    return float(abs(avg_train_EFE)), float(dev_ce), float(dev_ppl)