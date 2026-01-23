import sys
import os
# Disable pre-allocation so JAX only takes what it needs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.nn as jnn
from jax import numpy as jnp, random
from math import inf
from model import NGCTransformer
import time as _time
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

def _build_cfg(params_override=None):
    cfg = type("Cfg", (), {})()
    for key, value in config.__dict__.items():
        if not key.startswith("_"):
            setattr(cfg, key, value)
    if isinstance(params_override, dict):
        for k, v in params_override.items():
            setattr(cfg, k, v)
    return cfg

def run_training(params_override=None, save_model=False, max_train_batches=None):
    cfg = _build_cfg(params_override)
    
   
    
    print("\n" + "-"*60)
    print(" INITIALIZING TRAINING TRIAL")
    print("-"*60)
    print(f"   [ARCH] n_heads:      {cfg.n_heads}")
    print(f"   [ARCH] embed_mult:   {cfg.embed_mult} (Total n_embed: {cfg.n_embed})")
    print(f"   [DATA] batch_size:   {cfg.batch_size}")
    print(f"   [OPTS] eta (LR):     {cfg.eta:.2e}")
    print(f"   [OPTS] act_fx:       {cfg.act_fx}")
    print(f"   [PHYS] n_iter (T):   {cfg.n_iter}")
    print("-"*60)

    dkey = random.PRNGKey(1234)
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    model = NGCTransformer(
        dkey, batch_size=cfg.batch_size, seq_len=cfg.seq_len, n_embed=cfg.n_embed,
        vocab_size=cfg.vocab_size, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        T=cfg.n_iter, dt=1.0, tau_m=cfg.tau_m, act_fx=cfg.act_fx, eta=cfg.eta,
        dropout_rate=cfg.dropout_rate if hasattr(cfg, 'dropout_rate') else 0.0, 
        exp_dir="exp", pos_learnable=cfg.pos_learnable,
        optim_type=cfg.optim_type, wub=cfg.wub, wlb=cfg.wlb, model_name="ngc_transformer"
    )

    total_efe, total_ce, total_batches = 0.0, 0.0, 0
    # best_train_efe_abs, best_train_ce = inf, inf
    # last_efe_window = []
    # plateau_triggered = False

    for i in range(cfg.num_iter):
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch[0][1], batch[1][1]
            
            # MEMORY FIX: Use one_hot directly instead of jnp.eye
            targets_onehot = jnn.one_hot(targets.flatten(), cfg.vocab_size)
            targets_flat = targets_onehot.reshape(-1, cfg.vocab_size)

            # Process through the NGC Transformer
            yMu_inf, _, efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            # --- CRITICAL NAN/INF PENALTY ---
            # If this triggers, we return a dictionary immediately so Nevergrad knows this trial failed.
            if jnp.isnan(efe) or jnp.isinf(efe):
                print(f"!!! NAN/INF DETECTED at Step {batch_idx} >> Applying Penalty and Terminating Trial.")
                return 1e10,1e10,1e10

            efe_val = float(efe)
            total_efe += efe_val
            total_batches =batch_idx
            
            y_pred = yMu_inf.reshape(-1, cfg.vocab_size)
            # Use targets_onehot we already created instead of allocating jnp.eye again
            batch_ce = float(measure_CatNLL(y_pred, targets_onehot).mean())
            total_ce += batch_ce

            # if abs(efe_val) < best_train_efe_abs:
            #     best_train_efe_abs = abs(efe_val)
            
            # best_train_ce = min(best_train_ce, batch_ce)

            # Plateau logic
            

    if total_batches == 0:
        raise RuntimeError("No batches were processed! Cannot compute average EFE.")
    avg_efe = total_efe / total_batches
    

    dev_ce, dev_ppl = eval_model(model, valid_loader, cfg.vocab_size)
    if batch_idx %10 ==0:
        print(f"Trial complete -> Real avg EFE: {avg_efe:.4f}, CE: {dev_ce:.4f} ,PPL: {dev_ppl:.4f}")
    return avg_efe,dev_ce,dev_ppl
    