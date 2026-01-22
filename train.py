import sys
import os

# Disable pre-allocation so JAX only takes what it needs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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
    
    # Force frequent logging for HPO visibility
    log_interval = 10 
    plateau_window = 3
    plateau_tol = 1.0  # stop if EFE stays within ±1 for 3 consecutive batches
    
    # ============================================================
    #  FULL DESCRIPTIVE EXECUTION CARD (For Mentor Visibility)
    # ============================================================
    print("\n" + "-"*60)
    print(" INITIALIZING TRAINING TRIAL")
    print("-"*60)
    # Printing every parameter explicitly as requested
    print(f"   [ARCH] n_heads:      {cfg.n_heads}")
    print(f"   [ARCH] embed_mult:   {cfg.embed_mult} (Total n_embed: {cfg.n_embed})")
    print(f"   [DATA] batch_size:   {cfg.batch_size}")
    print(f"   [DATA] seq_len:      {cfg.seq_len}")
    print(f"   [OPTS] eta (LR):     {cfg.eta:.2e}")
    print(f"   [OPTS] optim_type:   {cfg.optim_type}")
    print(f"   [OPTS] act_fx:       {cfg.act_fx}")
    print(f"   [PHYS] tau_m:        {cfg.tau_m}")
    print(f"   [PHYS] n_iter (T):   {cfg.n_iter}")
    print(f"   [REGS] wub / wlb:    {cfg.wub} / {cfg.wlb}")
    if hasattr(cfg, 'dropout_rate'):
        print(f"   [REGS] dropout:      {cfg.dropout_rate}")
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
    best_train_efe_abs, best_train_efe_signed, best_train_ce = inf, None, inf
    last_efe_window = []
    plateau_triggered = False

    # Start training iterations
    for i in range(cfg.num_iter):
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch[0][1], batch[1][1]
            targets_onehot = jnp.eye(cfg.vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, cfg.vocab_size)

            # Process through the NGC Transformer
            yMu_inf, _, efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            efe_val = float(efe)
            total_efe += efe_val
            total_batches += 1
            y_pred = yMu_inf.reshape(-1, cfg.vocab_size)
            y_true = jnp.eye(cfg.vocab_size)[targets.flatten()]
            batch_ce = float(measure_CatNLL(y_pred, y_true).mean())
            total_ce += batch_ce

            if abs(efe_val) < best_train_efe_abs:
                best_train_efe_abs = abs(efe_val)
                best_train_efe_signed = efe_val
            best_train_ce = min(best_train_ce, batch_ce)
            last_efe_window.append(efe_val)
            if len(last_efe_window) > plateau_window:
                last_efe_window.pop(0)
            if len(last_efe_window) == plateau_window:
                window_span = max(last_efe_window) - min(last_efe_window)
                if window_span <= plateau_tol:
                    plateau_triggered = True
                    print(
                        f"   Plateau detected over {plateau_window} steps (Δ<= {plateau_tol}); early stopping.",
                        flush=True,
                    )
                    break
            
            # --- LIVE HEARTBEAT ---
            # Using flush=True to make sure it prints in real-time in the terminal
            if batch_idx % log_interval == 0:
                print(f"   Step {batch_idx:03d} >> EFE: {float(efe):.4f} | CE: {batch_ce:.4f}", flush=True)

            if max_train_batches and total_batches >= max_train_batches:
                break

        if plateau_triggered:
            break

    avg_efe = total_efe / total_batches if total_batches > 0 else 0
    avg_ce = total_ce / total_batches if total_batches > 0 else 0
    best_efe_abs_out = best_train_efe_abs if best_train_efe_abs != inf else abs(avg_efe)
    best_efe_signed_out = best_train_efe_signed if best_train_efe_signed is not None else avg_efe
    best_ce_out = best_train_ce if best_train_ce != inf else avg_ce
    dev_ce, dev_ppl = eval_model(model, valid_loader, cfg.vocab_size)

    # FINAL SUMMARY FOR THIS TRIAL
    print(f"\n✅ TRIAL COMPLETE | Avg EFE: {avg_efe:.2f} | Final Val CE: {dev_ce:.4f}")
    print(
        f"   Best Train EFE: {best_efe_signed_out:.4f} (abs {best_efe_abs_out:.4f}) | "
        f"Best Train CE: {best_ce_out:.4f} | Val PPL: {dev_ppl:.4f}"
    )
    if plateau_triggered:
        print("   Early stop reason: plateau")
    print("")

    return {
        "val_ce": float(dev_ce),
        "val_ppl": float(dev_ppl),
        "avg_train_efe": avg_efe,
        "best_train_efe": float(best_efe_signed_out),
        "best_train_efe_abs": float(best_efe_abs_out),
        "best_train_ce": float(best_ce_out),
        "best_val_ce": float(dev_ce),
        "best_val_ppl": float(dev_ppl),
        "batches_ran": total_batches,
        "plateau_triggered": plateau_triggered
    }