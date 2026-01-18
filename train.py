import sys
from jax import numpy as jnp, random
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
    
    # ============================================================
    # 📋 FULL DESCRIPTIVE EXECUTION CARD (For Mentor Visibility)
    # ============================================================
    print("\n" + "-"*60)
    print("🚀 INITIALIZING TRAINING TRIAL")
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

    # Start training iterations
    for i in range(cfg.num_iter):
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch[0][1], batch[1][1]
            targets_onehot = jnp.eye(cfg.vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, cfg.vocab_size)

            # Process through the NGC Transformer
            yMu_inf, _, efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            # Metric calculation
            total_efe += float(efe)
            total_batches += 1
            y_pred = yMu_inf.reshape(-1, cfg.vocab_size)
            y_true = jnp.eye(cfg.vocab_size)[targets.flatten()]
            batch_ce = float(measure_CatNLL(y_pred, y_true).mean())
            total_ce += batch_ce
            
            # --- LIVE HEARTBEAT ---
            # Using flush=True to make sure it prints in real-time in the terminal
            if batch_idx % log_interval == 0:
                print(f"   Step {batch_idx:03d} >> EFE: {float(efe):.4f} | CE: {batch_ce:.4f}", flush=True)

            if max_train_batches and total_batches >= max_train_batches:
                break

    avg_efe = total_efe / total_batches if total_batches > 0 else 0
    dev_ce, dev_ppl = eval_model(model, valid_loader, cfg.vocab_size)

    # FINAL SUMMARY FOR THIS TRIAL
    print(f"\n✅ TRIAL COMPLETE | Avg EFE: {avg_efe:.2f} | Final Val CE: {dev_ce:.4f}\n")

    return {
        "val_ce": float(dev_ce),
        "val_ppl": float(dev_ppl),
        "avg_train_efe": avg_efe
    }