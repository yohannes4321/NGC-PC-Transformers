import sys
import os
# Disable pre-allocation so JAX only takes what it needs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
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
    # Cap runtime: default to first 50 batches unless overridden
    max_train_batches = 40 if max_train_batches is None else int(max_train_batches)
    
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
    train_EFE = 0.0
    # best_train_efe_abs, best_train_ce = inf, inf
    # last_efe_window = []
    # plateau_triggered = False
    
    total_start_time = time.time()
    for i in range(cfg.num_iter):
        for batch_idx, batch in enumerate(train_loader):
            if total_batches >= max_train_batches:
                break
            inputs = batch[0][1]
            targets = batch[1][1]
            
            #Convert targets to one-hot and flatten
            targets_onehot = jnp.eye(config.vocab_size)[targets]  # (B, S, V)
            targets_flat = targets_onehot.reshape(-1, config.vocab_size)  # (B*S, V)

            
            yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            _EFE=_EFE / (cfg.seq_len * cfg.batch_size * cfg.n_iter)
            train_EFE += _EFE
            total_batches += 1

            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, config.vocab_size)
                y_true = jnp.eye(config.vocab_size)[targets.flatten()]
                
                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce_loss = batch_nll.mean()  
                batch_ppl = jnp.exp(batch_ce_loss)
                
                print(f"  Batch {batch_idx}: EFE = {_EFE:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}")
        
        avg_train_EFE = train_EFE / total_batches  if total_batches > 0 else 0
        
        dev_ce, dev_ppl = eval_model(model, valid_loader, config.vocab_size)
        print(f"Iter {i} Summary: CE = {dev_ce:.4f}, PPL = {dev_ppl:.4f}, Avg EFE = {avg_train_EFE:.4f}")
        if total_batches >= max_train_batches:
            print(f"Reached max_train_batches={max_train_batches}; stopping early.")
            break
        # if  i == (num_iter-1):
        #   model.save_to_disk(params_only=False) # save final state of model to disk
    total_time = time.time() - total_start_time
    print(f"\nTraining finished.")
    print(f"Total training time: {total_time:.0f} seconds")
    print(f"\nTraining finished.")
    return avg_train_EFE,dev_ce,dev_ppl
    