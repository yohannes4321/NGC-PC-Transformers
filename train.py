import sys
import os
import time
from math import inf

# JAX Configuration
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax.numpy as jnp
from jax import random

from model import NGCTransformer
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
    """
    Returns: (efe, ce, ppl) as floats.
    """
    cfg = _build_cfg(params_override)
    max_train_batches = 40 if max_train_batches is None else int(max_train_batches)

    dkey = random.PRNGKey(1234)
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    # Model Init
    model = NGCTransformer(
        dkey, batch_size=cfg.batch_size, seq_len=cfg.seq_len, n_embed=cfg.n_embed,
        vocab_size=config.vocab_size, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        T=cfg.n_iter, dt=1.0, tau_m=cfg.tau_m, act_fx=cfg.act_fx, eta=cfg.eta,
        dropout_rate=cfg.dropout_rate if hasattr(cfg, 'dropout_rate') else 0.0, 
        exp_dir="exp", pos_learnable=cfg.pos_learnable,
        optim_type=config.optim_type, wub=cfg.wub, wlb=cfg.wlb, model_name="ngc_transformer"
    )

    total_batches = 0
    train_EFE = 0.0
    
    total_start_time = time.time()
    
    for i in range(cfg.num_iter):
        for batch_idx, batch in enumerate(train_loader):
            if total_batches >= max_train_batches:
                break

            inputs = batch[0][1]
            targets = batch[1][1]
            targets_onehot = jnp.eye(config.vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, config.vocab_size)

            # Process
            yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            # --- CRASH PROTECTION / PRUNING ---
            # If EFE explodes or is NaN, return high loss immediately
            if jnp.isnan(_EFE) or jnp.abs(_EFE) > 2000:
                print(f"  [Pruning] Trial Unstable. EFE={_EFE}")
                return 1e9, 1e9, 1e9

            train_EFE += _EFE
            total_batches += 1

            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, config.vocab_size)
                y_true = jnp.eye(config.vocab_size)[targets.flatten()]
                batch_ce = measure_CatNLL(y_pred, y_true).mean()
                print(f"  Batch {batch_idx}: EFE={_EFE:.4f}, CE={batch_ce:.4f}")

        # End of Epoch Evaluation
        avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0.0
        
        # Validation
        dev_ce, dev_ppl = eval_model(model, valid_loader, config.vocab_size)
        print(f"Iter {i} Summary: CE={dev_ce:.4f}, PPL={dev_ppl:.4f}, Avg EFE={avg_train_EFE:.4f}")

    total_time = time.time() - total_start_time
    print(f"Training Time: {total_time:.1f}s")
    
    # Return primitive floats (not JAX arrays) for serialization
    return float(avg_train_EFE), float(dev_ce), float(dev_ppl)