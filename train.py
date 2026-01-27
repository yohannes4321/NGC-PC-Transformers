import sys
import os
# Disable pre-allocation so JAX only takes what it needs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
import jax.nn as jnn
from jax import numpy as jnp, random
from math import inf
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
    cfg = _build_cfg(params_override)
    max_train_batches = 40 if max_train_batches is None else int(max_train_batches)

    dkey = random.PRNGKey(1234)
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

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
    pruned_trial = False  # Flag to indicate pruning

    total_start_time = time.time()
    for i in range(cfg.num_iter):
        for batch_idx, batch in enumerate(train_loader):
            if total_batches >= max_train_batches:
                break

            inputs = batch[0][1]
            targets = batch[1][1]

            # Convert targets to one-hot and flatten
            targets_onehot = jnp.eye(config.vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, config.vocab_size)

            yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            train_EFE += _EFE
            total_batches += 1

            # Periodic logging
            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, config.vocab_size)
                y_true = jnp.eye(config.vocab_size)[targets.flatten()]

                batch_nll = measure_CatNLL(y_pred, y_true)
                batch_ce_loss = batch_nll.mean()
                batch_ppl = jnp.exp(batch_ce_loss)

                print(f"  Batch {batch_idx}: EFE = {_EFE:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}")

                # Prune trial if EFE > 1000
                if abs(_EFE) > 1000:
                    print("  Pruned trial: EFE > 1000 → returning high loss for HPO")
                    pruned_trial = True
                    return 1e9, 1e9, 1e9  # Nevergrad will see this as a bad trial

        # Compute average EFE so far
        avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0.0

        # Evaluate on validation set
        dev_ce, dev_ppl = eval_model(model, valid_loader, config.vocab_size)
        print(f"Iter {i} Summary: CE = {dev_ce:.4f}, PPL = {dev_ppl:.4f}, Avg EFE = {avg_train_EFE:.4f}")

        if pruned_trial:
            break  # stop current trial but phase continues with other candidates

    total_time = time.time() - total_start_time
    print(f"\nTuning finished. Total training time: {total_time:.0f} seconds")
    return avg_train_EFE, dev_ce, dev_ppl
