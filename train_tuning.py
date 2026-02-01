import os
import jax
import jax.numpy as jnp
from jax import random, device_put
import time

# Set memory allocation behavior BEFORE imports
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Project Imports
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

class PruningError(Exception):
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
    max_train_batches = 30 if max_train_batches is None else int(max_train_batches)

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

            # --- OPTIMIZATION 1: Use device_put to keep data on GPU ---
            inputs = device_put(jnp.array(batch[0][1]))
            targets = device_put(jnp.array(batch[1][1]))
            
            # --- OPTIMIZATION 2: Efficient Target Handling ---
            # If your model.process requires one-hot, only expand it right before the call
            # and reshape it to avoid giant intermediate matrices.
            targets_one_hot = jax.nn.one_hot(targets, config.vocab_size).reshape(-1, config.vocab_size)

            # FORWARD
            # Using model.process with device-resident data
            y_mu, _, efe = model.process(obs=inputs, lab=targets_one_hot, adapt_synapses=True)
            current_efe = float(abs(efe))

            # --- DYNAMIC PRUNING ---
            if pruning_threshold is not None and total_batches > 10:
                if current_efe > (pruning_threshold * 3.0):
                    raise PruningError(f"Trial EFE {current_efe} exceeded threshold.")

            # --- SAFETY PRUNING ---
            if not jnp.isfinite(efe) or current_efe > 1e7:
                raise PruningError("Numerical instability.")

            sum_efe += float(efe)
            total_batches += 1

            if batch_idx % 10 == 0:
                # OPTIMIZATION 3: Memory-efficient Categorical NLL
                y_pred = y_mu.reshape(-1, config.vocab_size)
                # Instead of jnp.eye, use targets directly for CE
                # We reuse the one-hot version only for the calculation to save VRAM
                batch_ce = measure_CatNLL(y_pred, targets_one_hot).mean()
                print(f"B:{batch_idx} | EFE:{efe:.2f} | CE:{batch_ce:.4f}")

    avg_train_EFE = sum_efe / max(total_batches, 1)
    dev_ce, dev_ppl = eval_model(model, valid_loader, config.vocab_size)

    print(f"Time: {time.time() - start_time:.2f}s")
    return float(abs(avg_train_EFE)), float(dev_ce), float(dev_ppl)