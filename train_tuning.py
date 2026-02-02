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
    cfg_params = {
        k: getattr(cfg, k)
        for k in dir(cfg)
        if not k.startswith("_") and not callable(getattr(cfg, k))
    }

    dkey = random.PRNGKey(1234)
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    # We still load, but we won't use valid_loader here
    train_loader, _, _ = data_loader.load_and_prepare_data()

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
    sum_abs_efe = 0.0
    sum_ce = 0.0 # Track CE during training
    start_time = time.time()

    print(f"Params: {cfg_params}")

    for epoch in range(cfg.epoch):
        for batch_idx, batch in enumerate(train_loader):
            if total_batches >= max_train_batches:
                break

            inputs = device_put(jnp.array(batch[0][1]))
            targets = device_put(jnp.array(batch[1][1]))
            targets_one_hot = jax.nn.one_hot(targets, config.vocab_size).reshape(-1, config.vocab_size)

            # FORWARD
            y_mu, _, efe = model.process(obs=inputs, lab=targets_one_hot, adapt_synapses=True)
            
            # Calculate CE for this batch
            y_pred = y_mu.reshape(-1, config.vocab_size)
            batch_ce = measure_CatNLL(y_pred, targets_one_hot).mean()
            batch_ppl = jnp.exp(batch_ce)
            # Update accumulators
            current_efe = float(efe)
            current_abs_efe = float(abs(efe))
            sum_efe += float(efe)
            sum_abs_efe += current_abs_efe
            sum_ce += float(batch_ce)
            total_batches += 1

            # DYNAMIC PRUNING
            if pruning_threshold is not None and total_batches > 10:
                if current_abs_efe > (pruning_threshold * 3.0):
                    raise PruningError(
                        f"Trial |EFE| {current_abs_efe} exceeded threshold {pruning_threshold}."
                    )

            if not jnp.isfinite(efe) or current_abs_efe > 1e7:
                raise PruningError("Numerical instability.")

            if batch_idx % 10 == 0:
                print(f"B:{batch_idx} | EFE:{efe:.2f} | CE:{batch_ce:.4f} | PPL:{batch_ppl:.2f}")

    # Calculate final stats from the training batches
    avg_train_efe = sum_efe / max(total_batches, 1)
    avg_train_abs_efe = sum_abs_efe / max(total_batches, 1)
    avg_train_ce = sum_ce / max(total_batches, 1)
    # Perplexity approximation: exp(CrossEntropy)
    approx_ppl = jnp.exp(avg_train_ce)

    print(f"Trial Finished. Time: {time.time() - start_time:.2f}s")
    return float(avg_train_abs_efe), float(avg_train_ce), float(approx_ppl)