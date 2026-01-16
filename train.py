from jax import numpy as jnp, random
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

def _build_cfg(params_override=None):
    """Create a config-like object, applying overrides if provided."""
    cfg = type("Cfg", (), {})()
    for key, value in config.__dict__.items():
        if not key.startswith("_"):
            setattr(cfg, key, value)
    if isinstance(params_override, dict):
        for k, v in params_override.items():
            setattr(cfg, k, v)
        if hasattr(cfg, "embed_mult") and hasattr(cfg, "n_heads") and not getattr(cfg, "n_embed", None):
            try:
                cfg.n_embed = int(cfg.n_heads) * int(cfg.embed_mult)
            except: pass
    return cfg

def run_training(params_override=None, save_model=False, max_train_batches=None):
    cfg = _build_cfg(params_override)

    # --- SAFETY FIX: Define logging variables before they are used ---
    log_interval = max(1, int(getattr(cfg, "log_batch_interval", 10)))
    live_logging = bool(getattr(cfg, "live_logging", False))
    # ----------------------------------------------------------------

    dkey = random.PRNGKey(1234)
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    model = NGCTransformer(
        dkey,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        n_embed=cfg.n_embed,
        vocab_size=cfg.vocab_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        T=cfg.n_iter,
        dt=1.0,
        tau_m=cfg.tau_m,
        act_fx=cfg.act_fx,
        eta=cfg.eta,
        dropout_rate=cfg.dropout_rate,
        exp_dir="exp",
        loadDir=None,
        pos_learnable=cfg.pos_learnable,
        optim_type=cfg.optim_type,
        wub=cfg.wub,
        wlb=cfg.wlb,
        model_name="ngc_transformer",
    )

    total_efe = 0.0
    total_ce = 0.0
    total_batches = 0
    train_batches_seen = 0

    for i in range(cfg.num_iter):
        print(f"\n iter {i}:")
        for batch_idx, batch in enumerate(train_loader):
            # Check if we should stop early based on HPO budget
            if max_train_batches is not None and train_batches_seen >= max_train_batches:
                break
            
            inputs = batch[0][1]
            targets = batch[1][1]
            targets_onehot = jnp.eye(cfg.vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, cfg.vocab_size)

            yMu_inf, _, efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            # Metrics Calculation
            total_efe += float(efe)
            total_batches += 1
            y_pred = yMu_inf.reshape(-1, cfg.vocab_size)
            y_true = jnp.eye(cfg.vocab_size)[targets.flatten()]
            batch_ce = float(measure_CatNLL(y_pred, y_true).mean())
            total_ce += batch_ce
            
            # Use defined variables safely
            should_log = live_logging or (batch_idx % log_interval == 0)
            if should_log:
                print(f"  Batch {batch_idx}: EFE = {float(efe):.4f}, CE = {batch_ce:.4f}")
            
            train_batches_seen += 1

    avg_train_efe = (total_efe / total_batches) if total_batches > 0 else 0.0
    avg_train_ce = (total_ce / total_batches) if total_batches > 0 else 0.0

    dev_ce, dev_ppl = eval_model(model, valid_loader, cfg.vocab_size)

    return {
        "val_ce": float(dev_ce),
        "val_ppl": float(dev_ppl),
        "avg_train_efe": avg_train_efe,
        "avg_train_ce": avg_train_ce,
    }