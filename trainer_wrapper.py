def run_training(params_override=None, save_model=False, max_train_batches=None):
    cfg = _build_cfg(params_override)
    
    # Force frequent logging for HPO visibility
    log_interval = 10 
    
    # MENTOR-FRIENDLY CONFIGURATION DUMP
    print("\n" + "-"*50)
    print("⚡ MODEL INITIALIZATION (FULL PARAMETER TRACE):")
    print(f"   - n_heads:      {cfg.n_heads}")
    print(f"   - embed_mult:   {cfg.embed_mult} (Total n_embed: {cfg.n_embed})")
    print(f"   - batch_size:   {cfg.batch_size}")
    print(f"   - seq_len:      {cfg.seq_len}")
    print(f"   - eta (LR):     {cfg.eta:.2e}")
    print(f"   - tau_m:        {cfg.tau_m}")
    print(f"   - n_iter:       {cfg.n_iter}")
    print(f"   - wub / wlb:    {cfg.wub} / {cfg.wlb}")
    print(f"   - optim / act:  {cfg.optim_type} / {cfg.act_fx}")
    print("-"*50)

    dkey = random.PRNGKey(1234)
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    model = NGCTransformer(
        dkey, batch_size=cfg.batch_size, seq_len=cfg.seq_len, n_embed=cfg.n_embed,
        vocab_size=cfg.vocab_size, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        T=cfg.n_iter, dt=1.0, tau_m=cfg.tau_m, act_fx=cfg.act_fx, eta=cfg.eta,
        dropout_rate=cfg.dropout_rate, exp_dir="exp", pos_learnable=cfg.pos_learnable,
        optim_type=cfg.optim_type, wub=cfg.wub, wlb=cfg.wlb, model_name="ngc_transformer"
    )

    total_efe, total_ce, total_batches = 0.0, 0.0, 0

    for i in range(cfg.num_iter):
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch[0][1], batch[1][1]
            targets_onehot = jnp.eye(cfg.vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, cfg.vocab_size)

            yMu_inf, _, efe = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            # Metric calculation
            total_efe += float(efe)
            total_batches += 1
            y_pred = yMu_inf.reshape(-1, cfg.vocab_size)
            y_true = jnp.eye(cfg.vocab_size)[targets.flatten()]
            batch_ce = float(measure_CatNLL(y_pred, y_true).mean())
            total_ce += batch_ce
            
            # --- LIVE HEARTBEAT ---
            if batch_idx % log_interval == 0:
                print(f"   [Iter {i} | Batch {batch_idx:03d}] Current EFE: {float(efe):.2f} | Current CE: {batch_ce:.4f}")
                sys.stdout.flush() 

            if max_train_batches and total_batches >= max_train_batches:
                break

    avg_efe = total_efe / total_batches if total_batches > 0 else 0
    dev_ce, dev_ppl = eval_model(model, valid_loader, cfg.vocab_size)

    return {
        "val_ce": float(dev_ce),
        "val_ppl": float(dev_ppl),
        "avg_train_efe": avg_efe
    }