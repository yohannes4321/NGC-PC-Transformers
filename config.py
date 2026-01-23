class Config:
    SEED = 42

    # Default selector for Nevergrad HPO main
    case_nevergrad = 1

    # Two-phase Nevergrad budgets
    p1_budget = 6
    p2_budget = 6
    # Plateau early-stop for Nevergrad phases
    # Phase 1: stop if EFE does not change by > 1 over the window
    phase1_plateau_window = 3
    phase1_plateau_min_delta = 1.0
    phase1_plateau_warmup = 2
    # Phase 2: stop if CE does not change by > 0.5 over the window
    phase2_plateau_window = 3
    phase2_plateau_min_delta = 0.5
    phase2_plateau_warmup = 2

    seq_len = 8
    embed_mult = 16
    n_embed = 32
    
    batch_size = 16
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 2
    n_layers = 4
    dropout_rate = 0.1
    eta = 2.24e-06
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "sgd"
    num_iter = 2
    n_iter= 3
    # Early stop if batches plateau (within a trial)
    # Stop when both EFE and CE vary by < early_stop_min_delta across the window
    # Increase to 1.0 to stop earlier when changes are small
    early_stop_window = 4
    early_stop_min_delta = 1.0
    early_stop_warmup_batches = 20
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.025
    wlb = -0.025
    tau_m = 10.
    act_fx = "identity"
    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None