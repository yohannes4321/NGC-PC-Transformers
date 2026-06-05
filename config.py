class Config:
    SEED = 42
    seq_len = 64
    n_embed = 96
    batch_size = 12
    vocab_size = 11710  # data vocab size + special tokens = 11706 + 4
    n_heads = 8
    n_layers = 2
    dropout_rate = 0.0
    eta = 4.919042890915579e-06
    eta_o = 2.9e-03
    exp_dir = "exp"
    pos_learnable = True
    optim_type = "sgd"
    epoch = 20
    n_iter = 26
    tau_o = 2
    wub = 0.035284728580901155
    wlb = -0.07318664527441558
    wu = 0.035284728580901155
    wl = -0.035284728580901155
    tau_m = 2.7
    act_fx = "identity"
    act_fx_o = "identity"
    # Tokenizer selection: "BPE" (custom trained) or "tiktoken" (OpenAI's fast tokenizer)
    tokenizer = "BPE"          # Choose: "BPE" or "tiktoken"
    tokenizer_name = "gpt2"    # tiktoken encoding name (used only when tokenizer="tiktoken")
    tokenizer_vocab_file = None # Path to saved BPE tokenizer JSON (used only when tokenizer="BPE")
