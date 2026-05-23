class Config:
    SEED = 42

    # Architecture (Phase 1 best: Trial 3)
    seq_len = 8
    n_embed = 64
    batch_size = 2
    vocab_size = 11710  # data vocab size + special tokens = 11706 + 4
    n_heads = 8
    n_layers = 8
    embed_mult = 8

    # Regularization / optimization
    dropout_rate = 0.0
    eta = 1.0315873044754272e-06
    eta_o = 2.9e-03
    optim_type = "adam"

    # Predictive coding parameters
    epoch = 1
    n_iter = 50
    tau_o = 5
    tau_m = 10

    # Weight initialization bounds
    wub = 0.08590467449638088
    wlb = -0.09407300665329076

    # Internal state init
    wu = 0.035284728580901155
    wl = -0.035284728580901155

    # Activations
    act_fx = "identity"
    act_fx_o = "relu"

    # Positional embeddings
    pos_learnable = False

    # Misc
    exp_dir = "exp"

    # Tokenizer selection: "BPE" or "tiktoken"
    tokenizer = "BPE"

    # When tokenizer == "tiktoken"
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE"
    tokenizer_vocab_file = None