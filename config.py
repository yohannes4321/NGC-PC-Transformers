class Config:
    SEED = 42

    # Architecture (updated from Trial 22)
    seq_len = 11
    n_embed = 64
    batch_size = 3
    vocab_size = 11710  # data vocab size + special tokens = 11706 + 4
    n_heads = 8
    n_layers = 3
    embed_mult = 8

    # Regularization / optimization
    dropout_rate = 0.0
    eta = 1.534019603677043e-05
    eta_o = 2.9e-03
    optim_type = "adam"

    # Predictive coding parameters
    epoch = 1
    n_iter = 1
    tau_o = 2
    tau_m = 16

    # Weight initialization bounds
    wub = 0.08258106460608025
    wlb = -0.08039278154210332

    # Internal state init
    wu = 0.035284728580901155
    wl = -0.035284728580901155

    # Activations
    act_fx = "relu"
    act_fx_o = "relu"

    # Positional embeddings
    pos_learnable = True

    # Misc
    exp_dir = "exp"

    # Tokenizer selection: "BPE" or "tiktoken"
    tokenizer = "BPE"

    # When tokenizer == "tiktoken"
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE"
    tokenizer_vocab_file = None