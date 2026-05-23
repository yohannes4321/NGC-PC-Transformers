class Config:
    SEED = 42

    # Architecture (updated per user)
    seq_len = 20
    n_embed = 80
    batch_size = 3
    vocab_size = 11710  # data vocab size + special tokens = 11706 + 4
    n_heads = 5
    n_layers = 1
    embed_mult = 16

    # Regularization / optimization
    dropout_rate = 0.0
    eta = 7.082985879191883e-05
    eta_o = 2.9e-03
    optim_type = "adam"

    # Predictive coding parameters
    epoch = 1
    n_iter = 25
    tau_o = 2
    tau_m = 15

    # Weight initialization bounds
    wub = 0.010347072497916304
    wlb = -0.09374572836668062

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