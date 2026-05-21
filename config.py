class Config:
    SEED = 42
    seq_len = 25
    n_embed = 72
    batch_size = 6
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 3
    n_layers = 3
    dropout_rate = 0.08338083927952628
    eta = 2.8302987877207265e-06
    eta_o= 1.0e-03
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "adam"
    epoch = 1
    n_iter= 4
    tau_o = 2
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.016874106399569447
    wlb = -0.017259679300295518
    wu = 0.025
    wl = -0.025
    tau_m = 34.0
    act_fx = "tanh"
    act_fx_o = "identity"
    embed_mult = 24

    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None