class Config:
    SEED = 42
    seq_len = 16
    n_embed = 32
    batch_size = 16
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 4
    n_layers = 7
    dropout_rate = 0.032370
    eta = 0.000096
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "adam"
    epoch = 1
    n_iter= 12
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.018249
    wlb =  -0.077128
    tau_m = 12
    act_fx = "relu"
    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"
    hebb_scale = 1.0 / ((batch_size * seq_len) ** 0.5)
    sigma_norm = (n_embed * batch_size * seq_len) ** 0.5
    sigma_norm_mlp1 = float((batch_size * seq_len * (4 * n_embed)) ** 0.5)

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None