class Config:
    SEED = 42
    seq_len = 12
    n_embed = 24
    batch_size = 3
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 2
    n_layers = 1
    dropout_rate = 0.010617
    eta = 0.000149
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "adam"
    epoch = 1
    n_iter= 10
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.055677
    wlb =  -0.093495
    tau_m = 12
    act_fx = "relu"
    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None