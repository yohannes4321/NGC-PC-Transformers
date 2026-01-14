class Config:
    SEED = 42
    seq_len =  8
    n_embed = 16
    batch_size = 2
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 2
    n_layers = 8
    dropout_rate = 0.0
    eta = 1.124861336672477e-06
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "adam"
    epoch = 1
    n_iter= 17
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.052687144037646905
    wlb = -0.09881737158539727
    tau_m = 17.
    act_fx = "identity"
    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None