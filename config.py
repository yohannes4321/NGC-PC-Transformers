class Config:
    SEED = 42
    seq_len = 19
    n_embed = 64
    batch_size = 2
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 2
    n_layers = 3
    dropout_rate = 0.05382108663347571
    eta = 1.3221610259416427e-06
    eta_o= 1.0e-03
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "sgd"
    epoch = 1
    n_iter= 3
    tau_o = 2
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.025
    wlb = -0.026564232301038024
    wu = 0.025
    wl = -0.025
    tau_m = 32.0
    act_fx = "identity"
    act_fx_o = "identity"
    embed_mult = 32

    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None