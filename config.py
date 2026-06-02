class Config:
    SEED = 42
    seq_len = 4
    n_embed = 32
    batch_size = 2
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 4
    n_layers = 5
    dropout_rate = 0.15662144617228202
    eta = 1.0744583847151172e-06
    eta_o= 2.9e-03
    exp_dir = "exp" 
    pos_learnable = False
    optim_type = "sgd"
    epoch = 1
    n_iter= 13
    tau_o = 2
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.047663792993867665
    wlb = -0.0988265802505217
    wu = 0.035284728580901155
    wl = -0.035284728580901155
    tau_m = 14
    act_fx = "identity"
    act_fx_o = "identity"

    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None