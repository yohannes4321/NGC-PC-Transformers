class Config:
    SEED = 42
    seq_len =64
    n_embed = 96
    batch_size = 12
    n_heads = 8
    n_layers = 2
    dropout_rate = 0.0
    eta = 4.919042890915579e-06
    eta_o= 2.9e-03
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "sgd"
    epoch = 1
    n_iter= 26
    tau_o = 2
    # Approximate Xavier scaling: 1 / sqrt(512) is about 0.04
    wub = 0.035284728580901155
    wlb =  -0.07318664527441558
    wu = 0.035284728580901155
    wl = -0.035284728580901155
    tau_m = 2.7
    act_fx = "identity"
    act_fx_o = "identity"

    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "tiktoken"
    tokenizer_encoding = "gpt2"

    tokenizer_vocab_file = None

    # ---- vocab_size: auto-detected from tokenizer backend ----
    # For BPE:     set this to your trained BPE vocab (e.g. 11710)
    # For tiktoken: MUST match the encoding's actual vocab size
    #   o200k_base  → 200256
    #   cl100k_base → 100277
    #   p50k_base   → 50281
    #   gpt2        → 50257
    @classmethod
    def _resolve_vocab_size(cls):
        backend = getattr(cls, "tokenizer", "BPE")
        if isinstance(backend, str) and backend.lower() == "tiktoken":
            from pathlib import Path
            import json
            map_path = Path(__file__).parent / "data_preprocess" / "outputs" / "tokenizer" / f"{cls.tokenizer_encoding}_token_map.json"
            if map_path.exists():
                with open(map_path, "r") as f:
                    return len(json.load(f))
            import tiktoken
            enc = tiktoken.get_encoding(cls.tokenizer_encoding)
            return enc.n_vocab
        else:
            return 11710  # BPE default: data vocab + special tokens

# Set vocab_size once at import time
Config.vocab_size = Config._resolve_vocab_size()
