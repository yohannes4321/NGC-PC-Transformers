class Config:
    SEED = 42
    n_embed = 64
    seq_len = 32
    batch_size = 32
    vocab_size = 11710 # data vocab size + special tokens = 11706 + 4
    n_heads = 8
    n_layers = 6
    dropout_rate = 0.1
    eta = 0.001
    exp_dir = "exp" 
    pos_learnable = True
    optim = "adam"
    num_iter = 10