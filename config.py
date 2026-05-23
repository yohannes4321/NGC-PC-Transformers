import numpy as np

class Config:
    SEED = 42

    # Architecture
    seq_len = 8
    n_embed = 64
    batch_size = 4      # Scaled up slightly for stable batch gradients
    vocab_size = 11710  # data vocab size + special tokens
    n_heads = 8
    n_layers = 8
    embed_mult = 8

    # Regularization / Optimization
    dropout_rate = 0.01 # Small amount of regularized noise to break uniform plateaus
    eta = 1.5e-5        # Stable predictive coding state step-size
    eta_o = 1.0e-3      # Adam weight learning rate
    optim_type = "adam" # Switch back to adam for non-zero gradient moment tracking

    # Predictive Coding Core Dynamics (Fixed for continuous reduction)
    epoch = 5
    n_iter = 25         # CRITICAL: Budget for settling internal states to minimize EFE
    tau_o = 5           # Time constant for output error settling
    tau_m = 12          # Time constant for internal layer representations

    # Weight Initialization Bounds (Xavier-adjusted for predictive scaling)
    wub = 0.05
    wlb = -0.05

    # Internal State Initializations
    wu = 0.01
    wl = -0.01

    # Activations
    act_fx = "relu"     # Non-linear internal representation to handle token non-linearities
    act_fx_o = "identity"

    # Positional Embeddings
    # NOTE: If absolute learnable embeddings are False, ensure you have
    # switched to a relative scheme like Rotary Positional Embedding (RoPE) 
    # to avoid semantic word-position entanglement.
    pos_learnable = False

    # Misc
    exp_dir = "exp"
    tokenizer = "BPE"
    tokenizer_name = "gpt2"
    tokenizer_vocab_file = None