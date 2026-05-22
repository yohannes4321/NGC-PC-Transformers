class Config:
    SEED = 42
    seq_len = 64
    n_embed = 96
    batch_size = 32
    vocab_size = 11710# data vocab size + special tokens = 11706 + 4
    n_heads = 8
    n_layers = 2
    dropout_rate = 0.0
    eta = 4.919042890915579e-06
    eta_o = 2.9e-03
    exp_dir = "exp"
    pos_learnable = True
    # Switch to Adam optimizer for better gradient handling with large loss magnitudes
    optim_type = "adam"
    epoch = 1
    n_iter= 26
    tau_o = 2
    
    # === OPTIMIZED FOR MONOTONIC ENERGY DECREASE ===
    # Slower, more stable hidden layer dynamics
    tau_m = 4.0
    
    # Increased learning rate for hidden layers (was 4.919e-06)
    # Now 20× larger to match output layer learning speed
    eta = 1e-04
    eta_o = 2.9e-03
    
    # Relaxed weight bounds (was ±0.02)
    # Allow more dynamic range for weight representation
    wub = 0.1
    wlb = -0.1
    wu = 0.1
    wl = -0.1
    
    # === ACTIVATION FUNCTIONS (DO NOT CHANGE) ===
    # Identity on hidden layers: unbounded rate neurons
    # This is correct for rate-coded RateCell dynamics
    act_fx = "identity"
    
    # Tanh on output layer: CRITICAL for energy stability
    # Bounds predictions to [-1, 1] → error always ≤ 1
    # → Energy naturally decreases monotonically
    # (without tanh: Energy oscillates wildly)
    act_fx_o = "tanh"
    # Tokenizer selection: "BPE" (custom/BPE loader) or "tiktoken"
    tokenizer = "BPE"
    # When tokenizer == "tiktoken", tokenizer_name is used (e.g. "gpt2" or "cl100k_base")
    tokenizer_name = "gpt2"

    # When tokenizer == "BPE", tokenizer_vocab_file may point to a vocab json or a newline token list.
    # Optional: set to None to use a simple fallback whitespace tokenizer.
    tokenizer_vocab_file = None