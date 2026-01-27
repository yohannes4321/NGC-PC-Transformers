"""trainer_wrapper: HPO wrapper around train.run_training
Optimized for Nevergrad minimization using absolute values for EFE.
"""
import os
import gc
import sys
import time

# Pre-import environment setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6" 

import jax
import numpy as np
from experiment_logger import DualLogger, LOG_DIR
from train import run_training

def clean_memory():
    gc.collect()
    try:
        jax.clear_caches()
    except:
        pass

def _prepare_params(args, kwargs):
    """Handles both positional dict and keyword arguments from Nevergrad."""
    raw_params = args[0] if (args and isinstance(args[0], dict)) else kwargs
    params = raw_params.copy()
    
    # Cast necessary keys to integers for JAX compatibility
    int_keys = ["n_layers", "n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter"]
    for k in int_keys:
        if k in params:
            params[k] = int(params[k])
            
    # Calculate model dimension
    if "n_heads" in params and "embed_mult" in params:
        params["n_embed"] = params["n_heads"] * params["embed_mult"]
    
    return params

def _run_trial_internal(args, kwargs):
    params = _prepare_params(args, kwargs)
    
    # Unique log ID based on timestamp and Process ID
    trial_id = int(time.time() * 1000) % 1000000
    log_path = os.path.join(LOG_DIR, f"trial_{trial_id}_pid_{os.getpid()}.txt")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    original_stdout = sys.stdout
    logger = DualLogger(log_path)
    sys.stdout = logger
    
    try:
        print(f"--- STARTING TRIAL: PID {os.getpid()} ---")
        print(f"Hyperparameters: {params}")
        clean_memory()
        
        # Execute training
        efe, ce, ppl = run_training(
            params_override=params,
            save_model=False,
            max_train_batches=None 
        )
        
        return float(efe), float(ce), float(ppl)

    except Exception as e:
        print(f"Trial Failed: {e}")
        return 1e9, 1e9, 1e9 # Return large penalty on failure
    finally:
        logger.close()
        sys.stdout = original_stdout
        clean_memory()

# --- NEVERGRAD ENTRY POINTS ---

def evaluate_objective_efe(*args, **kwargs):
    """Phase 1: Minimize the absolute value of EFE."""
    efe, ce, ppl = _run_trial_internal(args, kwargs)
    
    # Logic: If EFE is negative (e.g., -500), abs() makes it 500. 
    # If the goal is to reach a 'target' of 0 or reduce energy, 
    # Nevergrad will push this positive value toward 0.
    return abs(efe)

def evaluate_objective_ce(*args, **kwargs):
    """Phase 2: Minimize Cross-Entropy (standard loss)."""
    efe, ce, ppl = _run_trial_internal(args, kwargs)
    return ce