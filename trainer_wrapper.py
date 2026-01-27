"""trainer_wrapper: HPO wrapper around train.run_training
Adapts Nevergrad parameters to JAX types and handles logging.
"""
import os
import gc
import sys
import json
import time

# Ensure JAX memory settings before importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6" # Adjust based on workers

import jax
import numpy as np
from experiment_logger import DualLogger, LOG_DIR
from train import run_training

# --- UTILITIES ---

def clean_memory():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass

def _prepare_params(kwargs):
    """Converts Nevergrad floats to ints and calculates derived params."""
    params = kwargs.copy()
    
    # 1. Cast Integers
    int_keys = ["n_layers", "n_heads", "embed_mult", "batch_size", 
                "seq_len", "tau_m", "n_iter"]
    for k in int_keys:
        if k in params:
            params[k] = int(params[k])
            
    # 2. Derive n_embed (Constraint Logic handled here)
    # n_embed is always divisible by n_heads because we multiply integers
    params["n_embed"] = params["n_heads"] * params["embed_mult"]
    
    return params

def _run_trial_internal(kwargs):
    """Common logic for running a trial and handling logging."""
    params = _prepare_params(kwargs)
    
    # Setup Logging per trial
    # We use a timestamp ID to avoid collision in parallel workers
    trial_id = int(time.time() * 1000) % 1000000
    log_path = os.path.join(LOG_DIR, f"worker_{os.getpid()}_trial_{trial_id}.txt")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    original_stdout = sys.stdout
    logger = DualLogger(log_path)
    sys.stdout = logger
    
    loss_metrics = (1e9, 1e9, 1e9) # Default failure
    
    try:
        print(f"--- START TRIAL {trial_id} (PID {os.getpid()}) ---")
        print(f"Params: {params}")
        
        clean_memory()
        
        # Run Training
        efe, ce, ppl = run_training(
            params_override=params,
            save_model=False,
            max_train_batches=None 
        )
        
        loss_metrics = (float(efe), float(ce), float(ppl))
        print(f"--- END TRIAL {trial_id} --- Result: {loss_metrics}")

    except Exception as e:
        print(f"!!! CRASH TRIAL {trial_id} !!! Error: {e}")
        # Return infinite loss on crash
        loss_metrics = (1e9, 1e9, 1e9)
        
    finally:
        logger.close()
        sys.stdout = original_stdout
        clean_memory()
        
    return loss_metrics

# --- TOP LEVEL FUNCTIONS FOR NEVERGRAD MINIMIZE ---

def evaluate_objective_efe(**kwargs):
    """Objective function for Phase 1. Returns -EFE (to minimize)."""
    efe, ce, ppl = _run_trial_internal(kwargs)
    
    # If efe is infinite (pruned/crashed), return huge number
    if efe >= 1e8:
        return 1e9
        
    # We want to MAXIMIZE EFE (usually negative free energy), 
    # so we MINIMIZE negative EFE.
    # Note: Check your specific EFE definition. 
    # If your EFE is a "loss" (lower is better), return efe.
    # If EFE is "energy" (lower is better), return efe.
    # Assuming standard NGC usage where we minimize Free Energy:
    return efe

def evaluate_objective_ce(**kwargs):
    """Objective function for Phase 2. Returns CE (to minimize)."""
    efe, ce, ppl = _run_trial_internal(kwargs)
    return ce