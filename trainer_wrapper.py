import os
import sys
import gc
import numpy as np
import uuid
import jax

# Import your actual training function
# from train import run_training  <-- Uncomment this in your real environment
# Mocking it for this example so the code runs if you test it:
def run_training(params_override):
    # This is a dummy return to simulate your training. 
    # Replace with your actual import.
    return {
        "best_train_efe": -50.0 + np.random.randn(), # Negative EFE
        "best_val_ce": 15.0 - np.random.rand(),
        "best_val_ppl": 200.0,
        "plateau_triggered": False
    }

def _cleanup_run():
    """Release JAX compilation caches and trigger garbage collection."""
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()

def train_evaluate_model(params, objective="efe"):
    """
    Args:
        params (dict): Hyperparameters.
        objective (str): 'efe' (optimize stability) or 'ce' (optimize accuracy).
        
    Returns:
        np.array: Loss value for the optimizer.
    """
    trial_id = uuid.uuid4().hex[:4]
    
    # Reset state
    _cleanup_run()

    try:
        # --- RUN YOUR TRAINING ---
        metrics = run_training(params_override=params)
        
        # EXTRACT METRICS
        efe_val_signed = metrics.get("best_train_efe", 1e10) # e.g., -100.5
        efe_val_abs = abs(efe_val_signed)                    # e.g., 100.5 (Distance from 0)
        
        ce_val = metrics.get("best_val_ce", 1e10)
        ppl_val = metrics.get("best_val_ppl", 99999.0)
        
        # SELECT LOSS FOR OPTIMIZER
        if objective == "efe":
            # OPTIMIZE DISTANCE TO ZERO
            # If EFE is -100, loss is 100. If EFE is -10, loss is 10.
            # Optimizer minimizes loss, so it prefers -10.
            loss = float(efe_val_abs) 
            obj_tag = "EFE (Abs)"
        else:
            loss = float(ce_val)
            obj_tag = "CE"

        # --- PRINT RESULT TO SCREEN (FLUSHED) ---
        print(f"--- [Trial {trial_id}] Finished. ---", flush=True)
        print(f"    > EFE: {efe_val_signed:.4f} (Abs: {efe_val_abs:.4f})", flush=True)
        print(f"    > CE:  {ce_val:.4f}", flush=True)
        print(f"    > PPL: {ppl_val:.4f}", flush=True)
        print(f"    > Returning Loss ({obj_tag}): {loss:.4f}", flush=True)

        return np.array([[float(loss)]])

    except Exception as e:
        print(f"\n[Trial {trial_id}] CRITICAL FAILURE: {str(e)}", flush=True)
        return np.array([[1e20]]) # Return huge penalty

    finally:
        _cleanup_run()

if __name__ == "__main__":
    test_params = {"n_heads": 2, "embed_mult": 8, "batch_size": 16}
    print("Testing Wrapper...", flush=True)
    res = train_evaluate_model(test_params, objective="efe")