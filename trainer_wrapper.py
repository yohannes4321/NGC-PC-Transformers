import os
import sys
import gc
import numpy as np
import uuid
import jax

from train import run_training 


def _cleanup_run():
    """Release JAX compilation caches and trigger garbage collection."""
    try:
        jax.clear_caches()
    except Exception:
        # If JAX backend is already torn down, just continue.
        pass
    gc.collect()

def train_evaluate_model(params, objective="efe"):
    """
    The Translator/Bridge Function.
    
    Args:
        params (dict): The full set of 11+ hyperparameters from Nevergrad.
        objective (str): Either 'efe' (Phase 1) or 'ce' (Phase 2).
        
    Returns:
        np.array: A 2D array containing the single loss value for the optimizer.
    """
    trial_id = uuid.uuid4().hex[:4]
    
    print(f"\n" + "!"*80)
    print(f"ORCHESTRATOR: Launching Trial [{trial_id}] | Objective: {objective.upper()}")
    print("!"*80)

    # Reset any residual state before starting a new trial
    _cleanup_run()

    try:
        metrics = run_training(params_override=params)
        efe_val_signed = metrics.get("best_train_efe", metrics.get("avg_train_efe", 1e10))
        efe_val_abs = metrics.get("best_train_efe_abs", abs(efe_val_signed))
        ce_val = metrics.get("best_val_ce", metrics.get("val_ce", 1e10))
        ppl_val = metrics.get("best_val_ppl", metrics.get("val_ppl", None))
        batches_ran = metrics.get("batches_ran", None)
        plateau = metrics.get("plateau_triggered", False)

        if objective == "efe":
            loss = float(efe_val_abs)  # minimize magnitude of EFE (closer to 0 is better)
            print(f"\n [Trial {trial_id}] Phase 1 (EFE) Result: {efe_val_signed:.4f} (abs {loss:.4f})")
        else:
            loss = float(ce_val)  # use best CE found
            print(f"\n [Trial {trial_id}] Phase 2 (CE) Result: {loss:.4f} (best CE)")

        print(
            f"   Best EFE: {efe_val_signed:.4f} (abs {efe_val_abs:.4f}) | "
            f"Best CE: {ce_val:.4f}" + (f" | Best PPL: {ppl_val:.4f}" if ppl_val is not None else "")
        )
        # Intentionally skip printing batch count; focus on best metrics only
        if plateau:
            print("   Early stop reason: plateau stability")

        print(f"   -> Returning loss {loss:.4f} to Nevergrad")

        return np.array([[float(loss)]])

    except Exception as e:
        print(f"\n[Trial {trial_id}] CRITICAL FAILURE: {str(e)}")
        return np.array([[1e20]])

    finally:
        # Make each trial memory-neutral for all workers
        _cleanup_run()

if __name__ == "__main__":
    test_params = {
        "n_heads": 2, 
        "embed_mult": 8, 
        "n_embed": 16,
        "batch_size": 16,
        "eta": 1e-5
    }
    print("Testing Wrapper...")
    res = train_evaluate_model(test_params, objective="efe")
    print(f"Test Result: {res}")