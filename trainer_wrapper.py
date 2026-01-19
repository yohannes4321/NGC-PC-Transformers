import os
import sys
import numpy as np
import uuid

from train import run_training 

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

    try:
        metrics = run_training(params_override=params)
        efe_val = metrics.get("best_train_efe", metrics.get("avg_train_efe", 1e10))
        ce_val = metrics.get("best_val_ce", metrics.get("val_ce", 1e10))
        ppl_val = metrics.get("best_val_ppl", metrics.get("val_ppl", None))
        batches_ran = metrics.get("batches_ran", None)
        plateau = metrics.get("plateau_triggered", False)

        if objective == "efe":
            loss = abs(efe_val)
            print(f"\n [Trial {trial_id}] Phase 1 (EFE) Result: {loss:.4f}")
        else:
            loss = ce_val
            print(f"\n [Trial {trial_id}] Phase 2 (CE) Result: {loss:.4f}")

        print(f"   Best EFE: {efe_val:.4f} | Best CE: {ce_val:.4f}" + (f" | Best PPL: {ppl_val:.4f}" if ppl_val is not None else ""))
        if batches_ran is not None:
            print(f"   Batches processed: {batches_ran}")
        if plateau:
            print("   Early stop reason: plateau stability")

        return np.array([[float(loss)]])

    except Exception as e:
        print(f"\n[Trial {trial_id}] CRITICAL FAILURE: {str(e)}")
        return np.array([[1e20]])

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