import os
import sys
import numpy as np
import uuid
# Import the actual training function from your train.py file
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
    # Create a unique ID for this specific trial to help tracking in logs
    trial_id = uuid.uuid4().hex[:4]
    
    # 1. ORCHESTRATOR ANNOUNCEMENT
    print(f"\n" + "!"*80)
    print(f"📢 ORCHESTRATOR: Launching Trial [{trial_id}] | Objective: {objective.upper()}")
    print("!"*80)

    try:
        # 2. THE TRAINER EXECUTION
        # This calls the run_training function in train.py which prints the full config
        metrics = run_training(params_override=params)
        
        # 3. METRIC EXTRACTION
        efe_val = metrics.get("avg_train_efe", 1e10)
        ce_val = metrics.get("val_ce", 1e10)

        # 4. OBJECTIVE SELECTION (The "Score" sent back to Nevergrad)
        if objective == "efe":
            # Phase 1: We minimize the absolute value of EFE
            # This ensures the 'Body' of the model is stable.
            loss = abs(efe_val)
            print(f"\n✅ [Trial {trial_id}] Phase 1 (EFE) Result: {loss:.4f}")
        else:
            # Phase 2: We minimize Cross-Entropy
            # This ensures the 'Brain' of the model is accurate.
            loss = ce_val
            print(f"\n✅ [Trial {trial_id}] Phase 2 (CE) Result: {loss:.4f}")

        # Nevergrad expects a 2D array [ [value] ]
        return np.array([[float(loss)]])

    except Exception as e:
        # If the model crashes (like OOM), report a massive penalty score
        print(f"\n❌ [Trial {trial_id}] CRITICAL FAILURE: {str(e)}")
        return np.array([[1e20]])

if __name__ == "__main__":
    # Small test case to ensure the wrapper works independently
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