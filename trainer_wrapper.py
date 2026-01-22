# trainer wrapper

# Disable pre-allocation so JAX only takes what it needs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import gc
import numpy as np
import uuid
import jax

# IMPORTANT: Ensure the directory containing your train.py is in the path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your real training function
from train import run_training 

def _cleanup_run():
    """Release JAX compilation caches and trigger garbage collection."""
    try:
        # This clears the compiled kernels to prevent memory bloat 
        # over 100+ HPO trials
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()

def train_evaluate_model(params, objective="efe"):
    """
    Args:
        params (dict): Hyperparameters from Nevergrad.
        objective (str): 'efe' (stability) or 'ce' (accuracy).
    """
    trial_id = uuid.uuid4().hex[:4]
    
    # Pre-run cleanup
    _cleanup_run()

    try:
        # --- EXECUTE REAL TRAINING ---
        # This calls your function that initializes NGCTransformer
        metrics = run_training(params_override=params)
        
        # EXTRACT METRICS FROM YOUR REAL DICTIONARY
        # We use .get() with high defaults to penalize failures
        efe_val_signed = metrics.get("best_train_efe", 1e10)
        efe_val_abs = metrics.get("best_train_efe_abs", 1e10)
        ce_val = metrics.get("best_val_ce", 1e10)
        ppl_val = metrics.get("best_val_ppl", 99999.0)
        plateau = metrics.get("plateau_triggered", False)
        
        # SELECT LOSS FOR OPTIMIZER
        if objective == "efe":
            # Minimizing the absolute distance of EFE to zero
            loss = float(efe_val_abs) 
            obj_tag = "EFE (Abs)"
        else:
            # Minimizing Cross-Entropy
            loss = float(ce_val)
            obj_tag = "CE"

        # --- LOGGING ---
        status = "PLATEAU-STOP" if plateau else "SUCCESS"
        print(f"--- [Trial {trial_id} | {status}] Finished. ---", flush=True)
        print(f"    > EFE: {efe_val_signed:.4f} (Abs: {efe_val_abs:.4f})", flush=True)
        print(f"    > CE:  {ce_val:.4f} | PPL: {ppl_val:.4f}", flush=True)
        print(f"    > Returning Loss ({obj_tag}): {loss:.4f}", flush=True)

        return np.array([[float(loss)]])

    except Exception as e:
        print(f"\n[Trial {trial_id}] CRITICAL FAILURE: {str(e)}", flush=True)
        # Import traceback to see why it's failing if it does
        import traceback
        traceback.print_exc()
        return np.array([[1e20]]) 

    finally:
        _cleanup_run()

if __name__ == "__main__":
    # Test with a small configuration
    test_params = {
        "n_heads": 2, 
        "embed_mult": 4, 
        "batch_size": 8, 
        "num_iter": 1, 
        "n_iter": 5 # This is 'T' in your model
    }
    print("Testing Wrapper with Real Model...", flush=True)
    res = train_evaluate_model(test_params, objective="ce")