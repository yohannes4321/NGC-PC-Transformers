# # trainer wrapper
# import os
# # Disable pre-allocation so JAX only takes what it needs
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# import sys
# import gc
# import numpy as np
# import uuid
# import jax

# from train import run_training 

# def _cleanup_run():
#     """Release JAX compilation caches and trigger garbage collection."""
#     try:
#         # This clears the compiled kernels to prevent memory bloat 
#         # over 100+ HPO trials
#         jax.clear_caches()
#     except Exception:
#         pass
#     gc.collect()

# def train_evaluate_model(params, objective="efe"):
#     """
#     Args:
#         params (dict): Hyperparameters from Nevergrad.
#         objective (str): 'efe' (stability) or 'ce' (accuracy).
#     """
#     trial_id = uuid.uuid4().hex[:4]
    
#     # Pre-run cleanup
#     _cleanup_run()

#     try:

#         metrics = run_training(params_override=params)
        
       
#         efe_val_signed = metrics.get("best_train_efe", 1e10)
#         efe_val_abs = metrics.get("best_train_efe_abs", 1e10)
#         ce_val = metrics.get("best_val_ce", 1e10)
#         ppl_val = metrics.get("best_val_ppl", 99999.0)
#         plateau = metrics.get("plateau_triggered", False)
        
#         # SELECT LOSS FOR OPTIMIZER
#         if objective == "efe":
#             # Minimizing the absolute distance of EFE to zero
#             loss = float(abs(efe_val_abs) )
#             obj_tag = "EFE (Abs)"
#         else:
#             # Minimizing Cross-Entropy
#             loss = float(ce_val)
#             obj_tag = "CE"

#         # --- LOGGING ---
#         status = "PLATEAU-STOP" if plateau else "SUCCESS"
#         print(f"--- [Trial {trial_id} | {status}] Finished. ---", flush=True)
#         print(f"    > EFE: {efe_val_signed:.4f} (Abs: {efe_val_abs:.4f})", flush=True)
#         print(f"    > CE:  {ce_val:.4f} | PPL: {ppl_val:.4f}", flush=True)
#         print(f"    > Returning Loss ({obj_tag}): {loss:.4f}", flush=True)

#         return np.array([[float(loss)]])

#     except Exception as e:
#         print(f"\n[Trial {trial_id}] CRITICAL FAILURE: {str(e)}", flush=True)
#         # Import traceback to see why it's failing if it does
#         import traceback
#         traceback.print_exc()
#         return np.array([[1e20]]) 

#     finally:
#         _cleanup_run()

"""trainer_wrapper: thin HPO wrapper around train.run_training

This module receives hyperparameters from Nevergrad (main_hebo.py), calls
train.run_training() with those overrides, logs the trial, and persists the
best-so-far params across runs. It also aggressively clears JAX memory caches
between trials to avoid OOMs.
"""

# filename: trainer_wrapper.py
import time
import os
import math
import gc
import sys
import json

# Ensure JAX memory settings before importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"

import jax
import numpy as np
import pandas as pd

from experiment_logger import save_to_csv, DualLogger, LOG_DIR

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train import run_training


def clean_memory():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


def _load_best_payload():
    best_path = os.path.join(LOG_DIR, "best_params.json")
    if os.path.exists(best_path):
        try:
            with open(best_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _maybe_update_best(trial_id: int, params: dict, loss: float, ppl: float):
    best_payload = _load_best_payload()
    if best_payload is None or loss < float(best_payload.get("cross_entropy", float("inf"))):
        serializable_params = {
            k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()
        }
        payload = {
            "trial_id": trial_id,
            "cross_entropy": float(loss),
            "ppl": float(ppl),
            "params": serializable_params,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        best_path = os.path.join(LOG_DIR, "best_params.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True
    return False


def train_evaluate_model(params: dict, objective: str = "ce"):
    # standardize to pandas Series for CSV logging
    if not isinstance(params, dict):
        raise ValueError("train_evaluate_model expects a dict of hyperparameters")

    existing_trials = len(os.listdir(LOG_DIR)) if os.path.exists(LOG_DIR) else 0
    trial_id = existing_trials
    log_path = os.path.join(LOG_DIR, f"trial_{trial_id}.txt")

    os.makedirs(LOG_DIR, exist_ok=True)
    original_stdout = sys.stdout
    logger = DualLogger(log_path)
    sys.stdout = logger

    try:
        print("==========================================")
        print(f"STARTING TRIAL {trial_id}")
        print("Params:", params)
        print("==========================================")

        clean_memory()

        # Ensure Nevergrad-provided batch/seq flow consistently into model and DataLoader
        if "batch_size" in params:
            params["batch_size"] = int(params["batch_size"])
        if "seq_len" in params:
            params["seq_len"] = int(params["seq_len"])

        # Run full training (no batch cap); adjust in params if you need a limit
        efe,ce,ppl = run_training(
            params_override=params,
            save_model=False,
            max_train_batches=None,
        )

        if objective == "efe":
            # The goal is to move from a large negative number toward 0.
            # We want to MAXIMIZE the raw EFE. 

            efe_raw = float(efe)
            
          #
            loss = -(efe_raw)
        

        elif objective == "ce":
            loss=ce
        else:
            raise ValueError(f"Unsupported objective '{objective}'")

        # Return 2D array to keep main_hebo.py untouched
        return np.array([[loss]], dtype=float)

    except Exception as e:
        print(f"!!! CRASH IN TRIAL {trial_id} !!!")
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg:
            print("Detected GPU OOM; returning inf and proceeding.")
        else:
            print(f"Error: {msg}")
        return np.array([[float("inf")]], dtype=float)

    finally:
        logger.close()
        sys.stdout = original_stdout
        clean_memory()