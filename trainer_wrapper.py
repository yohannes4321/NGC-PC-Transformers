import time
import os
import math
import gc
import sys
import uuid
import jax
import numpy as np
from experiment_logger import save_to_csv, DualLogger, LOG_DIR

def train_evaluate_model(params: dict, objective: str = "ce"):
    # Generate a tag for this specific worker
    short_id = uuid.uuid4().hex[:4]
    unique_tag = f"{int(time.time())}_{short_id}"
    log_path = os.path.join(LOG_DIR, f"trial_{unique_tag}.txt")
    
    # Redirect output to our DualLogger
    original_stdout = sys.stdout
    logger = DualLogger(log_path, terminal_prefix=short_id)
    sys.stdout = logger

    try:
        from train import run_training
        
        # Cast JAX-critical params
        for k in ["batch_size", "seq_len", "n_heads", "n_embed"]:
            if k in params: params[k] = int(params[k])

        print(f"--- STARTING TRIAL {unique_tag} ---")
        metrics = run_training(params_override=params, save_model=False)

        # Determine loss based on the HPO phase
        if objective == "efe":
            loss = abs(float(metrics.get("avg_train_efe", 1e6)))
        else:
            loss = float(metrics.get("val_ce", 1e6))

        save_to_csv(unique_tag, params, metrics)
        print(f"--- TRIAL FINISHED. LOSS ({objective}): {loss:.4f} ---")
        
        return np.array([[loss]], dtype=float)

    except Exception as e:
        # Crucial: print error to original terminal so you see crashes
        original_stdout.write(f"\n[CRASH {short_id}] Error: {str(e)}\n")
        return np.array([[float("inf")]], dtype=float)
    finally:
        sys.stdout = original_stdout
        logger.close()
        gc.collect()
        try: jax.clear_caches()
        except: pass