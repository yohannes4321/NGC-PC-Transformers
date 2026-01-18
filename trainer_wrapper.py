import time
import os
import math
import gc
import sys
import uuid
import jax
import numpy as np
from experiment_logger import save_to_csv, DualLogger, LOG_DIR

def train_evaluate_model(params: dict, objective: str = "efe"): # Default changed to efe
    short_id = uuid.uuid4().hex[:4]
    unique_tag = f"{int(time.time())}_{short_id}"
    log_path = os.path.join(LOG_DIR, f"trial_{unique_tag}.txt")
    
    original_stdout = sys.stdout
    logger = DualLogger(log_path, terminal_prefix=short_id)
    sys.stdout = logger

    try:
        from train import run_training
        
        for k in ["batch_size", "seq_len", "n_heads", "n_embed"]:
            if k in params: params[k] = int(params[k])

        print(f"--- STARTING TRIAL {unique_tag} ---")
        metrics = run_training(params_override=params, save_model=False)

        # LOGIC: If EFE is -2000 and we want it to reach 0, 
        # we minimize the absolute value.
        efe_val = float(metrics.get("avg_train_efe", 1e6))
        ce_val = float(metrics.get("val_ce", 1e6))

        if objective == "efe":
            loss = abs(efe_val)
        elif objective == "combined":
            # Focus 80% on EFE and 20% on CE
            # We use log(ce) to keep it in a similar range to EFE
            loss = abs(efe_val) + (ce_val * 10) 
        else:
            loss = ce_val

        save_to_csv(unique_tag, params, metrics)
        print(f"--- TRIAL FINISHED. EFE: {efe_val:.2f} | CE: {ce_val:.4f} ---")
        
        return np.array([[loss]], dtype=float)

    except Exception as e:
        original_stdout.write(f"\n[CRASH {short_id}] Error: {str(e)}\n")
        return np.array([[float("inf")]], dtype=float)
    finally:
        sys.stdout = original_stdout
        logger.close()
        gc.collect()
        try: jax.clear_caches()
        except: pass