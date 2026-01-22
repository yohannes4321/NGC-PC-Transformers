import time
import os
import math
import gc
import sys
import json
import uuid
import jax
import numpy as np
import pandas as pd
from experiment_logger import save_to_csv, DualLogger, LOG_DIR

def clean_memory():
    gc.collect()
    try: jax.clear_caches()
    except: pass

def train_evaluate_model(params: dict, objective: str = "ce"):
    # Create a TRULY unique ID for this specific run
    unique_tag = f"{int(time.time())}_{uuid.uuid4().hex[:4]}"
    log_path = os.path.join(LOG_DIR, f"trial_{unique_tag}.txt")

    os.makedirs(LOG_DIR, exist_ok=True)
    original_stdout = sys.stdout
    logger = DualLogger(log_path)
    
    # We only redirect if we aren't already inside a DualLogger
    sys.stdout = logger

    try:
        print("\n" + "="*40)
        print(f"STARTING UNIQUE TRIAL: {unique_tag}")
        print(f"OBJECTIVE: {objective}")
        print(f"PARAMS: {params}")
        print("="*40, flush=True)

        from train import run_training
        
        # JAX Integer casting
        for k in ["batch_size", "seq_len", "n_heads", "n_embed"]:
            if k in params: params[k] = int(params[k])

        metrics = run_training(params_override=params, save_model=False)

        if objective == "efe":
            raw_val = float(metrics.get("avg_train_efe", 1e6))
            loss = abs(raw_val)
        else:
            loss = float(metrics.get("val_ce", 1e6))

        ppl = math.exp(min(loss, 20)) 

        # Log to CSV
        save_to_csv(unique_tag, pd.Series(params), {
            "cross_entropy": float(metrics.get("val_ce", 0)),
            "ppl": ppl,
            "efe": float(metrics.get("avg_train_efe", 0)),
        })

        print(f"\nTrial {unique_tag} Finished. Loss: {loss:.4f}", flush=True)
        return np.array([[loss]], dtype=float)

    except Exception as e:
        print(f"!!! CRASH IN TRIAL {unique_tag} !!! Error: {e}", flush=True)
        return np.array([[float("inf")]], dtype=float)

    finally:
        logger.close()
        sys.stdout = original_stdout
        clean_memory()