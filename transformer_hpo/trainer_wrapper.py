# filename: trainer_wrapper.py
import time
import math
import gc
import sys
import os
import jax
import jax.numpy as jnp
# Force JAX to use the GPU for all operations
jax.config.update("jax_default_device", jax.devices("gpu")[0])
# Ensure JAX doesn't pre-allocate 90% of your VRAM immediately (prevents OOM)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import pandas as pd
from experiment_logger import save_to_csv, DualLogger, LOG_DIR
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import Config as config
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader

def clean_memory():
    """Forces garbage collection and clears JAX compilation caches."""
    gc.collect()
    # This is the modern replacement for clearing JAX memory
    jax.clear_caches()

def train_evaluate_model(params_df: pd.DataFrame):
    """
    The Objective Function for HEBO.
    Receives a DataFrame of suggestions, trains the model, logs results, 
    and returns the Loss (CL) to be minimized.
    """
    results = []
    
    for i in range(len(params_df)):
        # 1. SETUP LOGGING FOR THIS TRIAL
        # Determine trial ID based on existing files to avoid overwrites
        existing_trials = len(os.listdir(LOG_DIR))
        trial_id = existing_trials
        log_path = os.path.join(LOG_DIR, f"trial_{trial_id}.txt")
        
        # Hijack stdout to print to file and console
        original_stdout = sys.stdout
        logger = DualLogger(log_path)
        sys.stdout = logger
        
        try:
            # 2. PARSE PARAMETERS
            p = params_df.iloc[i]
            
            # --- Enforce Constraint: n_embed % n_heads == 0 ---
            n_heads = int(p['n_heads'])
            raw_n_embed = int(p['n_embed'])
            n_embed = (raw_n_embed // n_heads) * n_heads
            if n_embed == 0: n_embed = n_heads
            
            curr_batch_size = int(p['batch_size'])
            block_size = int(p['block_size'])
            
            print(f"==========================================")
            print(f"STARTING TRIAL {trial_id}")
            print(f"Params: n_embed={n_embed} (adj), heads={n_heads}, layers={int(p['n_layers'])}")
            print(f"==========================================")
            
            clean_memory()
            
            # 3. INITIALIZE MODEL & DATA
            loader = DataLoader(seq_len=block_size, batch_size=curr_batch_size)
            train_loader, _, _ = loader.load_and_prepare_data()
            dkey = jax.random.PRNGKey(int(time.time()))
            
            model = NGCTransformer(
                dkey,
                batch_size=curr_batch_size,
                seq_len=block_size,
                n_embed=n_embed,
                vocab_size=config.vocab_size,
                n_layers=int(p['n_layers']),
                n_heads=n_heads,
                T=int(p['T']),
                dt=1.0,
                tau_m=float(p['tau_m']),
                act_fx=p['act_fx'],
                eta=float(p['eta']),
                dropout_rate=float(p['dropout']),
                pos_learnable=config.pos_learnable,
                optim_type=config.optim_type,
                wub=float(p['wub']),
                wlb=float(p['wlb']),
                exp_dir="exp",
                model_name=f"HEBO_{trial_id}"
            )
            
            # 4. TRAINING LOOP
            total_efe = 0.0
            total_ce = 0.0
            batch_count = 0
            max_batches = 50 # Optimization speed hack
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_batches: break
                
                # Adapt this to match your actual DataLoader output structure
                inputs = batch[0][1] 
                targets = batch[1][1]
                
                # One-hot encoding logic
                if hasattr(jax.nn, 'one_hot'):
                    targets_onehot = jax.nn.one_hot(targets, config.vocab_size)
                else:
                    # Fallback for dummy
                    targets_onehot = jnp.eye(config.vocab_size)[targets.astype(int)]
                    
                targets_flat = targets_onehot.reshape(-1, config.vocab_size)
                
                # Forward
                yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
                y_pred = yMu_inf.reshape(-1, config.vocab_size)
                
                batch_ce = float(measure_CatNLL(y_pred, targets_flat).mean())
                batch_efe = float(_EFE)
                
                total_ce += batch_ce
                total_efe += batch_efe
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}: CE={batch_ce:.4f}, EFE={batch_efe:.4f}")

            # 5. CALCULATE METRICS
            avg_ce = total_ce / batch_count if batch_count > 0 else float('inf')
            avg_efe = total_efe / batch_count if batch_count > 0 else float('inf')
            try:
                avg_ppl = math.exp(avg_ce)
            except OverflowError:
                avg_ppl = float('inf')
                
            print(f"\n--- TRIAL COMPLETE ---")
            print(f"TOTAL CL (Avg): {avg_ce:.4f}")
            print(f"TOTAL PPL:      {avg_ppl:.2f}")
            print(f"TOTAL EFE:      {avg_efe:.4f}")
            
            # 6. SAVE TO PROFESSIONAL CSV
            metrics = {
                'loss_cl': avg_ce,
                'ppl': avg_ppl,
                'efe': avg_efe,
                'actual_n_embed': n_embed
            }
            save_to_csv(trial_id, p, metrics)
            
            results.append(avg_ce)
            
            del model

        except Exception as e:
            print(f"!!! CRASH IN TRIAL {trial_id} !!!")
            print(str(e))
            results.append(float('inf'))
            
        finally:
            # Restore output and clean up
            logger.close()
            sys.stdout = original_stdout
            clean_memory()
            
    return np.array(results).reshape(-1, 1)