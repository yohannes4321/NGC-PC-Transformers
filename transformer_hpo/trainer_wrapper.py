# filename: trainer_wrapper.py
import time
import os
import math
import gc
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Ensure JAX memory settings
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6" 

from experiment_logger import save_to_csv, DualLogger, LOG_DIR
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Config as config
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader

def clean_memory():
    gc.collect()
    jax.clear_caches()

def estimate_trial_memory_bytes(batch_size, seq_len, n_embed, n_heads, n_layers, vocab_size):
    # Rough estimate of peak arrays (float32 = 4 bytes)
    B, S, D, H, L, V = batch_size, seq_len, n_embed, n_heads, n_layers, vocab_size
    base = B * S * D
    attn_scores = B * H * S * S
    per_layer_elems = (3 * base) + attn_scores + (3 * base)  # Q,K,V + scores + intermediates
    projection_target = B * S * V
    total_elems = per_layer_elems * L + projection_target + base
    return total_elems * 4

def train_evaluate_model(params):
    if isinstance(params, dict):
        params_df = pd.DataFrame([params])
    else:
        params_df = params

    results = []
    
    for i in range(len(params_df)):
        existing_trials = len(os.listdir(LOG_DIR))
        trial_id = existing_trials
        log_path = os.path.join(LOG_DIR, f"trial_{trial_id}.txt")
        
        original_stdout = sys.stdout
        logger = DualLogger(log_path)
        sys.stdout = logger
        
        try:
            # 2. PARSE PARAMETERS
            p = params_df.iloc[i]
            n_heads = int(p['n_heads'])
            raw_n_embed = int(p['n_embed'])
            # Enforce head divisibility constraint
            n_embed = (raw_n_embed // n_heads) * n_heads
            if n_embed == 0: n_embed = n_heads
            
            curr_batch_size = int(p['batch_size'])
            block_size = int(p['block_size'])
            
            print(f"==========================================")
            print(f"STARTING TRIAL {trial_id}")
            # Print full parameter set for visibility
            keys_to_show = [
                'n_embed','n_heads','n_layers','block_size','batch_size',
                'T','eta','dropout','wlb','wub','tau_m','act_fx'
            ]
            summary_items = []
            for k in keys_to_show:
                if k in p.index:
                    summary_items.append(f"{k}={p[k]}")
            summary_items.append(f"n_embed_adjusted={n_embed}")
            print("Params: " + ", ".join(summary_items))
            print(f"==========================================")
            
            clean_memory()
            # Pre-check estimated memory to avoid GPU OOM
            est_bytes = estimate_trial_memory_bytes(curr_batch_size, block_size, n_embed, n_heads, int(p['n_layers']), config.vocab_size)
            cap_bytes = int(os.environ.get('HPO_MEMORY_CAP_BYTES', str(800 * 1024 * 1024)))
            if est_bytes > cap_bytes:
                print(f"Skipping trial {trial_id}: estimated memory {est_bytes/1e6:.1f} MB exceeds cap {cap_bytes/1e6:.1f} MB")
                results.append(float('inf'))
                continue
            
            # 3. INITIALIZE DATA & MODEL
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
                model_name=f"NEVERGRAD_{trial_id}"
            )
            
            total_ce = 0.0
            total_efe = 0.0
            batch_count = 0
            max_batches = 50 
            
            nan_found = False

            # 4. TRAINING LOOP
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_batches: break
                
                inputs = batch[0][1] 
                targets = batch[1][1]
                
                # --- CRITICAL FIX: Skip partial/remainder batches ---
                if inputs.shape[0] != curr_batch_size:
                    continue

                # One-hot encoding
                targets_onehot = jax.nn.one_hot(targets, config.vocab_size)
                
                # Reshape with dynamic dimensions based on actual batch size
                targets_flat = targets_onehot.reshape(-1, config.vocab_size)
                
                # Forward pass
                yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
                y_pred = yMu_inf.reshape(-1, config.vocab_size)
                
                batch_ce = float(measure_CatNLL(y_pred, targets_flat).mean())
                batch_efe = float(_EFE)
                batch_ppl = math.exp(batch_ce) if batch_ce < 100 else float('inf')

                

                total_ce += batch_ce
                total_efe += batch_efe
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}: CL={batch_ce:.4f}, PPL={batch_ppl:.4f}, EFE={batch_efe:.4f}")

            # 5. CALCULATE METRICS
            if nan_found:
                avg_ce = float('inf')
                avg_efe = float('inf')
                avg_ppl = float('inf')
            else:
                avg_ce = total_ce / batch_count if batch_count > 0 else float('inf')
                avg_efe = total_efe / batch_count if batch_count > 0 else float('inf')
                avg_ppl = math.exp(avg_ce) if avg_ce < 100 else float('inf')
            
            metrics = {'loss_cl': avg_ce, 'ppl': avg_ppl, 'efe': avg_efe, 'actual_n_embed': n_embed}
            save_to_csv(trial_id, p, metrics)
            results.append(avg_ce)
            del model

        except Exception as e:
            print(f"!!! CRASH IN TRIAL {trial_id} !!!")
            msg = str(e)
            if 'RESOURCE_EXHAUSTED' in msg or 'Out of memory' in msg:
                print("Detected GPU OOM; marking trial as inf and proceeding.")
            else:
                print(f"Error: {msg}")
            results.append(float('inf'))
            
        finally:
            logger.close()
            sys.stdout = original_stdout
            clean_memory()
            
    return np.array(results).reshape(-1, 1)