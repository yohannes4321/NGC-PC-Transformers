import alogos as al
import jax
import jax.numpy as jnp
from jax import random, clear_caches
import numpy as np
import sys
import gc
import os
import time
import traceback
from config import Config as config

# --- 1. Environment & Setup ---
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

try:
    from model import NGCTransformer
    from ngclearn.utils.metric_utils import measure_CatNLL
    from data_preprocess.data_loader import DataLoader
    from eval import eval_model
except ImportError as e:
    print(f"Import Error: {e}. Check project path.")
    sys.exit(1)

LOG_FILE = "search_progress.log"
RESULT_FILE = "tuning_results.txt"

def log_message(message, end="\n"):
    print(message, end=end, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + end)


def clean_memory():
    """Aggressively free JAX/device and host allocations."""
    try:
        clear_caches()
        jax.block_until_ready(jnp.array(0))
    except Exception:
        pass
    gc.collect()

with open(LOG_FILE, "w") as f:
    f.write("=== New Search Session ===\n")

# --- 2. Grammar Guided Search Space (CLEANED) ---
# CRITICAL FIX: No comments (#) allowed inside this string!
bnf_text = """
<hparams>     ::= <arch_pair> "," <block_conf> "," <depth_conf> "," <steps> "," <learn_rate> "," <drop> "," <bounds>
<arch_pair>   ::= "n_embed=64,n_heads=4" | "n_embed=128,n_heads=4" | "n_embed=128,n_heads=8" | "n_embed=256,n_heads=8"
<block_conf>  ::= "block_size=64" | "block_size=128" | "block_size=256"
<depth_conf>  ::= "n_layers=2" | "n_layers=4" | "n_layers=6"
<steps>       ::= "T=5" | "T=10" | "T=15" | "T=20"
<learn_rate>  ::= "eta=0.001" | "eta=0.0005" | "eta=0.0001"
<drop>        ::= "dropout=0.1" | "dropout=0.2" | "dropout=0.3"
<bounds>      ::= "wlb=-0.2,wub=0.2" | "wlb=-0.1,wub=0.1" | "wlb=-0.5,wub=0.5" | "wlb=-0.02,wub=0.02"
"""

def get_dynamic_batch_size(n_embed, block_size):
    """
    Calculates batch size to maintain constant memory usage.
    """
    complexity_score = n_embed * block_size
    # More conservative sizing to avoid OOM seen in prior runs
    if complexity_score >= 32000:
        return 8
    elif complexity_score >= 20000:
        return 12
    elif complexity_score >= 12000:
        return 16
    elif complexity_score >= 6000:
        return 24
    else:
        return 32

# --- 3. The Objective Function (The "Trial") ---
def objective_function(phenotype_string):
    start_time = time.time()
    
    # 1. CLEANING THE STRING
    # Removes quotes and spaces to handle "n_embed=64" format
    clean_string = phenotype_string.replace('"', '').replace(' ', '').replace('\n', '')
    
    try:
        # 2. ROBUST PARSING
        # Splits by comma and creates key-value pairs
        params = {}
        for item in clean_string.split(','):
            if '=' in item:
                k, v = item.split('=')
                params[k] = v
            else:
                # If we encounter garbage (like comments), we ignore or fail safely
                continue
                
        # Extract Key Params
        n_embed = int(params['n_embed'])
        n_heads = int(params['n_heads'])
        seq_len = int(params['block_size'])
        n_layers = int(params['n_layers'])
        T = int(params['T'])
        eta = float(params['eta'])
        dropout = float(params['dropout'])
        wlb = float(params['wlb'])
        wub = float(params['wub'])

    except Exception as e:
        # If parsing fails, print exactly why so we don't get silent "0.0 runtime"
        log_message(f"[!] Parsing Failed: {clean_string}")
        log_message(f"    Error: {e}")
        return 1e9 # Penalty

    if n_embed % n_heads != 0:
        log_message(f"[!] Invalid Config: n_embed ({n_embed}) not divisible by n_heads ({n_heads}). Skipping trial.")
        return 1e9
    if wlb >= wub:
        log_message(f"[!] Invalid Bounds: wlb ({wlb}) must be < wub ({wub}). Skipping trial.")
        return 1e9
    # Clip bounds to a safe envelope to prevent extreme init leading to NaNs
    safe_wlb = max(wlb, -0.25)
    safe_wub = min(wub, 0.25)
    if (safe_wlb != wlb) or (safe_wub != wub):
        log_message(f"[i] Bounds clipped to safe range: wlb={safe_wlb}, wub={safe_wub}")
    wlb, wub = safe_wlb, safe_wub

    log_message(f"\n" + "="*60)
    log_message(f"[Trial Start] Config: {clean_string}")
    
    curr_batch_size = get_dynamic_batch_size(n_embed, seq_len)
    log_message(f" >> Dynamic Sizing: Seq_Len={seq_len} | Embed={n_embed} -> Batch_Size={curr_batch_size}")

    # Sync global config so downstream components relying on it stay consistent with tuned params
    config.seq_len = seq_len
    config.batch_size = curr_batch_size
    config.n_embed = n_embed
    config.n_heads = n_heads

    model = None
    try:
        # Reload Data with new Block Size
        loader = DataLoader(seq_len=seq_len, batch_size=curr_batch_size)
        train_loader, valid_loader, _ = loader.load_and_prepare_data()
        clean_memory()
        
        # Initialize Model
        dkey = random.PRNGKey(int(time.time()))
        model = NGCTransformer(
            dkey, 
            batch_size=curr_batch_size, 
            seq_len=seq_len, 
            n_embed=n_embed,
            vocab_size=config.vocab_size, 
            n_layers=n_layers, 
            n_heads=n_heads,
            T=T, 
            dt=1.0, 
            tau_m=config.tau_m, 
            act_fx=config.act_fx, 
            eta=eta,
            dropout_rate=dropout, 
            pos_learnable=config.pos_learnable,
            optim_type=config.optim_type,
            wub=wub, 
            wlb=wlb, 
            exp_dir="exp",
            model_name="ngc_evo_temp"
        )

        # Training Loop
        log_message(f"\n {'Batch':<6} | {'Energy (EFE)':<14} | {'PPL':<12} | {'CL (Loss)':<12}")
        log_message(f" {'-'*52}")

        total_efe = 0.0
        total_ppl = 0.0
        batch_count = 0
        max_batches_per_trial = 20 
        
        train_iter = iter(train_loader)
        
        for i in range(max_batches_per_trial):
            try:
                batch = next(train_iter)
            except StopIteration:
                break

            inputs = batch[0][1] 
            targets = batch[1][1]
            targets_onehot = jnp.eye(config.vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, config.vocab_size)
            
            # Forward + Update
            yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            y_pred = yMu_inf.reshape(-1, config.vocab_size)
            
            # NaN Check
            if jnp.isnan(_EFE) or jnp.isnan(y_pred).any():
                log_message(f" [!] Trial Killed: NaN detected at batch {i}")
                return 1e9 
                
            batch_ce = float(measure_CatNLL(y_pred, targets_flat).mean())
            batch_ppl = float(np.exp(batch_ce))
            batch_efe = float(_EFE)
            
            log_message(f" {i:<6} | {batch_efe:<14.4f} | {batch_ppl:<12.2f} | {batch_ce:<12.4f}")
            
            total_efe += batch_efe
            total_ppl += batch_ppl
            batch_count += 1
            clean_memory()

        # Calculate Averages
        avg_efe = total_efe / batch_count if batch_count > 0 else 0
        avg_ppl = total_ppl / batch_count if batch_count > 0 else 0
        eval_time = time.time() - start_time

        log_message(f"\n[Summary] Avg Energy: {avg_efe:.4f} | Avg PPL: {avg_ppl:.2f} | Time: {eval_time:.2f}s")

        # Validation (The Fitness Function)
        val_ce, _ = eval_model(model, valid_loader, config.vocab_size)
        log_message(f"[Result] Validation Score (CE): {val_ce:.4f}")
        
        clean_memory()
        return float(val_ce)

    except KeyboardInterrupt:
        clean_memory()
        return 1e9
    except Exception as e:
        log_message(f"\n[!] ERROR in Trial: {e}")
        error_trace = traceback.format_exc()
        # log_message(error_trace) # Uncomment if you need deep debugging
        clean_memory()
        return 1e9 
    finally:
        if model: del model
        clean_memory()
        try:
            del loader, train_loader, valid_loader, train_iter
        except Exception:
            pass
        clean_memory()

# --- 4. The Genetic Algorithm Execution ---
def main():
    # Load Grammar
    grammar = al.Grammar(bnf_text=bnf_text)
    
    # Define Evolutionary Algorithm
    ea = al.EvolutionaryAlgorithm(
        grammar, 
        objective_function, 
        'min', 
        population_size=10,   
        max_generations=5,    
        verbose=True
    )
    
    log_message("\n" + "#"*50)
    log_message("STARTING GRAMMAR-GUIDED EVOLUTION")
    log_message("#"*50)
    
    try:
        best_ind = ea.run()
        
        # --- SAVE RESULTS ---
        log_message("\n" + "#"*50)
        log_message("OPTIMIZATION COMPLETED")
        log_message("#"*50)
        
        report = f"""
Optimization Success.

Best Fitness (Validation CE): {best_ind.fitness:.5f}
Best Parameters Found:
{best_ind.phenotype}
"""
        print(report)
        with open(RESULT_FILE, "w") as f:
            f.write(report)
            
    except KeyboardInterrupt:
        print("\nOptimization stopped manually.")

if __name__ == "__main__":
    main()