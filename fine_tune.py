import alogos as al
import jax.numpy as jnp
from jax import random, clear_caches
import numpy as np
import sys
import gc
import os
import traceback
from config import Config as config

# Environment setup
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

def log_message(message, end="\n"):
    print(message, end=end, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + end)

with open(LOG_FILE, "w") as f:
    f.write("=== New Search Session ===\n")

# Fixed Constants from Config
FIXED_BS = config.batch_size
FIXED_BLOCK = config.seq_len
FIXED_VOCAB = config.vocab_size

# Grammar definition (Batch Size removed, wlb/wub restored)
bnf_text = """
<hparams> ::= <e64> "," <layers> "," <eta> "," <act> "," <bounds>
            | <e128> "," <layers> "," <eta> "," <act> "," <bounds>
            | <e256> "," <layers> "," <eta> "," <act> "," <bounds>

<e64>    ::= "n_embed=64,n_heads=4" | "n_embed=64,n_heads=8"
<e128>   ::= "n_embed=128,n_heads=4" | "n_embed=128,n_heads=8"
<e256>   ::= "n_embed=256,n_heads=4" | "n_embed=256,n_heads=8"

<layers> ::= "n_layers=2" | "n_layers=4" | "n_layers=6"
<eta>    ::= "eta=0.01" | "eta=0.005" | "eta=0.001"
<act>    ::= "act_fx=identity" | "act_fx=lrelu" | "act_fx=tanh"
<bounds> ::= <wlb> "," <wub>
<wlb>    ::= "wlb=-0.10" | "wlb=-0.05" | "wlb=-0.01"
<wub>    ::= "wub=0.01" | "wub=0.05" | "wub=0.10"
"""

# Initialize data loader once outside the objective function
data_loader = DataLoader(seq_len=FIXED_BLOCK, batch_size=FIXED_BS)
train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

def objective_function(phenotype_string):
    clean_string = phenotype_string.replace('"', '').replace(' ', '')
    log_message(f"\n[Testing Config]: {clean_string}")
    
    model = None
    val_ce = None
    
    try:
        # Parse params from phenotype string
        params = {p.split('=')[0]: p.split('=')[1] for p in clean_string.split(',')}
        dkey = random.PRNGKey(42)
        
        # Initialize model with FIXED_BS
        model = NGCTransformer(
            dkey,  
            batch_size=FIXED_BS, 
            seq_len=FIXED_BLOCK, 
            n_embed=int(params['n_embed']),
            vocab_size=FIXED_VOCAB, 
            n_layers=int(params['n_layers']), 
            n_heads=int(params['n_heads']),
            T=config.n_iter, 
            dt=1.0, 
            tau_m=config.tau_m, 
            act_fx=params['act_fx'], 
            eta=float(params['eta']),
            dropout_rate=float(config.dropout_rate), 
            pos_learnable=config.pos_learnable,
            optim_type=config.optim_type,
            wub=float(params['wub']), 
            wlb=float(params['wlb']), 
            exp_dir="exp",
            loadDir=None,
            model_name="ngc_transformer"
        )

        log_message(f"    {'Iter':<5} | {'Batch CE':<12} | {'Batch PPL':<12}")
        log_message(f"    {'-' * 35}")

        train_iter = iter(train_loader)
        for i in range(10):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Ensure data matches FIXED_BS
            inputs = batch[0][1][:FIXED_BS, :FIXED_BLOCK]
            targets = batch[1][1][:FIXED_BS, :FIXED_BLOCK]
            targets_flat = jnp.eye(FIXED_VOCAB)[targets].reshape(-1, FIXED_VOCAB)
            
            # Forward and Update
            preds = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            
            # Handle NGCLearn tuple output
            if isinstance(preds, tuple):
                preds = preds[0]
            
            it_ce = float(measure_CatNLL(preds, targets_flat))
            it_ppl = float(np.exp(it_ce))
            
            log_message(f"    {i+1:<5} | {it_ce:<12.4f} | {it_ppl:<12.2f}")

        # Final Validation
        val_ce, _ = eval_model(model, valid_loader, FIXED_VOCAB)
        return float(val_ce)

    except Exception as e:
        error_trace = traceback.format_exc()
        log_message(f"\n    [!] ERROR: {e}")
        log_message(f"    DEBUG TRACE:\n{error_trace}")
        return 5000.0  # Penalty for failed configuration

    finally:
        # Aggressive memory cleanup for JAX
        if model is not None:
            del model
        clear_caches()
        gc.collect()

def main():
    grammar = al.Grammar(bnf_text=bnf_text)
    ea = al.EvolutionaryAlgorithm(
        grammar, 
        objective_function, 
        'min', 
        population_size=10, 
        max_generations=5
    )
    
    log_message("\n" + "="*45)
    log_message("STARTING SEARCH (FIXED BATCH SIZE)")
    log_message("="*45)
    
    best_ind = ea.run()
    log_message(f"\nBEST CONFIG: {best_ind.phenotype}")

if __name__ == "__main__":
    main()