"""Nevergrad HPO entrypoint - DEEP SEARCH VERSION."""
import os
import math
import uuid
import sys
import time
from concurrent import futures
import nevergrad as ng
from config import Config as config
from trainer_wrapper import train_evaluate_model

# ==============================================================================
# 1. WORKER FUNCTIONS
# ==============================================================================

def parallel_func_phase1(**x):
    """Deep EFE Exploration Worker (Returns Absolute EFE)."""
    params = dict(x)
    worker_id = uuid.uuid4().hex[:4]
    
    # Ensure Integer Types
    for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter"]:
        if k in params: params[k] = int(params[k])
    if "n_heads" in params and "embed_mult" in params:
        params["n_embed"] = int(params["n_heads"]) * int(params["embed_mult"])

    print(f"--- [Worker {worker_id}] Starting EFE Trial ---", flush=True)
    try:
        arr = train_evaluate_model(params, objective="efe")
        val = float(arr[0][0]) 
    except Exception as e:
        print(f"[Worker {worker_id}] CRASHED: {e}", flush=True)
        return 1e9 

    if val > 10000:
        print(f"--- [Worker {worker_id}] EFE EXPLOSION ({val:.1f}). Skipping. ---", flush=True)
        return 1e9 

    return val

def run_deep_phase(optimizer, objective_name, fixed_params=None, budget_limit=16):
    """
    Deep Runner: Runs EVERY trial in the budget. No early stopping.
    """
    best_loss = float("inf")
    best_params = None
    
    print(f"\n[{objective_name.upper()}] Starting DEEP SEARCH ({budget_limit} trials)...", flush=True)

    for iteration in range(budget_limit):
        candidate = optimizer.ask()
        x_dict = candidate.value
        full_params = {**fixed_params, **x_dict} if fixed_params else x_dict.copy()
        
        # Type enforcement
        for k in ["n_heads", "batch_size", "seq_len", "tau_m", "n_iter"]:
            if k in full_params: full_params[k] = int(full_params[k])
        if "n_heads" in full_params and "embed_mult" in full_params:
            full_params["n_embed"] = int(full_params["n_heads"]) * int(full_params["embed_mult"])

        print(f"   > Combination {iteration+1}/{budget_limit}: Testing...", flush=True)
        
        try:
            arr = train_evaluate_model(full_params, objective=objective_name)
            loss_val = float(arr[0][0])
        except:
            loss_val = 1e9

        optimizer.tell(candidate, loss_val)

        if loss_val < best_loss:
            print(f"     >>> FOUND BETTER {objective_name.upper()}: {loss_val:.4f}", flush=True)
            best_loss = loss_val
            best_params = full_params
            
    return best_loss, best_params

# ==============================================================================
# 2. EXPANDED SEARCH SPACES
# ==============================================================================

def phase1_space():
    return ng.p.Dict(
        n_heads    = ng.p.Choice([2, 4, 8, 12]), # Added 12 for more combinations
        embed_mult = ng.p.Choice([8, 16, 24, 32]), # Added 32
        batch_size = ng.p.Choice([16, 32, 64]), 
        seq_len    = ng.p.Choice([16, 32, 64]), 
        eta=ng.p.Log(lower=1e-7, upper=1e-4),
        tau_m=ng.p.Scalar(lower=5, upper=25).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=10).set_integer_casting(),
        wub=ng.p.Scalar(lower=0.01, upper=0.06),
        wlb=ng.p.Scalar(lower=-0.06, upper=-0.01),
        optim_type=ng.p.Choice(["adam", "sgd", "adamw"]),
        act_fx=ng.p.Choice(["identity", "relu", "gelu"]),
    )

def phase2_deep_space(best):
    """Much wider range for fine-tuning to find the true best result."""
    eta_best = float(best.get("eta", 1e-5))
    return ng.p.Dict(
        eta=ng.p.Log(lower=eta_best * 0.2, upper=min(eta_best * 5.0, 5e-4)), 
        wub=ng.p.Scalar(lower=0.0001, upper=0.1),
        wlb=ng.p.Scalar(lower=-0.1, upper=-0.0001),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.3),
    )

# ==============================================================================
# 3. THE DEEP SEARCH PIPELINE
# ==============================================================================

def run_deep_pipeline(p1_budget=40, deep_budget=16, num_workers=4):
    print("\n" + "!"*60, flush=True)
    print("STARTING DEEP HPO SEARCH - NO EARLY EXIT", flush=True)
    print("!"*60, flush=True)

    # --- STEP 1: CMA-ES (Wide Parallel Search) ---
    print(f"\n[STEP 1/4] CMA-ES: EXPLORING {p1_budget} ARCHITECTURES", flush=True)
    opt_cma = ng.optimizers.CMA(parametrization=phase1_space(), budget=p1_budget, num_workers=num_workers)
    
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        rec_cma = opt_cma.minimize(parallel_func_phase1, executor=executor, batch_mode=False)
    
    current_best_params = rec_cma.value
    fixed_arch = {k: current_best_params[k] for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter", "optim_type", "act_fx"]}

    # --- STEP 2: CHAINING (Deep EFE Refinement) ---
    print(f"\n[STEP 2/4] CHAINING: SQUEEZING EFE (Forcing {deep_budget} trials)", flush=True)
    opt_chain = ng.optimizers.Chaining([ng.optimizers.LHSSearch, ng.optimizers.DE], [deep_budget//2])
    opt_chain = opt_chain(parametrization=phase2_deep_space(current_best_params), budget=deep_budget)
    
    _, current_best_params = run_deep_phase(opt_chain, "efe", fixed_params=fixed_arch, budget_limit=deep_budget)

    # --- STEP 3: PORTFOLIO (Deep CE Optimization) ---
    print(f"\n[STEP 3/4] PORTFOLIO: MAXIMIZING ACCURACY (Forcing {deep_budget} trials)", flush=True)
    opt_port = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase2_deep_space(current_best_params), budget=deep_budget)
    
    _, current_best_params = run_deep_phase(opt_port, "ce", fixed_params=fixed_arch, budget_limit=deep_budget)

    # --- STEP 4: DE (Final CE Polish) ---
    print(f"\n[STEP 4/4] DIFFERENTIAL EVOLUTION: FINAL POLISH", flush=True)
    opt_de = ng.optimizers.DE(parametrization=phase2_deep_space(current_best_params), budget=deep_budget)
    
    _, current_best_params = run_deep_phase(opt_de, "ce", fixed_params=fixed_arch, budget_limit=deep_budget)

    print("\n" + "="*60, flush=True)
    print(f"DEEP SEARCH COMPLETE. BEST COMBINATION FOUND ABOVE.", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    # Increased budgets for a "real" run
    run_deep_pipeline(p1_budget=40, deep_budget=16, num_workers=4)