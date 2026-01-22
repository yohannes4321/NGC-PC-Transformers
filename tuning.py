"""Nevergrad HPO entrypoint - SMART CHAINING & FAIL-FAST VERSION."""
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
        # Wrapper returns np.array([[abs_loss]])
        arr = train_evaluate_model(params, objective="efe")
        val = float(arr[0][0]) 
    except Exception as e:
        print(f"[Worker {worker_id}] CRASHED: {e}", flush=True)
        return 1e9 

    # FAIL-FAST CHECK: Explosion (Value too far from 0)
    if val > 10000:
        print(f"--- [Worker {worker_id}] EFE EXPLOSION (Abs Loss: {val:.1f}). STOPPING. ---", flush=True)
        return 1e9 

    # Note: Wrapper already printed EFE/CE/PPL details
    return val

def run_smart_phase(optimizer, objective_name, fixed_params=None, max_no_change=3, budget_limit=4):
    """
    Smart Runner: 
    1. Runs for a maximum of `budget_limit` (e.g., 4 times).
    2. Stops early if no improvement is seen after `max_no_change`.
    """
    best_loss = float("inf")
    best_params = None
    no_change_counter = 0
    
    print(f"\n[{objective_name.upper()}] Starting Smart Loop (Max {budget_limit} trials)...", flush=True)

    for iteration in range(budget_limit):
        candidate = optimizer.ask()
        x_dict = candidate.value
        full_params = {**fixed_params, **x_dict} if fixed_params else x_dict.copy()
        
        # Type enforcement
        for k in ["n_heads", "batch_size", "seq_len", "tau_m", "n_iter"]:
            if k in full_params: full_params[k] = int(full_params[k])
        if "n_heads" in full_params and "embed_mult" in full_params:
            full_params["n_embed"] = int(full_params["n_heads"]) * int(full_params["embed_mult"])

        print(f"   > Trial {iteration+1}: Running...", flush=True)
        
        try:
            arr = train_evaluate_model(full_params, objective=objective_name)
            loss_val = float(arr[0][0])
        except:
            loss_val = 1e9

        optimizer.tell(candidate, loss_val)

        # LOGIC: Check for Improvement
        if loss_val < best_loss:
            print(f"     >>> NEW BEST {objective_name.upper()}: {loss_val:.4f} (Improved!)", flush=True)
            best_loss = loss_val
            best_params = full_params
            no_change_counter = 0 
        else:
            no_change_counter += 1
            print(f"     (No improvement. Streak: {no_change_counter}/{max_no_change})", flush=True)

        # LOGIC: Smart Break (Fail-Fast)
        if no_change_counter >= max_no_change:
            print(f"   [STOPPING] No improvement for {max_no_change} trials. Moving to next step.", flush=True)
            break
            
    return best_loss, best_params

# ==============================================================================
# 2. SEARCH SPACES
# ==============================================================================

def phase1_space():
    return ng.p.Dict(
        n_heads    = ng.p.Choice([2, 4, 8]),
        embed_mult = ng.p.Choice([8, 16, 24]),
        batch_size = ng.p.Choice([16, 32]), 
        seq_len    = ng.p.Choice([16, 32]), 
        eta=ng.p.Log(lower=1e-7, upper=5e-5),
        tau_m=ng.p.Scalar(lower=5, upper=15).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=5).set_integer_casting(),
        wub=ng.p.Scalar(lower=0.01, upper=0.04),
        wlb=ng.p.Scalar(lower=-0.04, upper=-0.01),
        optim_type=ng.p.Choice(["adam", "sgd"]),
        act_fx=ng.p.Choice(["identity", "relu"]),
    )

def phase2_space(best):
    eta_best = float(best.get("eta", 1e-5))
    return ng.p.Dict(
        eta=ng.p.Log(lower=eta_best * 0.8, upper=min(eta_best * 1.2, 1e-4)), 
        wub=ng.p.Scalar(lower=0.001, upper=0.04),
        wlb=ng.p.Scalar(lower=-0.04, upper=-0.001),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.1),
    )

# ==============================================================================
# 3. THE "RELAY RACE" PIPELINE
# ==============================================================================

def run_advanced(p1_budget=20, p2_budget=20, num_workers=4):
    print("\n" + "="*60, flush=True)
    print("STARTING OPTIMIZATION RELAY RACE", flush=True)
    print("Plan: CMA (EFE) -> Chaining (EFE) -> Portfolio (CE) -> DE (CE)", flush=True)
    print("Objective: Get EFE closer to 0 (minimize absolute value)", flush=True)
    print("="*60, flush=True)

    # --- STEP 1: CMA (The EFE Foundation) ---
    print(f"\n[STEP 1/4] CMA-ES: FINDING EFE BASELINE (Parallel)", flush=True)
    opt_cma = ng.optimizers.CMA(parametrization=phase1_space(), budget=p1_budget, num_workers=num_workers)
    
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        rec_cma = opt_cma.minimize(parallel_func_phase1, executor=executor, batch_mode=False)
    
    current_best_params = rec_cma.value
    
    # INFER BEST LOSS
    print("... verifying baseline EFE ...", flush=True)
    try:
        baseline_efe = parallel_func_phase1(**current_best_params)
    except:
        baseline_efe = 10000.0

    print(f">>> STEP 1 FINISHED. Best EFE (Abs) so far: {baseline_efe:.4f}", flush=True)

    if baseline_efe > 10000:
        print("!!! FAILURE: EFE is too far from 0 (> 10,000). System unstable. Exiting.", flush=True)
        return

    # Lock the Skeleton
    fixed_arch = {k: current_best_params[k] for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter", "optim_type", "act_fx"]}
    
    # --- STEP 2: CHAINING (EFE Refinement) ---
    print(f"\n[STEP 2/4] CHAINING: IMPROVING EFE (Max 4 Trials)", flush=True)
    
    opt_chain = ng.optimizers.Chaining([ng.optimizers.LHSSearch, ng.optimizers.DE], [1])
    opt_chain = opt_chain(parametrization=phase2_space(current_best_params), budget=4)
    
    loss_efe, params_efe = run_smart_phase(opt_chain, "efe", fixed_params=fixed_arch, max_no_change=3, budget_limit=4)
    
    if loss_efe < baseline_efe:
        print(f">>> SUCCESS: Chaining improved EFE from {baseline_efe:.4f} to {loss_efe:.4f}", flush=True)
        current_best_params = params_efe
    else:
        print(f">>> NOTE: Chaining did not improve EFE. Keeping CMA result.", flush=True)

    # --- STEP 3: PORTFOLIO (CE Optimization) ---
    print(f"\n[STEP 3/4] PORTFOLIO: LOWERING CE (ACCURACY) (Max 4 Trials)", flush=True)
    
    opt_port = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase2_space(current_best_params), budget=4)
    loss_ce, params_ce = run_smart_phase(opt_port, "ce", fixed_params=fixed_arch, max_no_change=3, budget_limit=4)
    
    # Safety Check
    if loss_ce < 1000: 
        current_best_params = params_ce
    else:
        # If CE exploded, stick with previous params
        loss_ce = 1000

    # --- STEP 4: DE (Final CE Polish) ---
    print(f"\n[STEP 4/4] DE: FINAL CE POLISH (Max 4 Trials)", flush=True)
    opt_de = ng.optimizers.DE(parametrization=phase2_space(current_best_params), budget=4)
    loss_final, params_final = run_smart_phase(opt_de, "ce", fixed_params=fixed_arch, max_no_change=3, budget_limit=4)
    
    if loss_final < loss_ce:
        current_best_params = params_final

    # --- FINAL REPORT ---
    print("\n" + "="*60, flush=True)
    print(f"OPTIMIZATION COMPLETE.", flush=True)
    print(f"Final Architecture: {fixed_arch}", flush=True)
    print(f"Best Fine-Tuned Params: {current_best_params}", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    p1 = getattr(config, "p1_budget", 20)
    # Budget for steps 2,3,4 is hardcoded to 4 inside run_advanced to match your request
    run_advanced(p1, p2_budget=4, num_workers=4)