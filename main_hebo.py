"""Nevergrad HPO entrypoint."""
import os
import math
import uuid
import sys
from concurrent import futures
import numpy as np
import nevergrad as ng
from config import Config as config
from trainer_wrapper import train_evaluate_model

# ==============================================================================
# 1. EVALUATION FUNCTIONS
# ==============================================================================

def parallel_func_phase1(**x):
    """Worker function for Phase 1. Flushes output to ensure visibility."""
    params = dict(x)
    worker_id = uuid.uuid4().hex[:4]
    # Ensure types
    for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter"]:
        if k in params: params[k] = int(params[k])
    if "n_heads" in params and "embed_mult" in params:
        params["n_embed"] = int(params["n_heads"]) * int(params["embed_mult"])

    print(f"--- [Worker {worker_id}] Starting EFE Trial ---", flush=True)
    arr = train_evaluate_model(params, objective="efe")
    val = float(arr[0][0])
    print(f"--- [Worker {worker_id}] Finished. Loss: {val:.4f} ---", flush=True)
    return val

def parallel_func_phase2(**x):
    """Worker function for Phase 2."""
    params = dict(x)
    worker_id = uuid.uuid4().hex[:4]
    for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter"]:
        if k in params: params[k] = int(params[k])
    
    print(f"--- [Worker {worker_id}] Starting CE Trial ---", flush=True)
    arr = train_evaluate_model(params, objective="ce")
    val = float(arr[0][0])
    print(f"--- [Worker {worker_id}] Finished. Loss: {val:.4f} ---", flush=True)
    return val

def constraint_embed_divisible(x):
    return int(x.get("n_embed", 0)) % int(x.get("n_heads", 1)) == 0

# ==============================================================================
# 2. SEARCH SPACES
# ==============================================================================

def phase1_space():
    return ng.p.Dict(
        n_heads    = ng.p.Choice([2, 4, 8]),
        embed_mult = ng.p.Choice([8, 16, 24]),
        batch_size = ng.p.Choice([16, 32, 64]),
        seq_len    = ng.p.Choice([16, 32, 48]),
        eta=ng.p.Log(lower=1e-7, upper=5e-5),
        tau_m=ng.p.Scalar(lower=5, upper=15).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=10).set_integer_casting(),
        wub=ng.p.Scalar(lower=0.01, upper=0.04),
        wlb=ng.p.Scalar(lower=-0.04, upper=-0.01),
        dropout_rate=ng.p.Constant(0.0),
        optim_type=ng.p.Choice(["adam", "sgd"]),
        act_fx=ng.p.Choice(["identity", "relu"]),
    )

def phase2_space(best):
    eta_best = float(best.get("eta", 1e-5))
    return ng.p.Dict(
        eta=ng.p.Log(lower=eta_best * 0.5, upper=min(eta_best * 2.0, 1e-4)),
        wub=ng.p.Scalar(lower=0.001, upper=0.04),
        wlb=ng.p.Scalar(lower=-0.04, upper=-0.001),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.2),
    )

# ==============================================================================
# 3. GENERIC RUNNER (For Sequential Steps)
# ==============================================================================

def run_phase(optimizer, objective_name, fixed_params=None):
    best_loss = float("inf")
    best_params = None
    
    for iteration in range(optimizer.budget):
        candidate = optimizer.ask()
        x_dict = candidate.value
        full_params = {**fixed_params, **x_dict} if fixed_params else x_dict.copy()
        
        # JAX Type Casting
        for k in ["n_heads", "batch_size", "seq_len", "tau_m", "n_iter"]:
            if k in full_params: full_params[k] = int(full_params[k])
        if "n_heads" in full_params and "embed_mult" in full_params:
            full_params["n_embed"] = int(full_params["n_heads"]) * int(full_params["embed_mult"])

        print(f"\n[{objective_name.upper()}] Iter {iteration}/{optimizer.budget} starting...", flush=True)
        arr = train_evaluate_model(full_params, objective=objective_name)
        loss_val = float(arr[0][0])
        
        optimizer.tell(candidate, loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = full_params
            print(f">>> NEW BEST {objective_name.upper()}: {best_loss:.4f}")
            
    return best_loss, best_params

# ==============================================================================
# 4. THE ADVANCED PIPELINE
# ==============================================================================

def run_advanced(p1_budget=30, p2_budget=40, num_workers=4):
    print("\n" + "="*50)
    print("STARTING ADVANCED OPTIMIZATION PIPELINE")
    print("="*50)

    # --- STEP 1: PARALLEL NGOpt (The Broad Sweep) ---
    print(f"\n[STEP 1/4] PARALLEL EXPLORATION (Workers: {num_workers})")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=p1_budget, num_workers=num_workers)
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        rec1 = opt1.minimize(parallel_func_phase1, executor=executor, batch_mode=False)
    
    current_best_params = rec1.value
    # Fix the winning architecture for next steps
    fixed_arch = {k: current_best_params[k] for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter", "optim_type", "act_fx"]}

    # --- STEP 2: PORTFOLIO (Refining Choices) ---
    print(f"\n[STEP 2/4] PORTFOLIO REFINEMENT (Sequential)")
    opt2 = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase2_space(current_best_params), budget=p2_budget)
    loss2, params2 = run_phase(opt2, "ce", fixed_params=fixed_arch)
    
    if loss2 < 1000: # Assuming 1000 is a failure threshold
        current_best_params = params2

    # --- STEP 3: CHAINING (LHS -> DE for Global Search) ---
    print(f"\n[STEP 3/4] CHAINING SEARCH (LHS then Differential Evolution)")
    ChainOpt = ng.optimizers.Chaining([ng.optimizers.LHSSearch, ng.optimizers.DE], [int(p2_budget*0.2)])
    opt3 = ChainOpt(parametrization=phase2_space(current_best_params), budget=p2_budget)
    loss3, params3 = run_phase(opt3, "ce", fixed_params=fixed_arch)

    if loss3 < loss2:
        current_best_params = params3

    # --- STEP 4: MULTI-OBJECTIVE (The Final Polish) ---
    print(f"\n[STEP 4/4] FINAL PARETO ANALYSIS")
    opt4 = ng.optimizers.DE(parametrization=phase2_space(current_best_params), budget=p2_budget)
    for i in range(p2_budget):
        cand = opt4.ask()
        merged = {**fixed_arch, **cand.value}
        ce = float(train_evaluate_model(merged, objective="ce")[0][0])
        efe = float(train_evaluate_model(merged, objective="efe")[0][0])
        print(f"MO Trial {i}: CE={ce:.4f}, EFE={efe:.4f}", flush=True)
        opt4.tell(cand, [ce, abs(efe)])

    print("\n" + "="*50)
    print("ADVANCED PIPELINE COMPLETE")
    print("="*50)

if __name__ == "__main__":
    p1 = getattr(config, "p1_budget", 20)
    p2 = getattr(config, "p2_budget", 20)
    run_advanced(p1, p2, num_workers=4)