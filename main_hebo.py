import os
import uuid
import sys
from concurrent import futures
import nevergrad as ng
import numpy as np
from config import Config as config
from trainer_wrapper import train_evaluate_model

def _get_params(args, kwargs):
    if len(args) > 0:
        if hasattr(args[0], "value"): return dict(args[0].value)
        if isinstance(args[0], dict): return dict(args[0])
    return dict(kwargs)

def parallel_func_phase1(*args, **kwargs):
    params = _get_params(args, kwargs)
    params["n_embed"] = int(params.get("n_heads", 1)) * int(params.get("embed_mult", 1))
    arr = train_evaluate_model(params, objective="efe")
    return float(arr[0][0])

def run_phase(optimizer, objective_name, fixed_params=None, patience=5, min_delta=0.5):
    """
    patience: How many trials to wait without significant improvement.
    min_delta: The minimum change to be considered 'significant'.
    """
    best_loss = float("inf")
    best_params = None
    no_improve_counter = 0
    
    print(f"\n--- Starting {objective_name.upper()} Phase (Budget: {optimizer.budget}) ---", flush=True)
    
    for i in range(optimizer.budget):
        cand = optimizer.ask()
        merged = {**fixed_params, **cand.value} if fixed_params else cand.value
        
        arr = train_evaluate_model(merged, objective=objective_name)
        loss = float(arr[0][0])
        optimizer.tell(cand, loss)
        
        # Logic to check if the change is significant
        diff = best_loss - loss
        
        if diff > min_delta:
            print(f"  >>> Significant Improvement! Change: {diff:.4f}")
            best_loss = loss
            best_params = merged
            no_improve_counter = 0 # Reset counter
        else:
            no_improve_counter += 1
            print(f"  [Stagnation] No significant change ({diff:.4f}). Count: {no_improve_counter}/{patience}")

        if no_improve_counter >= patience:
            print(f"!!! Early Stopping Phase: Improvement less than {min_delta} for {patience} trials.")
            break
            
    return best_loss, best_params

def run_advanced(p1_budget=20, p2_budget=20, num_workers=2):
    print("\n" + "="*60 + "\n      🚀 EFE-PRIORITY WITH EARLY STOPPING\n" + "="*60, flush=True)

    # STEP 1: Parallel Exploration (Standard Nevergrad minimize doesn't easily early stop, 
    # but the budget is usually small enough here)
    print(f"\n[STEP 1/4] PARALLEL EFE SEARCH (Workers: {num_workers})")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=p1_budget, num_workers=num_workers)
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        rec1 = opt1.minimize(parallel_func_phase1, executor=executor, batch_mode=False)
    
    current_best = rec1.value
    current_best["n_embed"] = int(current_best["n_heads"]) * int(current_best["embed_mult"])
    fixed_arch = {k: current_best[k] for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter", "optim_type", "act_fx", "n_embed"]}

    # STEP 2: Portfolio with Early Stopping
    # We set min_delta to 0.5. If EFE moves from -234.34 to -234.29 (diff 0.05), it triggers the counter.
    print(f"\n[STEP 2/4] EFE REFINEMENT")
    opt2 = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase2_space(current_best), budget=p2_budget)
    _, current_best = run_phase(opt2, "combined", fixed_params=fixed_arch, patience=4, min_delta=0.5)

    # STEP 3: Chaining with Early Stopping
    print(f"\n[STEP 3/4] EFE CHAINING")
    ChainOpt = ng.optimizers.Chaining([ng.optimizers.LHSSearch, ng.optimizers.DE], [int(p2_budget*0.2)])
    opt3 = ChainOpt(parametrization=phase2_space(current_best), budget=p2_budget)
    _, current_best = run_phase(opt3, "combined", fixed_params=fixed_arch, patience=3, min_delta=1.0)

    # STEP 4: Pareto (Usually keep full budget to map the front)
    print(f"\n[STEP 4/4] FINAL PARETO FRONT")
    opt4 = ng.optimizers.DE(parametrization=phase2_space(current_best), budget=p2_budget)
    for i in range(p2_budget):
        cand = opt4.ask()
        merged = {**fixed_arch, **cand.value}
        ce = float(train_evaluate_model(merged, objective="ce")[0][0])
        efe = float(train_evaluate_model(merged, objective="efe")[0][0])
        opt4.tell(cand, [abs(efe), ce])

    print("\n" + "="*60 + "\n      ✅ PIPELINE COMPLETE\n" + "="*60, flush=True)

# Note: phase1_space and phase2_space remain the same as your previous code.