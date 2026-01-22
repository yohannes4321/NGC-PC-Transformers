"""Nevergrad HPO - UNLIMITED DEEP SEARCH VERSION"""
import os
import uuid
from concurrent import futures
import nevergrad as ng
import numpy as np
from trainer_wrapper import train_evaluate_model

# ==============================================================================
# 1. THE UNLIMITED RUNNER
# ==============================================================================

def run_unlimited_phase(optimizer, objective_name, fixed_params=None, budget=20):
    """Runs every single trial. No stagnation checks. No early stopping."""
    best_loss = float("inf")
    best_params = None
    
    print(f"\n>>> STARTING {objective_name.upper()} UNLIMITED SEARCH ({budget} Trials)")
    
    for i in range(budget):
        candidate = optimizer.ask()
        x_dict = candidate.value
        full_params = {**fixed_params, **x_dict} if fixed_params else x_dict.copy()
        
        # Type Casting
        for k in ["n_heads", "batch_size", "seq_len", "tau_m", "n_iter"]:
            if k in full_params: full_params[k] = int(full_params[k])
        if "n_heads" in full_params and "embed_mult" in full_params:
            full_params["n_embed"] = int(full_params["n_heads"]) * int(full_params["embed_mult"])

        print(f"   [Trial {i+1}/{budget}] Testing combination...", flush=True)
        
        try:
            # We call the wrapper which handles the abs(EFE) or CE logic
            arr = train_evaluate_model(full_params, objective=objective_name)
            loss_val = float(arr[0][0])
        except Exception:
            loss_val = 1e9

        optimizer.tell(candidate, loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = full_params
            print(f"   !!! NEW BEST {objective_name.upper()}: {best_loss:.4f}", flush=True)
            
    return best_loss, best_params

# ==============================================================================
# 2. THE SEARCH SPACES (WIDER RANGE)
# ==============================================================================

def phase1_wide_space():
    return ng.p.Dict(
        n_heads    = ng.p.Choice([2, 4, 8, 12, 16]),
        embed_mult = ng.p.Choice([8, 16, 24, 32, 64]),
        batch_size = ng.p.Choice([16, 32, 64]), 
        seq_len    = ng.p.Choice([16, 32, 64, 128]), 
        eta        = ng.p.Log(lower=1e-8, upper=1e-3), # Much wider
        tau_m      = ng.p.Scalar(lower=5, upper=30).set_integer_casting(),
        n_iter     = ng.p.Scalar(lower=1, upper=15).set_integer_casting(),
        wub        = ng.p.Scalar(lower=0.001, upper=0.1),
        wlb        = ng.p.Scalar(lower=-0.1, upper=-0.001),
        optim_type = ng.p.Choice(["adam", "sgd", "adamw"]),
        act_fx     = ng.p.Choice(["identity", "relu", "gelu", "swish"]),
    )

def phase2_ultra_space(best_params):
    # Centered on p1 winner but with massive 10x exploration range
    eta_ref = float(best_params.get("eta", 1e-5))
    return ng.p.Dict(
        eta          = ng.p.Log(lower=eta_ref * 0.1, upper=min(eta_ref * 10.0, 1e-2)),
        wub          = ng.p.Scalar(lower=0.0001, upper=0.2),
        wlb          = ng.p.Scalar(lower=-0.2, upper=-0.0001),
        dropout_rate = ng.p.Scalar(lower=0.0, upper=0.5),
    )

# ==============================================================================
# 3. THE PIPELINE
# ==============================================================================

def run_full_pipeline(p1_budget=50, p2_budget=100):
    print("="*80)
    print("CRITICAL MISSION: MINIMIZE CE & STABILIZE EFE")
    print(f"TOTAL TRIALS PLANNED: {p1_budget + p2_budget}")
    print("="*80)

    # --- STEP 1: CMA-ES for EFE Foundation ---
    # We use parallel workers here because p1_budget is large
    print(f"\n[PHASE 1] FOUNDATION SEARCH (Parallel 4 Workers)")
    opt_p1 = ng.optimizers.CMA(parametrization=phase1_wide_space(), budget=p1_budget, num_workers=4)
    
    # We need a small helper for parallel phase 1
    def p1_worker(**x):
        arr = train_evaluate_model(x, objective="efe")
        return float(arr[0][0])

    with futures.ProcessPoolExecutor(max_workers=4) as executor:
        recommendation = opt_p1.minimize(p1_worker, executor=executor, batch_mode=False)
    
    best_p1_params = recommendation.value
    print(f"\n>>> PHASE 1 COMPLETE. ARCHITECTURE SELECTED.")
    
    # Lock the Skeleton
    skeleton = {k: best_p1_params[k] for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter", "optim_type", "act_fx"]}

    # --- STEP 2: NGOpt for CE (The "Everything" Optimizer) ---
    # NGOpt is Nevergrad's best algorithm for finding the lowest possible value.
    print(f"\n[PHASE 2] NGOpt: PUSHING CE TO ABSOLUTE MINIMUM")
    opt_p2 = ng.optimizers.NGOpt(parametrization=phase2_ultra_space(best_p1_params), budget=p2_budget)
    
    final_loss, final_params = run_unlimited_phase(
        opt_p2, 
        "ce", 
        fixed_params=skeleton, 
        budget=p2_budget
    )

    print("\n" + "!"*80)
    print("FINAL BEST RESULTS FOUND:")
    print(f"Lowest CE: {final_loss:.6f}")
    print(f"Parameters: {final_params}")
    print("!"*80)

if __name__ == "__main__":
    # Set budgets high for a real run
    run_full_pipeline(p1_budget=40, p2_budget=60)