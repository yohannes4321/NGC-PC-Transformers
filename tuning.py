"""Nevergrad HPO - FINAL BULLETPROOF VERSION"""
import os
import uuid
from concurrent import futures
import nevergrad as ng
import numpy as np
from trainer_wrapper import train_evaluate_model

# ==============================================================================
# 1. FIXED TOP-LEVEL WORKER
# ==============================================================================

def global_efe_worker(*args, **kwargs):
    """
    Fixed Worker: Accepts any combination of positional/keyword args.
    Focuses on Absolute EFE Stability.
    """
    # Nevergrad usually passes a single dict in kwargs or the first positional arg
    params = kwargs if kwargs else (args[0] if args else {})
    
    try:
        # We pass objective="efe" so the wrapper returns abs(EFE)
        arr = train_evaluate_model(params, objective="efe")
        return float(arr[0][0])
    except Exception as e:
        print(f"Worker Error: {e}", flush=True)
        return 1e10

def run_unlimited_phase(optimizer, objective_name, fixed_params, budget):
    """Phase 2 Runner: Forcing full exploration of Cross-Entropy (CE)."""
    best_loss = float("inf")
    best_params = None
    
    print(f"\n>>> STARTING {objective_name.upper()} DEEP SEARCH ({budget} Trials)", flush=True)
    
    for i in range(budget):
        candidate = optimizer.ask()
        x_dict = candidate.value
        # Merge fixed architecture with the new hyperparameter guesses
        full_params = {**fixed_params, **x_dict}
        
        # Type Enforcement for the Model
        for k in ["n_heads", "batch_size", "seq_len", "tau_m", "n_iter"]:
            if k in full_params: full_params[k] = int(full_params[k])
        if "n_heads" in full_params and "embed_mult" in full_params:
            full_params["n_embed"] = int(full_params["n_heads"]) * int(full_params["embed_mult"])

        print(f"   [Trial {i+1}/{budget}] Running full training...", flush=True)
        
        # This will run the actual training and return the CE loss
        arr = train_evaluate_model(full_params, objective=objective_name)
        loss_val = float(arr[0][0])
        optimizer.tell(candidate, loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = full_params
            print(f"   !!! NEW BEST {objective_name.upper()}: {best_loss:.4f}", flush=True)
            
    return best_loss, best_params

# ==============================================================================
# 2. SEARCH SPACES
# ==============================================================================

def get_p1_space():
    return ng.p.Dict(
        n_heads    = ng.p.Choice([2, 4, 8, 12, 16]),
        embed_mult = ng.p.Choice([8, 16, 24, 32, 64]),
        batch_size = ng.p.Choice([16, 32, 64]), 
        seq_len    = ng.p.Choice([16, 32, 64]), 
        eta        = ng.p.Log(lower=1e-8, upper=1e-3),
        tau_m      = ng.p.Scalar(lower=5, upper=30).set_integer_casting(),
        n_iter     = ng.p.Scalar(lower=1, upper=15).set_integer_casting(),
        wub        = ng.p.Scalar(lower=0.001, upper=0.1),
        wlb        = ng.p.Scalar(lower=-0.1, upper=-0.001),
        optim_type = ng.p.Choice(["adam", "sgd", "adamw"]),
        act_fx     = ng.p.Choice(["identity", "relu", "gelu", "swish"]),
    )

def get_p2_space(best_p1):
    eta_ref = float(best_p1.get("eta", 1e-5))
    return ng.p.Dict(
        # Allow huge movement to find better CE
        eta          = ng.p.Log(lower=eta_ref * 0.1, upper=min(eta_ref * 10.0, 1e-2)),
        wub          = ng.p.Scalar(lower=0.0001, upper=0.2),
        wlb          = ng.p.Scalar(lower=-0.2, upper=-0.0001),
        dropout_rate = ng.p.Scalar(lower=0.0, upper=0.5),
    )

# ==============================================================================
# 3. MAIN PIPELINE
# ==============================================================================

def run_full_pipeline(p1_budget=40, p2_budget=100):
    print("="*80)
    print("CRITICAL MISSION: MINIMIZE CE & STABILIZE EFE (NO LIMITS)")
    print("="*80, flush=True)

    # --- PHASE 1: STABILITY ---
    print(f"\n[PHASE 1] RUNNING {p1_budget} PARALLEL TRIALS FOR EFE...")
    opt_p1 = ng.optimizers.CMA(parametrization=get_p1_space(), budget=p1_budget, num_workers=4)
    
    with futures.ProcessPoolExecutor(max_workers=2) as executor:
        recommendation = opt_p1.minimize(global_efe_worker, executor=executor, batch_mode=False)
    
    best_p1_params = recommendation.value
    skeleton = {k: best_p1_params[k] for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter", "optim_type", "act_fx"]}
    
    print(f"\n>>> PHASE 1 DONE. STARTING DEEP CE OPTIMIZATION.")

    # --- PHASE 2: ACCURACY ---
    # NGOpt will use several algorithms to try and force CE below 14.
    opt_p2 = ng.optimizers.NGOpt(parametrization=get_p2_space(best_p1_params), budget=p2_budget)
    
    final_ce, final_params = run_unlimited_phase(
        optimizer=opt_p2, 
        objective_name="ce", 
        fixed_params=skeleton, 
        budget=p2_budget
    )

    print("\n" + "!"*80)
    print(f"MISSION COMPLETE.")
    print(f"FINAL LOWEST CE: {final_ce:.6f}")
    print(f"BEST PARAMS: {final_params}")
    print("!"*80, flush=True)

if __name__ == "__main__":
    run_full_pipeline(p1_budget=3, p2_budget=3)