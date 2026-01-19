import os
import uuid
import sys
import time
from concurrent import futures
import nevergrad as ng
import numpy as np
from trainer_wrapper import train_evaluate_model

# --- 1. ENVIRONMENT SETUP ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- 2. SEARCH SPACES ---

def phase1_space():
    """Discrete Architecture Space for EFE Optimization."""
    return ng.p.Dict(
        n_heads    = ng.p.Choice([2, 4, 8]),
        embed_mult = ng.p.Choice([8, 16, 24]),
        batch_size = ng.p.Choice([16, 32]),
        seq_len    = ng.p.Choice([8,16 ]),
        eta        = ng.p.Log(lower=1e-7, upper=5e-5),
        tau_m      = ng.p.Scalar(lower=5, upper=15).set_integer_casting(),
        n_iter     = ng.p.Scalar(lower=1, upper=5).set_integer_casting(),
        wub        = ng.p.Scalar(lower=0.01, upper=0.04),
        wlb        = ng.p.Scalar(lower=-0.04, upper=-0.01),
        optim_type = ng.p.Choice(["adam", "sgd"]),
        act_fx     = ng.p.Choice(["identity", "relu"]),
    )

def phase2_space(best):
    """Continuous Hyperparameter Space for CE Refinement."""
    eta_best = float(best.get("eta", 1e-5))
    return ng.p.Dict(
        eta=ng.p.Log(lower=eta_best * 0.5, upper=min(eta_best * 2.0, 1e-4)),
        wub=ng.p.Scalar(lower=0.001, upper=0.04),
        wlb=ng.p.Scalar(lower=-0.04, upper=-0.001),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.2),
    )

def _get_params(args, kwargs):
    if len(args) > 0:
        if hasattr(args[0], "value"): return dict(args[0].value)
        if isinstance(args[0], dict): return dict(args[0])
    return dict(kwargs)

def parallel_func_phase1(*args, **kwargs):
    params = _get_params(args, kwargs)
    params["n_embed"] = int(params.get("n_heads", 1)) * int(params.get("embed_mult", 1))
    # Objective is hardcoded to 'efe' for Phase 1
    arr = train_evaluate_model(params, objective="efe")
    return float(arr[0][0])

def constraint_embed_divisible(x):
    return int(x.get("n_embed", 0)) % int(x.get("n_heads", 1)) == 0

# --- 3. EXECUTION BRAIN ---

def run_advanced(p1_budget=20, p2_budget=20, num_workers=2):
    print("\n" + "="*80)
    print("       STARTING ADVANCED MODEL OPTIMIZATION PIPELINE ")
    print("="*80)

    # PHASE 1
    print(f"\n[PHASE 1] >>> STARTING DISCRETE ARCHITECTURE OPTIMIZATION (EFE)")
    print(f"Goal: Minimize Expected Free Energy (EFE) to find stable model structure.")
    print("-" * 80)

    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=p1_budget, num_workers=num_workers)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        rec1 = opt1.minimize(parallel_func_phase1, executor=executor, batch_mode=False)
    
    current_best = rec1.value
    current_best["n_embed"] = int(current_best["n_heads"]) * int(current_best["embed_mult"])
    
    fixed_arch = {k: current_best[k] for k in [
        "n_heads", "embed_mult", "batch_size", "seq_len", 
        "tau_m", "n_iter", "optim_type", "act_fx", "n_embed"
    ]}

    print(f"\n PHASE 1 FINISHED: EFE OPTIMIZATION COMPLETE")
    print(f">>> HANDING OFF BEST ARCHITECTURE TO PHASE 2:")
    for k, v in fixed_arch.items(): print(f"    | {k:15}: {v}")
    print("=" * 80)

    # PHASE 2
    print(f"\n[PHASE 2] >>> STARTING CONTINUOUS HYPERPARAMETER OPTIMIZATION (CE)")
    print(f"Goal: Minimize Cross-Entropy (CE) for maximum accuracy on the chosen skeleton.")
    print("-" * 80)

    opt2 = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase2_space(current_best), budget=p2_budget)
    best_ce = float("inf")
    patience = 4
    min_delta = 0.5 
    no_improve = 0

    for i in range(p2_budget):
        cand = opt2.ask()
        merged = {**fixed_arch, **cand.value}
        
        print(f"\n--- [Phase 2] Trial {i+1}/{p2_budget} ---")
        print(f"🛠️  FULL PARAMETER LIST:")
        for key, val in merged.items():
            print(f"   | {key:15}: {val}")

        arr = train_evaluate_model(merged, objective="ce")
        loss = float(arr[0][0])
        opt2.tell(cand, loss)

        diff = best_ce - loss if best_ce != float("inf") else 100
        if diff > min_delta:
            best_ce = loss
            no_improve = 0
            print(f" NEW BEST CE: {best_ce:.4f} (Improved by {diff:.4f})")
        else:
            no_improve += 1
            print(f" Stagnation: Change ({diff:.4f}) is below threshold. Count: {no_improve}/{patience}")

        if no_improve >= patience:
            print(f" EARLY STOPPING PHASE 2: No significant change in CE.")
            break

    print("\n" + "" * 30)
    print("   OPTIMIZATION PIPELINE COMPLETE")
    print("" * 30)

if __name__ == "__main__":
    run_advanced(p1_budget=20, p2_budget=20, num_workers=2)