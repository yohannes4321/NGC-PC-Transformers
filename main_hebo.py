import os
import uuid
import sys
import time
from concurrent import futures
import nevergrad as ng
import numpy as np
from trainer_wrapper import train_evaluate_model

# --- MEMORY SAFETY FLAGS ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def _get_params(args, kwargs):
    if len(args) > 0:
        if hasattr(args[0], "value"): return dict(args[0].value)
        if isinstance(args[0], dict): return dict(args[0])
    return dict(kwargs)

def parallel_func_phase1(*args, **kwargs):
    """Worker for Phase 1: Focused on Stability (EFE)."""
    params = _get_params(args, kwargs)
    params["n_embed"] = int(params.get("n_heads", 1)) * int(params.get("embed_mult", 1))
    arr = train_evaluate_model(params, objective="efe")
    return float(arr[0][0])

# --- MAIN PIPELINE ---

def run_advanced(p1_budget=20, p2_budget=20, num_workers=2):
    print("\n" + "="*70)
    print("      🌟 STARTING ADVANCED MODEL OPTIMIZATION PIPELINE 🌟")
    print("="*70)

    # -------------------------------------------------------------------------
    # PHASE 1: DISCRETE ARCHITECTURE EXPLORATION (EFE FOCUS)
    # -------------------------------------------------------------------------
    print(f"\n[PHASE 1] >>> STARTING EFE ARCHITECTURE OPTIMIZATION")
    print(f"Goal: Minimize Expected Free Energy (EFE) to find a stable model skeleton.")
    print(f"Mode: Parallel Execution ({num_workers} Workers)")
    print("-" * 70)

    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=p1_budget, num_workers=num_workers)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        rec1 = opt1.minimize(parallel_func_phase1, executor=executor, batch_mode=False)
    
    current_best = rec1.value
    current_best["n_embed"] = int(current_best["n_heads"]) * int(current_best["embed_mult"])
    
    # LOCKING THE ARCHITECTURE
    fixed_arch = {k: current_best[k] for k in [
        "n_heads", "embed_mult", "batch_size", "seq_len", 
        "tau_m", "n_iter", "optim_type", "act_fx", "n_embed"
    ]}

    print(f"\n🏁 PHASE 1 FINISHED: ARCHITECTURE LOCKED")
    print(f"   Selected Skeleton: {fixed_arch['n_heads']} Heads | {fixed_arch['n_embed']} Embedding Size")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # PHASE 2: CONTINUOUS HYPERPARAMETER REFINEMENT (CE FOCUS)
    # -------------------------------------------------------------------------
    print(f"\n[PHASE 2] >>> STARTING CONTINUOUS CE OPTIMIZATION")
    print(f"Goal: Minimize Cross-Entropy (CE) to maximize model accuracy.")
    print(f"Mode: Sequential Execution (Polishing the best architecture)")
    print("-" * 70)

    # We start with the best result from Phase 1 and now tune the "Brain"
    opt2 = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase2_space(current_best), budget=p2_budget)
    
    best_ce = float("inf")
    patience = 4
    no_improve = 0

    for i in range(p2_budget):
        cand = opt2.ask()
        # Merge the fixed architecture with the new continuous parameters (eta, dropout, etc.)
        merged = {**fixed_arch, **cand.value}
        
        print(f"\n--- [Phase 2] Trial {i+1}/{p2_budget} ---")
        print(f"🛠️  Full Parameter Set:")
        for key, val in merged.items():
            print(f"   | {key:15}: {val}")

        # Run model with CE objective
        arr = train_evaluate_model(merged, objective="ce")
        loss = float(arr[0][0])
        opt2.tell(cand, loss)

        if loss < best_ce:
            improvement = best_ce - loss if best_ce != float("inf") else 0
            best_ce = loss
            no_improve = 0
            print(f"✅ NEW BEST CE FOUND: {best_ce:.4f} (Improved by {improvement:.4f})")
        else:
            no_improve += 1
            print(f"⚠️ No significant change. Stagnation: {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"🛑 EARLY STOPPING: Accuracy has plateaued. Moving to Pareto Analysis.")
            break

    # -------------------------------------------------------------------------
    # PHASE 4: PARETO ANALYSIS (EFE vs CE BALANCE)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("[PHASE 4] >>> STARTING PARETO BALANCE ANALYSIS")
    print("Goal: Find the final trade-off between Accuracy (CE) and Stability (EFE).")
    print("="*70)

    opt4 = ng.optimizers.DE(parametrization=phase2_space(current_best), budget=p2_budget)
    
    for i in range(p2_budget):
        cand = opt4.ask()
        merged = {**fixed_arch, **cand.value}
        
        print(f"\n[Pareto Trial {i+1}] Testing Balance for Params: LR={merged['eta']:.2e}, Dropout={merged.get('dropout_rate', 0)}")
        
        ce_score = float(train_evaluate_model(merged, objective="ce")[0][0])
        efe_score = float(train_evaluate_model(merged, objective="efe")[0][0])
        
        print(f"   ⚖️  Results: CE = {ce_score:.4f} | EFE = {efe_score:.2f}")
        opt4.tell(cand, [abs(efe_score), ce_score])

    print("\n" + "🚀" * 30)
    print("   OPTIMIZATION PIPELINE COMPLETE")
    print("   Check 'experiments_summary.csv' for the final ranking.")
    print("🚀" * 30)

if __name__ == "__main__":
    # Example budgets
    run_advanced(p1_budget=20, p2_budget=20, num_workers=2)