import os
import uuid
import sys
import time
from concurrent import futures
import nevergrad as ng
import numpy as np
from trainer_wrapper import train_evaluate_model

# --- 1. MEMORY & ENVIRONMENT SETUP ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- 2. UTILITY FUNCTIONS ---
def _get_params(args, kwargs):
    """Fix for Nevergrad TypeError: extracts dict from Candidate object."""
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

def constraint_embed_divisible(x):
    """Ensures n_embed is perfectly divisible by n_heads."""
    return int(x.get("n_embed", 0)) % int(x.get("n_heads", 1)) == 0

# --- 3. SEARCH SPACE DEFINITIONS (Must be above run_advanced) ---

def phase1_space():
    """Discrete Architecture Space for EFE Optimization."""
    return ng.p.Dict(
        n_heads    = ng.p.Choice([2, 4, 8]),
        embed_mult = ng.p.Choice([8, 16, 24]),
        batch_size = ng.p.Choice([16, 32, 64]),
        seq_len    = ng.p.Choice([16, 32, 48]),
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

# --- 4. THE BRAIN (The Orchestrator) ---

def run_advanced(p1_budget=20, p2_budget=20, num_workers=2):
    print("\n" + "="*75)
    print("      🌟 STARTING ADVANCED MODEL OPTIMIZATION PIPELINE 🌟")
    print("="*75)

    # --- PHASE 1: DISCRETE ARCHITECTURE EXPLORATION (EFE FOCUS) ---
    print(f"\n[PHASE 1] >>> STARTING DISCRETE ARCHITECTURE OPTIMIZATION (EFE)")
    print(f"Goal: Minimize Expected Free Energy to find a stable model skeleton.")
    print(f"Status: Searching for optimal Heads, Embedding Size, and Batch settings.")
    print("-" * 75)

    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=p1_budget, num_workers=num_workers)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        rec1 = opt1.minimize(parallel_func_phase1, executor=executor, batch_mode=False)
    
    current_best = rec1.value
    current_best["n_embed"] = int(current_best["n_heads"]) * int(current_best["embed_mult"])
    
    # Lock the best skeleton
    fixed_arch = {k: current_best[k] for k in [
        "n_heads", "embed_mult", "batch_size", "seq_len", 
        "tau_m", "n_iter", "optim_type", "act_fx", "n_embed"
    ]}

    print(f"\n🏁 PHASE 1 FINISHED: EFE OPTIMIZATION COMPLETE")
    print(f"   Handoff Architecture: {fixed_arch['n_heads']} Heads | {fixed_arch['n_embed']} Embed Size")
    print("=" * 75)

    # --- PHASE 2: CONTINUOUS HYPERPARAMETER REFINEMENT (CE FOCUS) ---
    print(f"\n[PHASE 2] >>> STARTING CONTINUOUS CE OPTIMIZATION")
    print(f"Goal: Minimize Cross-Entropy (CE) to maximize model accuracy.")
    print(f"Status: Fine-tuning Learning Rate and Weights for the locked architecture.")
    print("-" * 75)

    opt2 = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase2_space(current_best), budget=p2_budget)
    
    best_ce = float("inf")
    patience = 4
    no_improve = 0

    for i in range(p2_budget):
        cand = opt2.ask()
        merged = {**fixed_arch, **cand.value}
        
        print(f"\n--- [Phase 2] Continuous Trial {i+1}/{p2_budget} ---")
        print(f"🛠️  Active Parameters:")
        for key, val in merged.items():
            print(f"   | {key:15}: {val}")

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
            print(f"⚠️ No significant improvement. Stagnation: {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"🛑 EARLY STOPPING: Accuracy plateaued. Moving to Pareto Analysis.")
            break

    # --- PHASE 4: PARETO ANALYSIS ---
    print("\n" + "="*75)
    print("[PHASE 4] >>> STARTING FINAL PARETO BALANCE ANALYSIS")
    print("Goal: Evaluate the trade-off between Accuracy (CE) and Energy (EFE).")
    print("="*75)

    opt4 = ng.optimizers.DE(parametrization=phase2_space(current_best), budget=p2_budget)
    
    for i in range(p2_budget):
        cand = opt4.ask()
        merged = {**fixed_arch, **cand.value}
        
        print(f"\n[Pareto Trial {i+1}] Evaluating Balance: LR={merged['eta']:.2e}")
        
        ce_score = float(train_evaluate_model(merged, objective="ce")[0][0])
        efe_score = float(train_evaluate_model(merged, objective="efe")[0][0])
        
        print(f"   ⚖️  Results: CE Accuracy = {ce_score:.4f} | EFE Stability = {efe_score:.2f}")
        opt4.tell(cand, [abs(efe_score), ce_score])

    print("\n" + "🚀" * 30)
    print("   OPTIMIZATION PIPELINE COMPLETE")
    print("🚀" * 30)

# --- 5. EXECUTION ENTRANCE ---
if __name__ == "__main__":
    run_advanced(p1_budget=20, p2_budget=20, num_workers=2)