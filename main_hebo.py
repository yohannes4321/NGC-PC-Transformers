"""Nevergrad entrypoint.
Orchestrates a two-phase optimization (EFE -> CE) using parallel workers.
"""
import os
import nevergrad as ng
from concurrent import futures
from config import Config as config
# Import specific objective functions for pickling compatibility
from trainer_wrapper import evaluate_objective_efe, evaluate_objective_ce

def phase1_space():
    """
    Phase 1: Search Architecture & Dynamics.
    We optimize n_heads and embed_mult. n_embed is calculated inside the worker.
    """
    return ng.p.Dict(
        # Architecture
        n_layers=ng.p.Scalar(lower=1, upper=8).set_integer_casting(),
        n_heads=ng.p.Scalar(lower=2, upper=8).set_integer_casting(),
        embed_mult=ng.p.Choice([8, 12, 16]), # n_embed = n_heads * embed_mult
        
        # Dimensions
        batch_size=ng.p.Scalar(lower=2, upper=12).set_integer_casting(),
        seq_len=ng.p.Scalar(lower=8, upper=32).set_integer_casting(),
        pos_learnable=ng.p.Choice([True, False]),

        # Training dynamics
        eta=ng.p.Log(lower=1e-6, upper=1e-4),
        tau_m=ng.p.Scalar(lower=10, upper=20).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=30).set_integer_casting(),
        
        # Hyperparams
        wub=ng.p.Scalar(lower=0.01, upper=0.1),
        wlb=ng.p.Scalar(lower=-0.1, upper=-0.01),
        dropout_rate=ng.p.Constant(0.0), # Fixed in P1
        
        # Categorical
        optim_type=ng.p.Choice(["adam", "sgd"]),
        act_fx=ng.p.Choice(["identity", "relu"]),
    )

def phase2_space(best_p1):
    """
    Phase 2: Refine Hyperparameters (Learning Rate, Weights, Dropout).
    Locks the architecture found in Phase 1.
    """
    # Extract scalar values from the best recommendation
    eta_best = float(best_p1["eta"])
    wub_best = float(best_p1["wub"])
    wlb_best = float(best_p1["wlb"])
    dropout_best = float(best_p1.get("dropout_rate", 0.0))

    return ng.p.Dict(
        # Log-scale refinement around best eta
        eta=ng.p.Log(lower=eta_best * 0.2, upper=eta_best * 5.0),
        
        # Refine weights
        wub=ng.p.Scalar(lower=max(0.01, wub_best - 0.02), upper=min(0.1, wub_best + 0.02)),
        wlb=ng.p.Scalar(lower=max(-0.1, wlb_best - 0.02), upper=min(-0.01, wlb_best + 0.02)),
        
        # Allow dropout to vary now
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.3),
        
        # KEEP ARCHITECTURE FIXED
        n_layers=ng.p.Constant(best_p1["n_layers"]),
        n_heads=ng.p.Constant(best_p1["n_heads"]),
        embed_mult=ng.p.Constant(best_p1["embed_mult"]),
        batch_size=ng.p.Constant(best_p1["batch_size"]),
        seq_len=ng.p.Constant(best_p1["seq_len"]),
        pos_learnable=ng.p.Constant(best_p1["pos_learnable"]),
        tau_m=ng.p.Constant(best_p1["tau_m"]),
        n_iter=ng.p.Constant(best_p1["n_iter"]),
        optim_type=ng.p.Constant(best_p1["optim_type"]),
        act_fx=ng.p.Constant(best_p1["act_fx"]),
    )

def run_two_phase_optimization():
    # Number of workers (adjust based on your CPU cores / GPU VRAM)
    N_WORKERS = int(os.environ.get("NG_WORKERS", 1))
    
    # --- PHASE 1: OPTIMIZE EFE ---
    print(f"\n=== Phase 1: EFE Optimization (Budget: {config.p1_budget}) ===")
    
    # NGOpt is the recommended meta-optimizer
    opt1 = ng.optimizers.NGOpt(
        parametrization=phase1_space(), 
        budget=config.p1_budget, 
        num_workers=N_WORKERS
    )

    # Use ProcessPoolExecutor for true parallel processing
    with futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # .minimize() automatically handles ask/tell and distributing to workers
        recommendation_p1 = opt1.minimize(
            evaluate_objective_efe, 
            executor=executor, 
            batch_mode=False
        )

    best_p1_params = recommendation_p1.value
    print(f"\n>>> Phase 1 Best Params: {best_p1_params}")

    # --- PHASE 2: OPTIMIZE CE ---
    print(f"\n=== Phase 2: CE Refinement (Budget: {config.p2_budget}) ===")
    
    # Create space based on Phase 1 results
    p2_space = phase2_space(best_p1_params)
    
    opt2 = ng.optimizers.NGOpt(
        parametrization=p2_space, 
        budget=config.p2_budget, 
        num_workers=N_WORKERS
    )

    # Warm start: Suggest the exact best point from Phase 1 as a starting point
    # We construct a child based on best_p1_params, ensuring keys match
    try:
        # Filter best_p1_params to match keys needed for Phase 2 initialization if needed,
        # but NGOpt handles overlapping keys well usually.
        opt2.suggest(**best_p1_params) 
    except Exception as e:
        print(f"Warm start warning: {e}")

    with futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        recommendation_p2 = opt2.minimize(
            evaluate_objective_ce, 
            executor=executor, 
            batch_mode=False
        )

    best_final_params = recommendation_p2.value
    
    # Calculate final display info
    final_d_model = best_final_params['n_heads'] * best_final_params['embed_mult']
    
    print("\n==========================================")
    print("OPTIMIZATION COMPLETE")
    print(f"Final Architecture: Heads={best_final_params['n_heads']}, D_Model={final_d_model}")
    print(f"Final Dynamics: Eta={best_final_params['eta']:.6f}, Dropout={best_final_params['dropout_rate']:.4f}")
    print("==========================================")

if __name__ == "__main__":
    run_two_phase_optimization()