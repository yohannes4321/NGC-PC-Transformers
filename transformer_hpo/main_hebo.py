# filename: main_nevergrad.py
import os
import math
import numpy as np
import nevergrad as ng
from trainer_wrapper import train_evaluate_model

def phase1_space():
    """
    Architecture search space constrained to prevent JAX Tracer and Reshape errors.
    """
    return ng.p.Dict(
        # FIX 1: Discrete choices for architecture to prevent constant JIT recompilation
        n_heads=ng.p.Choice([2, 4, 8]), 
        embed_mult=ng.p.Choice([8, 16, 32]),
        
        # FIX 2: Constrain batch/seq to powers of 2 or common factors to avoid reshape mismatches
        batch_size=ng.p.Choice([4, 8]),
        seq_len=ng.p.Choice([16, 32]), 
        
        n_layers=ng.p.Choice([1, 2, 4]),
        pos_learnable=ng.p.Choice([True, False]),
        
        # Continuous hyperparameters are fine as they don't change tensor shapes
        eta=ng.p.Log(lower=1e-6, upper=1e-4),
        tau_m=ng.p.Scalar(lower=10, upper=20).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=20).set_integer_casting(),
        dropout_rate=ng.p.Constant(0.0),
        wub=ng.p.Scalar(lower=0.01, upper=0.1),
        wlb=ng.p.Scalar(lower=-0.1, upper=-0.01),
        optim_type=ng.p.Choice(["adam", "sgd"]),
        act_fx=ng.p.Choice(["identity", "relu"]),

        # Request live per-batch logging from the trainer (if run_training supports it).
        # Set interval to 10 to print EFE, CE and PPL every 10 batches.
        live_logging=ng.p.Constant(True),
        log_batch_interval=ng.p.Constant(10),
    )

def phase2_space(best):
    """Refine continuous params while keeping the 'best' architecture fixed."""
    eta_best = float(best.get("eta", 1e-5))
    wub_best = float(best.get("wub", 0.05))
    wlb_best = float(best.get("wlb", -0.05))

    return ng.p.Dict(
        eta=ng.p.Log(lower=max(eta_best * 0.2, 1e-7), upper=eta_best * 5.0),
        wub=ng.p.Scalar(lower=max(0.01, wub_best - 0.02), upper=min(0.1, wub_best + 0.02)),
        wlb=ng.p.Scalar(lower=max(-0.1, wlb_best - 0.02), upper=min(-0.01, wlb_best + 0.02)),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.3)
    )

def run_phase(optimizer, objective_name, fixed_params=None, history=None):
    best_loss = float("inf")
    best_params = None
    losses = [] if history is None else history

    for iteration in range(optimizer.budget):
        candidate = optimizer.ask()
        x_dict = candidate.value
        
        # Merge fixed architecture (if in Phase 2)
        full_params = {**fixed_params, **x_dict} if fixed_params else x_dict.copy()
        
        # --- THE DIVISIBILITY FIX ---
        h = int(full_params["n_heads"])
        m = int(full_params["embed_mult"])
        full_params["n_embed"] = h * m
        
        # --- THE JAX CONCRETE TYPE FIX ---
        # Ensure all dimensions are standard Python ints (not numpy types)
        int_keys = ["n_heads", "n_embed", "batch_size", "seq_len", "n_layers", "tau_m", "n_iter"]
        for k in int_keys:
            if k in full_params:
                full_params[k] = int(full_params[k])

        try:
            print(f"\nTrial {iteration} | Heads: {full_params['n_heads']} | D_Model: {full_params['n_embed']} | Seq: {full_params['seq_len']}")
            loss_array = train_evaluate_model(full_params, objective=objective_name)
            loss_value = float(loss_array[0][0])
            
            if np.isnan(loss_value):
                loss_value = float("inf")
        except Exception as e:
            print(f"!!! CRASH IN TRIAL {iteration} !!! Error: {e}")
            loss_value = float("inf")

        optimizer.tell(candidate, loss_value)
        losses.append(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_params = full_params
            print(f">>> NEW BEST {objective_name.upper()} = {best_loss:.4f}")

    return best_loss, best_params, losses

def run_two_phase_optimization(p1_budget=30, p2_budget=40):
    print("--- Phase 1: Arch Search (EFE) ---")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=p1_budget)
    best_efe, best_arch, history1 = run_phase(opt1, "efe")

    if best_arch is None:
        print("Search failed.")
        return

    print(f"\n--- Phase 2: Hyperparam Refinement (CE) ---")
    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_arch), budget=p2_budget)
    
    # Suggest best from Phase 1 to seed Phase 2
    opt2.suggest(eta=best_arch["eta"], wub=best_arch["wub"], wlb=best_arch["wlb"])

    best_ce, best_final, history2 = run_phase(opt2, "ce", fixed_params=best_arch, history=history1)

    print("\nOptimization Finished Successfully!")
    print(f"Final Architecture: Heads={best_final['n_heads']}, D_Model={best_final['n_embed']}")
    print(f"Final Params: Batch={best_final['batch_size']}, Seq={best_final['seq_len']}")
    print(f"Final Loss (CE): {best_ce:.4f}")

if __name__ == "__main__":
    run_two_phase_optimization()