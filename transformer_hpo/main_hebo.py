# filename: main_nevergrad.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import nevergrad as ng
from concurrent import futures
from trainer_wrapper import train_evaluate_model

def phase1_space():
    """
    Architecture search space. 
    Note: n_embed is NOT in the dict. We derive it from n_heads * embed_mult.
    """
    return ng.p.Dict(
        n_heads=ng.p.Scalar(lower=2, upper=8).set_integer_casting(),
        embed_mult=ng.p.Choice([8, 12, 16, 32]),
        batch_size=ng.p.Scalar(lower=2, upper=8).set_integer_casting(),
        seq_len=ng.p.Scalar(lower=8, upper=24).set_integer_casting(),
        n_layers=ng.p.Scalar(lower=1, upper=6).set_integer_casting(),
        pos_learnable=ng.p.Choice([True, False]),
        eta=ng.p.Log(lower=1e-6, upper=1e-4),
        tau_m=ng.p.Scalar(lower=10, upper=20).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=20).set_integer_casting(),
        dropout_rate=ng.p.Constant(0.0),
        wub=ng.p.Scalar(lower=0.01, upper=0.1),
        wlb=ng.p.Scalar(lower=-0.1, upper=-0.01),
        optim_type=ng.p.Choice(["adam", "sgd"]),
        act_fx=ng.p.Choice(["identity", "relu"]),
    )

def phase2_space(best):
    """Refine continuous params while keeping the 'best' architecture from Phase 1."""
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
        
        # --- THE FIX: ENFORCE DIVISIBILITY ---
        # If we have fixed_params (Phase 2), merge them first
        full_params = {**fixed_params, **x_dict} if fixed_params else x_dict.copy()
        
        # Explicitly calculate n_embed so it's guaranteed to be divisible by n_heads
        h = int(full_params["n_heads"])
        m = int(full_params["embed_mult"])
        full_params["n_embed"] = h * m
        
        # Ensure all shape-related params are strictly integers for JAX
        for k in ["n_heads", "n_embed", "batch_size", "seq_len", "n_layers", "tau_m", "n_iter"]:
            if k in full_params:
                full_params[k] = int(full_params[k])

        try:
            print(f"\nTrial {iteration} | heads: {full_params['n_heads']} | d_model: {full_params['n_embed']}")
            loss_array = train_evaluate_model(full_params, objective=objective_name)
            loss_value = float(loss_array[0][0])
            
            if np.isnan(loss_value):
                loss_value = float("inf")
        except Exception as e:
            # If the model still crashes (e.g. out of memory), report it as inf so Nevergrad learns
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
    # We fix the architecture (n_heads, embed_mult) and only tune continuous params
    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_arch), budget=p2_budget)
    
    # Inoculate with the best result from Phase 1
    opt2.suggest(eta=best_arch["eta"], wub=best_arch["wub"], wlb=best_arch["wlb"])

    # Pass the best_arch as 'fixed_params' so the architecture doesn't change
    best_ce, best_final, history2 = run_phase(opt2, "ce", fixed_params=best_arch, history=history1)

    print("\nDone!")
    print(f"Final Architecture: Heads={best_final['n_heads']}, D_Model={best_final['n_embed']}")
    print(f"Final Loss (CE): {best_ce:.4f}")

if __name__ == "__main__":
    run_two_phase_optimization()