import numpy as np
import os
import nevergrad as ng
from config import Config as config
from trainer_wrapper import evaluate_objective_efe, evaluate_objective_ce
from nevergrad import callbacks
import warnings
warnings.filterwarnings("ignore")

# --- Phase 1 Search Space (Architecture + Hyperparams) ---
def phase1_space():
    return ng.p.Dict(
        n_layers=ng.p.Choice([1, 2, 3, 4, 5, 6, 7, 8]),
        n_heads=ng.p.Choice([2, 3, 4, 5, 6, 7, 8]),
        embed_mult=ng.p.Choice([8, 12, 16]),

        batch_size=ng.p.Choice([2, 4, 6, 8, 10, 12]),
        seq_len=ng.p.Choice([8, 12, 16, 20, 24, 28, 32]),

        pos_learnable=ng.p.Choice([True, False]),
        act_fx=ng.p.Choice(["identity", "relu", "tanh"]),

        tau_m=ng.p.Choice([10, 12, 14, 16, 18, 20]),
        n_iter=ng.p.Choice([1, 4, 8, 16, 24, 30]),

        # FIXED: Added lower/upper keywords
        eta=ng.p.Log(lower=1e-6, upper=1e-4).set_mutation(sigma=1.0),
        wub=ng.p.Scalar(lower=0.01, upper=0.1).set_mutation(sigma=0.02),
        wlb=ng.p.Scalar(lower=-0.1, upper=-0.01).set_mutation(sigma=0.02),

        dropout_rate=ng.p.Constant(0.0),
    )

def phase2_space(best_p1):
    return ng.p.Dict(
        # FROZEN DISCRETE / ARCHITECTURE
        n_layers=ng.p.Constant(best_p1["n_layers"]),
        n_heads=ng.p.Constant(best_p1["n_heads"]),
        embed_mult=ng.p.Constant(best_p1["embed_mult"]),
        batch_size=ng.p.Constant(best_p1["batch_size"]),
        seq_len=ng.p.Constant(best_p1["seq_len"]),
        pos_learnable=ng.p.Constant(best_p1["pos_learnable"]),
        act_fx=ng.p.Constant(best_p1["act_fx"]),
        tau_m=ng.p.Constant(best_p1["tau_m"]),
        n_iter=ng.p.Constant(best_p1["n_iter"]),
        dropout_rate=ng.p.Constant(0.0),

        #  CONTINUOUS REFINEMENT ONLY
        # FIXED: Added lower/upper keywords
        eta=ng.p.Log(
            lower=max(1e-6, best_p1["eta"] * 0.2),
            upper=min(1e-4, best_p1["eta"] * 5.0),
        ).set_mutation(sigma=0.8),

        wub=ng.p.Scalar(
            lower=max(0.01, best_p1["wub"] - 0.02),
            upper=min(0.1, best_p1["wub"] + 0.02),
        ).set_mutation(sigma=0.01),

        wlb=ng.p.Scalar(
            lower=max(-0.1, best_p1["wlb"] - 0.02),
            upper=min(-0.01, best_p1["wlb"] + 0.02),
        ).set_mutation(sigma=0.01),
    )

def run_two_phase_optimization():
    FAILURE_PENALTY = 1e9
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n=== Phase 1: EFE Optimization (Budget: {config.p1_budget}) ===")
    
    optimizer1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=config.p1_budget)
    optimizer1.register_callback("tell", callbacks.OptimizerDump("checkpoints/optimizer1.pkl"))

    best_p1_params = None
    best_p1_loss = float("inf")

    for i in range(optimizer1.budget):
        cand = optimizer1.ask()
        # **kwargs expansion directly from cand.value (the Dict)
        loss = evaluate_objective_efe(**cand.value)

        if np.isnan(loss) or np.isinf(loss) or loss > 1e7:
            print(f"[P1 Trial {i}] ❌ Instability. Penalizing.")
            optimizer1.tell(cand, FAILURE_PENALTY)
        else:
            optimizer1.tell(cand, loss)
            if loss < best_p1_loss:
                best_p1_loss = loss
                best_p1_params = cand.value
                print(f"[P1 Trial {i}] ✅ New Best EFE: {loss:.4f}")

    # Use recommendation for the safest final point
    recommendation1 = optimizer1.provide_recommendation()
    best_architecture = recommendation1.value
    print(f"\n>>> PHASE 1 COMPLETE. Best Loss: {best_p1_loss}")

  
    print(f"\n=== Phase 2: CE Refinement (Budget: {config.p2_budget}) ===")
    
    optimizer2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_architecture), budget=config.p2_budget)
    optimizer2.register_callback("tell", callbacks.OptimizerDump("checkpoints/optimizer2.pkl"))

    # Seed the optimizer with Phase 1's winner
    optimizer2.suggest(**best_architecture)

    best_p2_loss = float("inf")

    for i in range(optimizer2.budget):
        cand = optimizer2.ask()
        loss = evaluate_objective_ce(**cand.value)

        # Penalize if above threshold or unstable
        if np.isnan(loss) or np.isinf(loss) or loss > 16:
            print(f"[P2 Trial {i}] ❌ Bad CE Loss ({loss:.2f}). Penalizing.")
            optimizer2.tell(cand, FAILURE_PENALTY)
        else:
            optimizer2.tell(cand, loss)
            if loss < best_p2_loss:
                best_p2_loss = loss
                print(f"[P2 Trial {i}] ✅ New Best CE: {loss:.4f}")

    recommendation2 = optimizer2.provide_recommendation()
    print("\n>>> OPTIMIZATION FINISHED")
    print(f"Final CE Loss: {best_p2_loss}")
    print(f"Best Params: {recommendation2.value}")

if __name__ == "__main__":
    run_two_phase_optimization()#