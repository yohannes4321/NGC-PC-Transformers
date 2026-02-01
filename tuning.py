import numpy as np
import os
import nevergrad as ng
from concurrent import futures
import warnings

# --- SETUP ---
warnings.filterwarnings("ignore")
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Assuming these imports exist in your project
from config import Config as config
from trainer_wrapper import evaluate_objective_efe, evaluate_objective_ce

# --- Phase 1 Search Space (Architecture + Hyperparams) ---
def phase1_space():
    return ng.p.Dict(
        # Architecture (Discrete choices)
        n_layers=ng.p.Choice([1, 2, 3, 4, 5, 6, 7, 8]),
        n_heads=ng.p.Choice([2, 3, 4, 5, 6, 7, 8]),
        embed_mult=ng.p.Choice([8, 12, 16]),
        batch_size=ng.p.Choice([2, 4, 6, 8, 10, 12]),
        seq_len=ng.p.Choice([8, 12, 16, 20, 24, 28, 32]),
        pos_learnable=ng.p.Choice([True, False]),
        act_fx=ng.p.Choice(["identity", "relu", "tanh"]),
        tau_m=ng.p.Choice([10, 12, 14, 16, 18, 20]),
        n_iter=ng.p.Choice([1, 4, 8, 16, 24, 30]),

        # Hyperparameters (Continuous)
        eta=ng.p.Log(lower=1e-6, upper=1e-4).set_mutation(sigma=1.0),
        wub=ng.p.Scalar(lower=0.01, upper=0.1).set_mutation(sigma=0.02),
        wlb=ng.p.Scalar(lower=-0.1, upper=-0.01).set_mutation(sigma=0.02),
        dropout_rate=ng.p.Constant(0.0),
    )

# --- Phase 2 Search Space (Refinement) ---
# We freeze the discrete architecture choices and only refine the continuous hyperparameters
def phase2_space(best_p1):
    return ng.p.Dict(
        # FROZEN: Set strictly to the values found in Phase 1
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

        # REFINEMENT: Tight bounds around the Phase 1 winner
        # We shrink the search space around the best 'eta', 'wub', 'wlb'
        eta=ng.p.Log(
            lower=max(1e-7, best_p1["eta"] * 0.1), 
            upper=min(1e-3, best_p1["eta"] * 10.0)
        ).set_mutation(sigma=0.5), # Lower sigma for fine-tuning

        wub=ng.p.Scalar(
            lower=max(0.001, best_p1["wub"] - 0.05),
            upper=min(0.2, best_p1["wub"] + 0.05),
        ).set_mutation(sigma=0.01),

        wlb=ng.p.Scalar(
            lower=max(-0.2, best_p1["wlb"] - 0.05),
            upper=min(-0.001, best_p1["wlb"] + 0.05),
        ).set_mutation(sigma=0.01),
    )

def run_two_phase_optimization():
    num_workers = 2
    os.makedirs("checkpoints", exist_ok=True)
    
    # ==========================================
    # PHASE 1: EXPLORATION (EFE)
    # ==========================================
    print("\n=== Phase 1: EFE Optimization ===")
    
    # Using NGOpt is good for general purpose
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=10)
    logger1 = ng.callbacks.ParametersLogger("checkpoints/phase1_logs.json")
    opt1.register_callback("tell", logger1)

    # Standard minimize usage for Phase 1
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation1 = opt1.minimize(evaluate_objective_efe, executor=executor)
    
    print(f"Phase 1 Best EFE Params: {recommendation1.value}")

    # ==========================================
    # PHASE 2: REFINEMENT (CE)
    # ==========================================
    print("\n=== Phase 2: CE Optimization (Refining Phase 1 Best) ===")
    
    # 1. Initialize Space centered around Phase 1 best
    p2_param = phase2_space(recommendation1.value)
    
    # 2. Create Optimizer
    # TwoPointsDE is often better for refinement/fine-tuning than NGOpt, 
    # but NGOpt is safe if you are unsure.
    opt2 = ng.optimizers.NGOpt(parametrization=p2_param, budget=10)
    
    # 3. CRITICAL STEP: INOCULATION using SUGGEST
    # This is how you transfer the "Good Trail"
    # We tell Opt2: "Start by checking this specific point."
    print("--> Inoculating Phase 2 with best Phase 1 candidate...")
    opt2.suggest(**recommendation1.value)
    
    # Note on Learning:
    # When opt2.minimize starts, the very first thing it does is 'ask'.
    # Because we called suggest(), the first 'ask' will return our suggested point.
    # It will then evaluate it (calculate CE loss) and 'tell' the optimizer.
    # The optimizer will see this result and center its Gaussian/Search around it
    # if it's good (which it likely is).
    
    logger2 = ng.callbacks.ParametersLogger("checkpoints/phase2_logs.json")
    opt2.register_callback("tell", logger2)

    # 4. Run Minimization
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation2 = opt2.minimize(evaluate_objective_ce, executor=executor)

    print(f"Phase 2 Best CE Params: {recommendation2.value}")
    return recommendation2.value

if __name__ == "__main__":
    run_two_phase_optimization()