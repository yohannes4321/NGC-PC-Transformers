import numpy as np
import os
import nevergrad as ng
from config import Config as config
from trainer_wrapper import evaluate_objective_efe, evaluate_objective_ce
from nevergrad import callbacks
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # Takes only what is needed
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

from concurrent import futures
import numpy as np

def run_two_phase_optimization():
    # 1. SETUP
    num_workers = 2  # Matches your 2 GPUs
    os.makedirs("checkpoints", exist_ok=True)
    
    # --- PHASE 1: EFE OPTIMIZATION ---
    print("\n=== Phase 1: EFE Optimization ===")
    p1_log_path = "checkpoints/phase1_logs.json"
    
    # Define Optimizer & Logger
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=10)
    logger1 = ng.callbacks.ParametersLogger(p1_log_path)
    opt1.register_callback("tell", logger1)

    # Parallel Execution Loop
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation1 = opt1.minimize(evaluate_objective_efe, executor=executor)
    
    print(f"Phase 1 Best EFE Params: {recommendation1.value}")

    # --- PHASE 2: CE OPTIMIZATION ---
    print("\n=== Phase 2: CE Optimization ===")
    p2_log_path = "checkpoints/phase2_logs.json"
    
    # Use best results from Phase 1 as a starting point for Phase 2
    p2_param = phase2_space()
    p2_param.set_name("CE_Space")
    # Suggesting the best candidate from Phase 1 to Phase 2
    p2_param.value = recommendation1.value 

    opt2 = ng.optimizers.NGOpt(parametrization=p2_param, budget=10)
    logger2 = ng.callbacks.ParametersLogger(p2_log_path)
    opt2.register_callback("tell", logger2)

    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation2 = opt2.minimize(evaluate_objective_ce, executor=executor)

    print(f"Phase 2 Best CE Params: {recommendation2.value}")
    return recommendation2.value
if __name__ == "__main__":
    run_two_phase_optimization()#