import numpy as np
import os
import nevergrad as ng
from concurrent import futures
import warnings
import jax

# --- 1. SETUP & CONFIGURATION ---
warnings.filterwarnings("ignore")
# Note: Since JAX is already imported in your notebook, this flag might not apply 
# unless you Restart Runtime, but that is okay.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Assuming these are available in your directory
from config import Config as config
from trainer_wrapper import evaluate_objective_efe, evaluate_objective_ce

# --- 2. DEFINE SEARCH SPACES ---
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
        eta=ng.p.Log(lower=1e-6, upper=1e-4).set_mutation(sigma=1.0),
        wub=ng.p.Scalar(lower=0.01, upper=0.1).set_mutation(sigma=0.02),
        wlb=ng.p.Scalar(lower=-0.1, upper=-0.01).set_mutation(sigma=0.02),
        dropout_rate=ng.p.Constant(0.0),
    )

def phase2_space(best_p1):
    return ng.p.Dict(
        # Frozen Architecture
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
        # Fine-tuning Hyperparams
        eta=ng.p.Log(lower=max(1e-7, best_p1["eta"]*0.1), upper=min(1e-3, best_p1["eta"]*10.0)).set_mutation(sigma=0.5),
        wub=ng.p.Scalar(lower=max(0.001, best_p1["wub"]-0.05), upper=min(0.2, best_p1["wub"]+0.05)).set_mutation(sigma=0.01),
        wlb=ng.p.Scalar(lower=max(-0.2, best_p1["wlb"]-0.05), upper=min(-0.001, best_p1["wlb"]+0.05)).set_mutation(sigma=0.01),
    )

# --- 3. RUN OPTIMIZATION ---
def run_optimization_in_notebook():
    # Use max_workers=1 or 2. Since JAX uses all GPU memory by default, 
    # running multiple concurrent JAX trainings on one GPU usually causes OOM.
    # We recommend num_workers=1 for single-GPU setups.
    num_workers = 1 
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # === Phase 1 ===
    print("\n=== Phase 1: EFE Optimization ===")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=10)
    logger1 = ng.callbacks.ParametersLogger("checkpoints/phase1_logs.json")
    opt1.register_callback("tell", logger1)

    # Note: We use ThreadPoolExecutor. JAX releases the GIL mostly, so this works fine.
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation1 = opt1.minimize(evaluate_objective_efe, executor=executor)
    
    print(f"Phase 1 Winner: {recommendation1.value}")

    # === Phase 2 ===
    print("\n=== Phase 2: CE Optimization ===")
    p2_param = phase2_space(recommendation1.value)
    opt2 = ng.optimizers.NGOpt(parametrization=p2_param, budget=10)
    
    # Inoculate with best Phase 1 result
    opt2.suggest(**recommendation1.value)
    
    logger2 = ng.callbacks.ParametersLogger("checkpoints/phase2_logs.json")
    opt2.register_callback("tell", logger2)

    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation2 = opt2.minimize(evaluate_objective_ce, executor=executor)

    print(f"Phase 2 Winner: {recommendation2.value}")

# Run it
run_optimization_in_notebook()