import os
import nevergrad as ng
from concurrent import futures
from config import Config as config
from trainer_wrapper import evaluate_objective_efe, evaluate_objective_ce

def print_callback(optimizer, candidate, value):
    """Nevergrad callback to print results from the main process immediately."""
    # Value is what the function returned (EFE in P1, CE in P2)
    print(f"[CALLBACK] Trial Finished. Metric Value: {value:.4f}")
    print(f"[CALLBACK] Best so far: {optimizer.provide_recommendation().value}")

def phase1_space():
    return ng.p.Dict(
        n_layers=ng.p.Scalar(lower=1, upper=8).set_integer_casting(),
        n_heads=ng.p.Scalar(lower=2, upper=8).set_integer_casting(),
        embed_mult=ng.p.Choice([8, 12, 16]),
        batch_size=ng.p.Scalar(lower=2, upper=12).set_integer_casting(),
        seq_len=ng.p.Scalar(lower=8, upper=32).set_integer_casting(),
        pos_learnable=ng.p.Choice([True, False]),
        eta=ng.p.Log(lower=1e-6, upper=1e-4),
        tau_m=ng.p.Scalar(lower=10, upper=20).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=30).set_integer_casting(),
        wub=ng.p.Scalar(lower=0.01, upper=0.1),
        wlb=ng.p.Scalar(lower=-0.1, upper=-0.01),
        dropout_rate=ng.p.Constant(0.0),
        optim_type=ng.p.Choice(["adam", "sgd"]),
        act_fx=ng.p.Choice(["identity", "relu"]),
    )

def phase2_space(best_p1):
    return ng.p.Dict(
        eta=ng.p.Log(lower=float(best_p1["eta"]) * 0.2, upper=float(best_p1["eta"]) * 5.0),
        wub=ng.p.Scalar(lower=max(0.01, float(best_p1["wub"]) - 0.02), upper=min(0.1, float(best_p1["wub"]) + 0.02)),
        wlb=ng.p.Scalar(lower=max(-0.1, float(best_p1["wlb"]) - 0.02), upper=min(-0.01, float(best_p1["wlb"]) + 0.02)),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.3),
        # Fixed Arch from P1
        **{k: ng.p.Constant(v) for k, v in best_p1.items() if k not in ["eta", "wub", "wlb", "dropout_rate"]}
    )

def run_two_phase_optimization():
    N_WORKERS = int(os.environ.get("NG_WORKERS", 4))
    
    # PHASE 1
    print(f"\n=== Phase 1: EFE Optimization (Budget: {config.p1_budget}) ===")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=config.p1_budget, num_workers=N_WORKERS)
    opt1.register_callback("tell", print_callback)

    with futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        recommendation_p1 = opt1.minimize(evaluate_objective_efe, executor=executor, batch_mode=False)

    best_p1 = recommendation_p1.value
    print(f"\n>>> PHASE 1 COMPLETE. Best EFE Params: {best_p1}")

    # PHASE 2
    print(f"\n=== Phase 2: CE Refinement (Budget: {config.p2_budget}) ===")
    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_p1), budget=config.p2_budget, num_workers=N_WORKERS)
    opt2.register_callback("tell", print_callback)
    opt2.suggest(**best_p1)

    with futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        recommendation_p2 = opt2.minimize(evaluate_objective_ce, executor=executor, batch_mode=False)

    print(f"\n>>> PHASE 2 COMPLETE. Final Params: {recommendation_p2.value}")

if __name__ == "__main__":
    run_two_phase_optimization()