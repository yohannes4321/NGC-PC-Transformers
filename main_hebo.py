import os
import uuid
import sys
from concurrent import futures
import nevergrad as ng
from config import Config as config
from trainer_wrapper import train_evaluate_model

def _get_params(args, kwargs):
    """Fix for Nevergrad TypeError: extracts dict from Candidate object."""
    if len(args) > 0:
        if hasattr(args[0], "value"): return dict(args[0].value)
        if isinstance(args[0], dict): return dict(args[0])
    return dict(kwargs)

def parallel_func_phase1(*args, **kwargs):
    params = _get_params(args, kwargs)
    params["n_embed"] = int(params.get("n_heads", 1)) * int(params.get("embed_mult", 1))
    arr = train_evaluate_model(params, objective="efe")
    return float(arr[0][0])

def constraint_embed_divisible(x):
    return int(x.get("n_embed", 0)) % int(x.get("n_heads", 1)) == 0

def phase1_space():
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
    eta_best = float(best.get("eta", 1e-5))
    return ng.p.Dict(
        eta=ng.p.Log(lower=eta_best * 0.5, upper=min(eta_best * 2.0, 1e-4)),
        wub=ng.p.Scalar(lower=0.001, upper=0.04),
        wlb=ng.p.Scalar(lower=-0.04, upper=-0.001),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.2),
    )

def run_phase(optimizer, objective_name, fixed_params=None):
    best_loss = float("inf")
    best_params = None
    print(f"\n--- Starting {objective_name.upper()} Phase (Budget: {optimizer.budget}) ---", flush=True)
    for i in range(optimizer.budget):
        cand = optimizer.ask()
        merged = {**fixed_params, **cand.value} if fixed_params else cand.value
        arr = train_evaluate_model(merged, objective=objective_name)
        loss = float(arr[0][0])
        optimizer.tell(cand, loss)
        if loss < best_loss:
            best_loss, best_params = loss, merged
            print(f"  >>> New Best {objective_name}: {loss:.4f}")
    return best_loss, best_params

def run_advanced(p1_budget=20, p2_budget=20, num_workers=4):
    print("\n" + "="*60 + "\n      🚀 ADVANCED OPTIMIZATION PIPELINE\n" + "="*60, flush=True)

    # STEP 1: Parallel Exploration
    print(f"\n[STEP 1/4] PARALLEL EXPLORATION (Workers: {num_workers})")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=p1_budget, num_workers=num_workers)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        rec1 = opt1.minimize(parallel_func_phase1, executor=executor, batch_mode=False)
    
    current_best = rec1.value
    current_best["n_embed"] = int(current_best["n_heads"]) * int(current_best["embed_mult"])
    fixed_arch = {k: current_best[k] for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter", "optim_type", "act_fx", "n_embed"]}

    # STEP 2 & 3: Refinement and Chaining
    print(f"\n[STEP 2/4] PORTFOLIO REFINEMENT")
    opt2 = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase2_space(current_best), budget=p2_budget)
    _, current_best = run_phase(opt2, "ce", fixed_params=fixed_arch)

    print(f"\n[STEP 3/4] CHAINING SEARCH")
    ChainOpt = ng.optimizers.Chaining([ng.optimizers.LHSSearch, ng.optimizers.DE], [int(p2_budget*0.2)])
    opt3 = ChainOpt(parametrization=phase2_space(current_best), budget=p2_budget)
    _, current_best = run_phase(opt3, "ce", fixed_params=fixed_arch)

    # STEP 4: Pareto
    print(f"\n[STEP 4/4] PARETO ANALYSIS")
    opt4 = ng.optimizers.DE(parametrization=phase2_space(current_best), budget=p2_budget)
    for i in range(p2_budget):
        cand = opt4.ask()
        merged = {**fixed_arch, **cand.value}
        ce = float(train_evaluate_model(merged, objective="ce")[0][0])
        efe = float(train_evaluate_model(merged, objective="efe")[0][0])
        opt4.tell(cand, [ce, abs(efe)])

    print("\n" + "="*60 + "\n      ✅ PIPELINE COMPLETE\n" + "="*60, flush=True)

if __name__ == "__main__":
    run_advanced(p1_budget=20, p2_budget=20, num_workers=4)