# filename: main_nevergrad.py
"""
Two-phase Nevergrad :
  Phase 1: search architecture + continuous params minimizing avg_train_efe (proxy).
  Phase 2: refine continuous params around Phase 1 best, minimizing validation CE.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import nevergrad as ng
from concurrent import futures
from trainer_wrapper import train_evaluate_model


def phase1_space():
  
    return ng.p.Dict(
        n_heads=ng.p.Scalar(lower=2, upper=8).set_integer_casting(),
        embed_mult=ng.p.Choice([8, 12, 16]),
        n_embed=ng.p.Scalar(lower=16, upper=4096).set_integer_casting(),  # will be overridden to n_heads*embed_mult
        batch_size=ng.p.Scalar(lower=2, upper=12).set_integer_casting(),
        seq_len=ng.p.Scalar(lower=8, upper=32).set_integer_casting(),

        n_layers=ng.p.Scalar(lower=1, upper=8).set_integer_casting(),
        pos_learnable=ng.p.Choice([True, False]),
        eta=ng.p.Log(lower=1e-6, upper=1e-4),
        tau_m=ng.p.Scalar(lower=10, upper=20).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=30).set_integer_casting(),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.0),  # fixed 0.0
        wub=ng.p.Scalar(lower=0.01, upper=0.1),
        wlb=ng.p.Scalar(lower=-0.1, upper=-0.01),
        optim_type=ng.p.Choice(["adam", "sgd"]),
        act_fx=ng.p.Choice(["identity", "relu"]),
    )


def phase2_space(best):
    """Phase 2: only continuous params around Phase 1 best, same names as Phase 1."""
    eta_best = float(best.get("eta", 1e-5))
    dropout_best = float(best.get("dropout_rate", 0.0))
    wub_best = float(best.get("wub", 0.05))
    wlb_best = float(best.get("wlb", -0.05))

    return ng.p.Dict(
        eta=ng.p.Log(lower=max(eta_best * 0.2, 1e-6), upper=eta_best * 5.0),
        dropout_rate=ng.p.Scalar(lower=max(0.0, dropout_best - 0.05), upper=min(0.3, dropout_best + 0.05)),
        wub=ng.p.Scalar(lower=max(0.01, wub_best - 0.02), upper=min(0.1, wub_best + 0.02)),
        wlb=ng.p.Scalar(lower=max(-0.1, wlb_best - 0.02), upper=min(-0.01, wlb_best + 0.02)),
    )


# Cheap constraint: ensure architecture validity (picklable function, no lambda)
def constraint_embed_divisible(value_dict):
    try:
        n_embed = int(value_dict["n_embed"])  # may be numpy or float
        n_heads = int(value_dict["n_heads"]) or 1
        embed_mult = int(value_dict.get("embed_mult", max(1, n_embed // max(1, n_heads))))
        # Enforce exact equality n_embed == n_heads * embed_mult
        return 0.0 if n_embed == n_heads * embed_mult else -1.0
    except Exception:
        return -1.0


def run_phase(optimizer, objective_name, fixed_params=None, history=None):
    best_loss = float("inf")
    best_params = None
    losses = [] if history is None else history

    for iteration in range(optimizer.budget):
        candidate = optimizer.ask()
        x_dict = candidate.value
        # Enforce exact n_embed = n_heads * embed_mult (overwrite before eval)
        if "n_heads" in x_dict and "embed_mult" in x_dict:
            x_dict = {**x_dict, "n_embed": int(x_dict["n_heads"]) * int(x_dict["embed_mult"]) }
        if fixed_params:
            x_dict = {**fixed_params, **x_dict}

        loss_array = train_evaluate_model(x_dict, objective=objective_name)
        loss_value = float(loss_array[0][0])

        optimizer.tell(candidate, loss_value)
        losses.append(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_params = x_dict
            print(f">>> Iter {iteration+1}: NEW BEST {objective_name.upper()} = {best_loss:.4f}")
        else:
            print(f"Iter {iteration+1}: {objective_name} = {loss_value:.4f}")

    return best_loss, best_params, losses


def run_two_phase_optimization(phase1_budget=30, phase2_budget=40):
    print("Starting Phase 1 (minimize EFE proxy)...")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=phase1_budget)
    # Ensure architecture validity: n_embed divisible by n_heads
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    best_efe, best_params_efe, history1 = run_phase(opt1, "efe")

    print("\nPhase 1 done.")
    print(f"Best EFE: {best_efe:.4f}")
    print(f"Best params (arch + cont): {best_params_efe}")

    print("\nStarting Phase 2 (refine continuous, minimize CE)...")
    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_params_efe), budget=phase2_budget)
    # Inoculate optimizer with known good values from Phase 1
    opt2.suggest(**{
        "eta": float(best_params_efe.get("eta", 1e-3)),
        "dropout": float(best_params_efe.get("dropout", 0.1)),
        "wub": float(best_params_efe.get("wub", 0.02)),
        "wlb": float(best_params_efe.get("wlb", -0.02)),
    })
    # Optional: register a callback to log each tell
    def _cb_logger(optimizer, candidate, value):
        try:
            print(f"Callback tell -> loss={value} for {candidate.value}")
        except Exception:
            pass
    opt2.register_callback("tell", _cb_logger)
    best_ce, best_params_ce, history2 = run_phase(opt2, "ce", fixed_params=best_params_efe, history=history1)

    print("\n--- Two-Phase Optimization Finished ---")
    print(f"Phase 1 best EFE: {best_efe:.4f}")
    print(f"Phase 2 best CE:  {best_ce:.4f}")
    if best_ce < float("inf"):
        print(f"Best PPL:       {math.exp(best_ce):.2f}")
    print("Best Params (arch + tuned cont):", best_params_ce)

    # Plot combined history (phase1 + phase2) for CE axis (proxy for efe shown too)
    plt.figure(figsize=(10, 6))
    plot_data = [l if l != float("inf") else 10.0 for l in history2]
    plt.plot(plot_data, marker="o", label="Trial Loss")
    plt.plot(np.minimum.accumulate(plot_data), "r--", label="Best So Far")
    plt.title("Nevergrad Two-Phase Optimization")
    plt.ylabel("Loss (phase1 EFE proxy, phase2 CE)")
    plt.xlabel("Trial")
    plt.legend()
    plt.savefig("nevergrad_two_phase_log.png")


# ------------------------------ Advanced Examples ------------------------------

def run_two_phase_parallel(phase1_budget=30, phase2_budget=40, num_workers=4):
    """Asynchronous parallel evaluation using ProcessPoolExecutor and minimize."""
    print("Starting Phase 1 (async minimize, EFE)...")

    def func_phase1(**x):
        arr = train_evaluate_model(x, objective="efe")
        return float(arr[0][0])

    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=phase1_budget, num_workers=num_workers)
    # Cheap constraint: ensure n_embed % n_heads == 0
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)

    with futures.ProcessPoolExecutor(max_workers=opt1.num_workers) as executor:
        rec1 = opt1.minimize(func_phase1, executor=executor, batch_mode=False)

    best_params_efe = rec1.value

    print("\nStarting Phase 2 (async minimize, CE)...")

    def func_phase2(**x):
        merged = {**best_params_efe, **x}
        arr = train_evaluate_model(merged, objective="ce")
        return float(arr[0][0])

    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_params_efe), budget=phase2_budget, num_workers=num_workers)
    # inoculate
    opt2.suggest(**{
        "eta": float(best_params_efe.get("eta", 1e-3)),
        "dropout": float(best_params_efe.get("dropout", 0.1)),
        "wub": float(best_params_efe.get("wub", 0.02)),
        "wlb": float(best_params_efe.get("wlb", -0.02)),
    })
    with futures.ProcessPoolExecutor(max_workers=opt2.num_workers) as executor:
        rec2 = opt2.minimize(func_phase2, executor=executor, batch_mode=False)

    print("Parallel two-phase finished.")
    print("Best Phase1 params:", best_params_efe)
    print("Best Phase2 params:", rec2.value)


def run_two_phase_with_portfolio(phase1_budget=30, phase2_budget=40):
    """Use PortfolioDiscreteOnePlusOne for mixed discrete space in Phase 1."""
    print("Phase 1 with PortfolioDiscreteOnePlusOne (EFE)...")
    opt1 = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase1_space(), budget=phase1_budget)
    # constraint for divisibility
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    best_efe, best_params_efe, _ = run_phase(opt1, "efe")

    print("\nPhase 2 with NGOpt (CE)...")
    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_params_efe), budget=phase2_budget)
    opt2.suggest(**{
        "eta": float(best_params_efe.get("eta", 1e-3)),
        "dropout": float(best_params_efe.get("dropout", 0.1)),
        "wub": float(best_params_efe.get("wub", 0.02)),
        "wlb": float(best_params_efe.get("wlb", -0.02)),
    })
    best_ce, best_params_ce, _ = run_phase(opt2, "ce", fixed_params=best_params_efe)
    print("Done. Phase1 EFE:", best_efe, "Phase2 CE:", best_ce)


def run_two_phase_with_chaining(phase1_budget=30, phase2_budget=40):
    """Use LHS then DE in Phase 2 via Chaining for refinement."""
    print("Phase 1 with NGOpt (EFE)...")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=phase1_budget)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    _, best_params_efe, _ = run_phase(opt1, "efe")

    print("\nPhase 2 with Chaining(LHS -> DE) (CE)...")
    ChainOpt = ng.optimizers.Chaining([ng.optimizers.LHSSearch, ng.optimizers.DE], [int(phase2_budget * 0.2)])
    opt2 = ChainOpt(parametrization=phase2_space(best_params_efe), budget=phase2_budget)
    best_ce, best_params_ce, _ = run_phase(opt2, "ce", fixed_params=best_params_efe)
    print("Chaining finished. Best CE:", best_ce)


def run_phase2_multiobjective_de(best_params, phase2_budget=40):
    """Multi-objective DE: minimize [CE, |EFE|], show Pareto front."""
    print("Phase 2 multi-objective DE ([CE, |EFE|])...")
    opt = ng.optimizers.DE(parametrization=phase2_space(best_params), budget=phase2_budget)
    # Provide a reference point (upper bounds) for [CE, |EFE|]
    opt.tell(ng.p.MultiobjectiveReference(), [10.0, 10.0])

    for _ in range(phase2_budget):
        cand = opt.ask()
        x = cand.value
        merged = {**best_params, **x}
        # Direct call to wrapper for metrics
        arr_ce = train_evaluate_model(merged, objective="ce")
        ce = float(arr_ce[0][0])
        arr_efe = train_evaluate_model(merged, objective="efe")
        efe = float(arr_efe[0][0])
        opt.tell(cand, [ce, abs(efe)])

    print("Pareto front (params and losses):")
    for p in sorted(opt.pareto_front(), key=lambda c: c.losses):
        print({"params": p.value, "losses": p.losses})


if __name__ == "__main__":
    run_two_phase_optimization()