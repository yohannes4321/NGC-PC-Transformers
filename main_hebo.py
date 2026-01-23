"""Nevergrad HPO entrypoint."""

import os
import math
from concurrent import futures
import numpy as np
import nevergrad as ng
from config import Config as config
from trainer_wrapper import train_evaluate_model


def constraint_embed_divisible(x):
    """Cheap constraint: ensure n_embed is divisible by n_heads."""
    try:
        return int(x.get("n_embed", 0)) % int(x.get("n_heads", 1)) == 0
    except Exception:
        return False


def phase1_space():
    """Architecture search space constrained to prevent JAX tracer/reshape errors."""
    return ng.p.Dict(
        n_heads=ng.p.Choice([2, 4, 8]),
        embed_mult=ng.p.Choice([8, 16, 32]),
        batch_size=ng.p.Choice([16, 32, 64]),
        seq_len=ng.p.Choice([16, 32, 64]),
        n_layers=ng.p.Choice([1, 2, 4]),
        pos_learnable=ng.p.Choice([True, False]),
        # Stability-oriented bounds
        eta=ng.p.Log(lower=1e-7, upper=5e-5),
        tau_m=ng.p.Scalar(lower=5, upper=15).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=10).set_integer_casting(),
        wub=ng.p.Scalar(lower=0.01, upper=0.02),
        wlb=ng.p.Scalar(lower=-0.02, upper=-0.01),
        dropout_rate=ng.p.Constant(0.0),
        optim_type=ng.p.Choice(["adam", "sgd"]),
        act_fx=ng.p.Choice(["identity", "relu"]),
    )


def phase2_space(best):
    eta_best = float(best.get("eta"))
    wub_best = float(best.get("wub"))
    wlb_best = float(best.get("wlb"))

    return ng.p.Dict(
        eta=ng.p.Log(lower=eta_best * 0.5, upper=eta_best * 2.0),
        wub=ng.p.Scalar(lower=wub_best - 0.005, upper= wub_best + 0.005),
        wlb=ng.p.Scalar(lower=wlb_best - 0.005, upper= wlb_best + 0.005),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.2),
    )


def run_phase(optimizer, objective_name, fixed_params=None, history=None):
    best_loss = float("inf")
    best_params = None
    losses = [] if history is None else history

    for iteration in range(optimizer.budget):
        candidate = optimizer.ask()
        x_dict = candidate.value

        full_params = {**fixed_params, **x_dict} if fixed_params else x_dict.copy()

        # Divisibility fix
        h = int(full_params["n_heads"])
        m = int(full_params["embed_mult"])
        full_params["n_embed"] = h * m

        # Concrete ints for JAX
        int_keys = ["n_heads", "n_embed", "batch_size", "seq_len", "n_layers", "tau_m", "n_iter"]
        for k in int_keys:
            if k in full_params:
                full_params[k] = int(full_params[k])

        try:
            print(
                f"\nTrial {iteration} | Heads: {full_params['n_heads']} | D_Model: {full_params['n_embed']} | Seq: {full_params['seq_len']}"
            )
            loss_array = train_evaluate_model(full_params, objective=objective_name)
            loss_value = float(loss_array[0][0])
            print ("**********",loss_value)
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

    return best_loss, best_params, losses #


def run_two_phase_optimization(p1_budget=config.p1_budget, p2_budget=config.p2_budget):
    print("--- Phase 1: Arch Search (EFE) ---")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=p1_budget)
    best_efe, best_arch, history1 = run_phase(opt1, "efe")

    if best_arch is None:
        print("Search failed.")
        return

    print("\n--- Phase 2: Hyperparam Refinement (CE) ---")
    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_arch), budget=p2_budget)
    try:
        warm = opt2.parametrization.spawn_child(
            new_value={
                "eta": float(best_arch.get("eta")),
                "wub": float(best_arch.get("wub")),
                "wlb": float(best_arch.get("wlb")),
                "dropout_rate":float(best_arch.get("dropout_rate"))

            }
        )
        opt2.suggest(warm)
    except Exception as e:
        print("Warm-start suggest failed (Phase 2):", e)

    best_ce, best_final, history2 = run_phase(opt2, "ce", fixed_params=best_arch, history=history1)

    print("\nOptimization Finished Successfully!")
    print(f"Final Architecture: Heads={best_final['n_heads']}, D_Model={best_final['n_embed']}")
    print(f"Final Params: Batch={best_final['batch_size']}, Seq={best_final['seq_len']}")
    print(f"Final Loss (CE): {best_ce:.4f}")


def run_two_phase_parallel(phase1_budget=3, phase2_budget=2, num_workers=4):
    """Asynchronous parallel evaluation using ProcessPoolExecutor and minimize."""
    print("Starting Phase 1 (async minimize, EFE)...")

    def func_phase1(**x):
        arr = train_evaluate_model(x, objective="efe")
        return float(arr[0][0])

    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=phase1_budget, num_workers=num_workers)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)

    with futures.ProcessPoolExecutor(max_workers=opt1.num_workers) as executor:
        rec1 = opt1.minimize(func_phase1, executor=executor, batch_mode=False)

    best_params_efe = rec1.value
    best_efe_loss = rec1.loss if hasattr(rec1, "loss") else None

    print("\nStarting Phase 2 (async minimize, CE)...")

    def func_phase2(**x):
        merged = {**best_params_efe, **x}
        arr = train_evaluate_model(merged, objective="ce")
        return float(arr[0][0])

    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_params_efe), budget=phase2_budget, num_workers=num_workers)
    try:
        warm = opt2.parametrization.spawn_child(
            new_value={
                "eta": float(best_params_efe.get("eta", 1e-3)),
                "dropout": float(best_params_efe.get("dropout", 0.1)),
                "wub": float(best_params_efe.get("wub", 0.02)),
                "wlb": float(best_params_efe.get("wlb", -0.02)),
            }
        )
        opt2.suggest(warm)
    except Exception as e:
        print("Warm-start suggest failed (parallel Phase 2):", e)

    with futures.ProcessPoolExecutor(max_workers=opt2.num_workers) as executor:
        rec2 = opt2.minimize(func_phase2, executor=executor, batch_mode=False)

    best_ce_loss = rec2.loss if hasattr(rec2, "loss") else None

    print("Parallel two-phase finished.")
    print("Best Phase1 params:", best_params_efe)
    print("Best Phase2 params:", rec2.value)
    return {
        "label": "parallel",
        "phase1_params": best_params_efe,
        "phase1_loss": best_efe_loss,
        "phase2_params": rec2.value,
        "phase2_loss": best_ce_loss,
    }


def run_two_phase_with_portfolio(phase1_budget=30, phase2_budget=40):
    """Use PortfolioDiscreteOnePlusOne for mixed discrete space in Phase 1."""
    print("Phase 1 with PortfolioDiscreteOnePlusOne (EFE)...")
    opt1 = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase1_space(), budget=phase1_budget)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    best_efe, best_params_efe, _ = run_phase(opt1, "efe")

    print("\nPhase 2 with NGOpt (CE)...")
    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_params_efe), budget=phase2_budget)
    try:
        warm = opt2.parametrization.spawn_child(
            new_value={
                "eta": float(best_params_efe.get("eta", 1e-3)),
                "dropout": float(best_params_efe.get("dropout", 0.1)),
                "wub": float(best_params_efe.get("wub", 0.02)),
                "wlb": float(best_params_efe.get("wlb", -0.02)),
            }
        )
        opt2.suggest(warm)
    except Exception as e:
        print("Warm-start suggest failed (portfolio Phase 2):", e)

    best_ce, best_params_ce, _ = run_phase(opt2, "ce", fixed_params=best_params_efe)
    print("Done. Phase1 EFE:", best_efe, "Phase2 CE:", best_ce)
    return {
        "label": "portfolio",
        "phase1_params": best_params_efe,
        "phase1_loss": best_efe,
        "phase2_params": best_params_ce,
        "phase2_loss": best_ce,
    }


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
    return {
        "label": "chaining",
        "phase1_params": best_params_efe,
        "phase1_loss": None,
        "phase2_params": best_params_ce,
        "phase2_loss": best_ce,
    }


def run_phase2_multiobjective_de(best_params, phase2_budget=40):
    """Multi-objective DE: minimize [CE, |EFE|], show Pareto front."""
    print("Phase 2 multi-objective DE ([CE, |EFE|])...")
    opt = ng.optimizers.DE(parametrization=phase2_space(best_params), budget=phase2_budget)
    opt.tell(ng.p.MultiobjectiveReference(), [10.0, 10.0])

    for _ in range(phase2_budget):
        cand = opt.ask()
        x = cand.value
        merged = {**best_params, **x}
        arr_ce = train_evaluate_model(merged, objective="ce")
        ce = float(arr_ce[0][0])
        arr_efe = train_evaluate_model(merged, objective="efe")
        efe = float(arr_efe[0][0])
        opt.tell(cand, [ce, abs(efe)])

    pareto = sorted(opt.pareto_front(), key=lambda c: c.losses)
    print("Pareto front (params and losses):")
    for p in pareto:
        print({"params": p.value, "losses": p.losses})
    return pareto


def run_advanced(phase1_budget=30, phase2_budget=40, num_workers=4):
    """Run a portfolio of advanced strategies and summarize the best result."""
    results = []

    try:
        results.append(run_two_phase_parallel(phase1_budget, phase2_budget, num_workers))
    except Exception as e:
        print("Parallel strategy failed:", e)

    try:
        results.append(run_two_phase_with_portfolio(phase1_budget, phase2_budget))
    except Exception as e:
        print("Portfolio strategy failed:", e)

    try:
        results.append(run_two_phase_with_chaining(phase1_budget, phase2_budget))
    except Exception as e:
        print("Chaining strategy failed:", e)

    scored = [r for r in results if r and r.get("phase2_loss") is not None]
    best = min(scored, key=lambda r: r["phase2_loss"]) if scored else None

    if best:
        print("\nBest single-strategy CE:", best["phase2_loss"], "from", best["label"])
        pareto = run_phase2_multiobjective_de(best["phase2_params"], phase2_budget)
        return {"strategies": results, "best": best, "pareto": pareto}

    print("No advanced strategy produced a valid result.")
    return {"strategies": results, "best": None, "pareto": None}


if __name__ == "__main__":
    default_case = getattr(config, "case_nevergrad", 1)
    case = int(os.environ.get("HPO_CASE", str(default_case)))
    if case == 1:
        run_two_phase_optimization()
    elif case == 2:
        run_advanced()
    else:
        print(f"Unknown case {case}; defaulting to case 1 (basic run)")
        run_two_phase_optimization()
        # finished