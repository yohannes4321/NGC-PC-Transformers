# filename: main_nevergrad.py
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
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.5)
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
#### advanced way 

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
    best_efe_loss = rec1.loss if hasattr(rec1, "loss") else None

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

    pareto = sorted(opt.pareto_front(), key=lambda c: c.losses)
    print("Pareto front (params and losses):")
    for p in pareto:
        print({"params": p.value, "losses": p.losses})
    return pareto


def run_advanced(phase1_budget=30, phase2_budget=40, num_workers=4):
    """Run a portfolio of advanced strategies and summarize the best result."""
    results = []

    # 1) Parallel async minimize
    try:
        results.append(run_two_phase_parallel(phase1_budget, phase2_budget, num_workers))
    except Exception as e:
        print("Parallel strategy failed:", e)

    # 2) Portfolio discrete + NGOpt
    try:
        results.append(run_two_phase_with_portfolio(phase1_budget, phase2_budget))
    except Exception as e:
        print("Portfolio strategy failed:", e)

    # 3) Chaining LHS -> DE
    try:
        results.append(run_two_phase_with_chaining(phase1_budget, phase2_budget))
    except Exception as e:
        print("Chaining strategy failed:", e)

    # Pick best CE among strategies that reported it
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