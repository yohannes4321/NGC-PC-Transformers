"""Nevergrad HPO entrypoint."""
import os
import math
from concurrent import futures
import numpy as np
import nevergrad as ng
from config import Config as config
from trainer_wrapper import train_evaluate_model

# ==============================================================================
# 1. EVALUATION FUNCTIONS
# ==============================================================================

def _extract_params_from_call(args, kwargs):
    """Nevergrad may call with positional or keyword args. Normalize to dict.
    - If a single positional arg has `.value`, use that.
    - If a single positional arg is a dict, use it.
    - Else, use kwargs.
    """
    if len(args) == 1:
        a0 = args[0]
        if hasattr(a0, "value"):
            try:
                return dict(a0.value)
            except Exception:
                pass
        if isinstance(a0, dict):
            return dict(a0)
    return dict(kwargs)


def parallel_func_phase1(*args, **x):
    """Top-level function for Phase 1 evaluations (picklable).
    Ensures integer parameters and computes n_embed when needed.
    """
    params = _extract_params_from_call(args, x)
    # Concrete ints for JAX
    for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "n_layers", "tau_m", "n_iter"]:
        if k in params:
            params[k] = int(params[k])
    # Divisibility/n_embed
    if "n_heads" in params and "embed_mult" in params:
        params["n_embed"] = int(params["n_heads"]) * int(params["embed_mult"])
    try:
        arr = train_evaluate_model(params, objective="efe")
        return float(arr[0][0])
    except Exception:
        return float("inf")

def parallel_func_phase2(*args, **x):
    """Top-level function for Phase 2 evaluations (picklable)."""
    params = _extract_params_from_call(args, x)
    for k in ["n_heads", "embed_mult", "batch_size", "seq_len", "n_layers", "tau_m", "n_iter"]:
        if k in params:
            params[k] = int(params[k])
    if "n_heads" in params and "embed_mult" in params:
        params["n_embed"] = int(params["n_heads"]) * int(params["embed_mult"])
    try:
        arr = train_evaluate_model(params, objective="ce")
        return float(arr[0][0])
    except Exception:
        return float("inf")

def constraint_embed_divisible(x):
    """Cheap constraint: ensure n_embed is divisible by n_heads."""
    try:
        return int(x.get("n_embed", 0)) % int(x.get("n_heads", 1)) == 0
    except Exception:
        return False

# ==============================================================================
# 2. SEARCH SPACES
# ==============================================================================

def phase1_space():
    """Architecture search space constrained to prevent JAX tracer/reshape errors."""
    return ng.p.Dict(
        n_heads    = ng.p.Choice([2, 4, 8]),
        embed_mult = ng.p.Choice([8, 16, 24]),
        batch_size = ng.p.Choice([16, 32, 64]),
        seq_len    = ng.p.Choice([16, 32, 48]),
       
        # Stability-oriented bounds
        eta=ng.p.Log(lower=1e-7, upper=5e-5),
        tau_m=ng.p.Scalar(lower=5, upper=15).set_integer_casting(),
        n_iter=ng.p.Scalar(lower=1, upper=10).set_integer_casting(),
        wub=ng.p.Scalar(lower=0.01, upper=0.04),
        wlb=ng.p.Scalar(lower=-0.04, upper=-0.01),
        dropout_rate=ng.p.Constant(0.0),
        optim_type=ng.p.Choice(["adam", "sgd"]),
        act_fx=ng.p.Choice(["identity", "relu"]),
    )

def phase2_space(best):
    eta_best = float(best.get("eta", 1e-5))
    wub_best = float(best.get("wub", 0.01))
    wlb_best = float(best.get("wlb", -0.01))
    return ng.p.Dict(
        eta=ng.p.Log(lower=eta_best * 0.5, upper=min(eta_best * 2.0, 1e-4)),
        wub=ng.p.Scalar(lower=max(0.001, wub_best - 0.005), upper=min(0.04, wub_best + 0.005)),
        wlb=ng.p.Scalar(lower=max(-0.04, wlb_best - 0.005), upper=min(-0.001, wlb_best + 0.005)),
        dropout_rate=ng.p.Scalar(lower=0.0, upper=0.2),
    )

# ==============================================================================
# 3. GENERIC RUNNER
# ==============================================================================

def run_phase(optimizer, objective_name, fixed_params=None, history=None):
    best_loss = float("inf")
    best_params = None
    losses = [] if history is None else history
    window = []
    
    if objective_name == "efe":
        plateau_window = getattr(config, "phase1_plateau_window", 3)
        plateau_min_delta = getattr(config, "phase1_plateau_min_delta", 1.0)
        plateau_warmup = getattr(config, "phase1_plateau_warmup", 2)
    else:
        plateau_window = getattr(config, "phase2_plateau_window", 3)
        plateau_min_delta = getattr(config, "phase2_plateau_min_delta", 0.01)
        plateau_warmup = getattr(config, "phase2_plateau_warmup", 2)

    for iteration in range(optimizer.budget):
        candidate = optimizer.ask()
        x_dict = candidate.value
        full_params = {**fixed_params, **x_dict} if fixed_params else x_dict.copy()
        
        # Divisibility fix
        if "n_heads" in full_params and "embed_mult" in full_params:
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
                f"\nTrial {iteration} | Heads: {full_params.get('n_heads')} | D_Model: {full_params.get('n_embed')} | Seq: {full_params.get('seq_len')}"
            )
            loss_array = train_evaluate_model(full_params, objective=objective_name)
            loss_value = float(loss_array[0][0])
            if np.isnan(loss_value):
                loss_value = float("inf")
        except Exception as e:
            print(f"!!! CRASH IN TRIAL {iteration} !!! Error: {e}")
            loss_value = float("inf")

        optimizer.tell(candidate, loss_value)
        losses.append(loss_value)
        window.append(loss_value)
        if len(window) > plateau_window:
            window.pop(0)

        if loss_value < best_loss:
            best_loss = loss_value
            best_params = full_params
            print(f">>> NEW BEST {objective_name.upper()} = {best_loss:.4f}")

        if iteration + 1 >= plateau_warmup and len(window) == plateau_window:
            span = max(window) - min(window)
            if span < plateau_min_delta:
                print(
                    f"Early stop {objective_name.upper()} phase: last {plateau_window} trials Δ={span:.4f} < {plateau_min_delta}; moving on."
                )
                break
                
    return best_loss, best_params, losses

# ==============================================================================
# 4. STRATEGIES
# ==============================================================================

def run_two_phase_parallel(phase1_budget=3, phase2_budget=2, num_workers=4):
    """Strategy 1: Asynchronous parallel evaluation (Maximize speed)."""
    print("--- [Parallel Strategy] Starting Phase 1 (async minimize, EFE) ---")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=phase1_budget, num_workers=num_workers)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    
    with futures.ProcessPoolExecutor(max_workers=opt1.num_workers) as executor:
        rec1 = opt1.minimize(parallel_func_phase1, executor=executor, batch_mode=False)
        
    best_params_efe = rec1.value
    best_efe_loss = rec1.loss if hasattr(rec1, "loss") else float("inf")
    
    print("\n--- [Parallel Strategy] Starting Phase 2 (async minimize, CE) ---")
    # Build a parametrization with fixed architecture as Constants for pickling
    eta_best = float(best_params_efe.get("eta", 1e-5))
    wub_best = float(best_params_efe.get("wub", 0.01))
    wlb_best = float(best_params_efe.get("wlb", -0.01))
    
    # Calculate derived n_embed for fixing
    n_h = int(best_params_efe.get("n_heads", 1))
    e_mult = int(best_params_efe.get("embed_mult", int(best_params_efe.get("n_embed", 0)) // n_h))
    
    fixed = {
        "n_heads": ng.p.Constant(n_h),
        "embed_mult": ng.p.Constant(e_mult),
        "batch_size": ng.p.Constant(int(best_params_efe.get("batch_size", 32))),
        "seq_len": ng.p.Constant(int(best_params_efe.get("seq_len", 32))),
        "tau_m": ng.p.Constant(int(best_params_efe.get("tau_m", 10))),
        "n_iter": ng.p.Constant(int(best_params_efe.get("n_iter", 6))),
        "optim_type": ng.p.Constant(best_params_efe.get("optim_type", "adam")),
        "act_fx": ng.p.Constant(best_params_efe.get("act_fx", "relu")),
    }
    search = {
        "eta": ng.p.Log(lower=eta_best * 0.5, upper=min(eta_best * 2.0, 1e-4)),
        "wub": ng.p.Scalar(lower=max(0.001, wub_best - 0.005), upper=min(0.04, wub_best + 0.005)),
        "wlb": ng.p.Scalar(lower=max(-0.04, wlb_best - 0.005), upper=min(-0.001, wlb_best + 0.005)),
        "dropout_rate": ng.p.Scalar(lower=0.0, upper=0.2),
    }
    
    param2 = ng.p.Dict(**fixed, **search)
    opt2 = ng.optimizers.NGOpt(parametrization=param2, budget=phase2_budget, num_workers=num_workers)
    
    try:
        warm = opt2.parametrization.spawn_child(
            new_value={
                "eta": float(best_params_efe.get("eta", 1e-3)),
                "dropout_rate": float(best_params_efe.get("dropout_rate", 0.1)),
                "wub": float(best_params_efe.get("wub", 0.02)),
                "wlb": float(best_params_efe.get("wlb", -0.02)),
            }
        )
        opt2.suggest(warm)
    except Exception as e:
        print("Warm-start suggest failed (parallel Phase 2):", e)

    with futures.ProcessPoolExecutor(max_workers=opt2.num_workers) as executor:
        rec2 = opt2.minimize(parallel_func_phase2, executor=executor, batch_mode=False)
        
    best_ce_loss = rec2.loss if hasattr(rec2, "loss") else float("inf")
    
    return {
        "label": "parallel",
        "phase1_params": best_params_efe,
        "phase1_loss": best_efe_loss,
        "phase2_params": rec2.value,
        "phase2_loss": best_ce_loss,
    }

def run_two_phase_with_portfolio(phase1_budget=30, phase2_budget=40):
    """Strategy 2: PortfolioDiscreteOnePlusOne for mixed discrete space."""
    print("--- [Portfolio Strategy] Phase 1 with PortfolioDiscreteOnePlusOne (EFE) ---")
    opt1 = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=phase1_space(), budget=phase1_budget)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    best_efe, best_params_efe, _ = run_phase(opt1, "efe")
    
    print("\n--- [Portfolio Strategy] Phase 2 with NGOpt (CE) ---")
    opt2 = ng.optimizers.NGOpt(parametrization=phase2_space(best_params_efe), budget=phase2_budget)
    
    try:
        warm = opt2.parametrization.spawn_child(
            new_value={
                "eta": float(best_params_efe.get("eta", 1e-3)),
                "dropout_rate": float(best_params_efe.get("dropout_rate", 0.1)),
                "wub": float(best_params_efe.get("wub", 0.02)),
                "wlb": float(best_params_efe.get("wlb", -0.02)),
            }
        )
        opt2.suggest(warm)
    except Exception as e:
        print("Warm-start suggest failed (portfolio Phase 2):", e)
        
    best_ce, best_params_ce, _ = run_phase(opt2, "ce", fixed_params=best_params_efe)
    return {
        "label": "portfolio",
        "phase1_params": best_params_efe,
        "phase1_loss": best_efe,
        "phase2_params": best_params_ce,
        "phase2_loss": best_ce,
    }

def run_two_phase_with_chaining(phase1_budget=30, phase2_budget=40):
    """Strategy 3: LHS -> DE via Chaining (Better Global Search)."""
    print("--- [Chaining Strategy] Phase 1 with NGOpt (EFE) ---")
    opt1 = ng.optimizers.NGOpt(parametrization=phase1_space(), budget=phase1_budget)
    opt1.parametrization.register_cheap_constraint(constraint_embed_divisible)
    _, best_params_efe, _ = run_phase(opt1, "efe")
    
    print("\n--- [Chaining Strategy] Phase 2 with Chaining(LHS -> DE) (CE) ---")
    # 20% budget for LHS exploration, rest for DE exploitation
    ChainOpt = ng.optimizers.Chaining([ng.optimizers.LHSSearch, ng.optimizers.DE], [int(phase2_budget * 0.2)])
    opt2 = ChainOpt(parametrization=phase2_space(best_params_efe), budget=phase2_budget)
    
    best_ce, best_params_ce, _ = run_phase(opt2, "ce", fixed_params=best_params_efe)
    return {
        "label": "chaining",
        "phase1_params": best_params_efe,
        "phase1_loss": None,
        "phase2_params": best_params_ce,
        "phase2_loss": best_ce,
    }

def run_phase2_multiobjective_de(best_params, phase2_budget=40):
    """Final Step: Multi-objective DE minimizing [CE, |EFE|] to find Pareto front."""
    print(f"\n--- [Multi-Objective Final] Running MO-DE on best architecture ---")
    
    # We create a new search space around the *best* parameters found so far
    # to fine-tune locally while balancing objectives
    opt = ng.optimizers.DE(parametrization=phase2_space(best_params), budget=phase2_budget)
    
    # Set reference point for hypervolume calculation (optional but good practice)
    opt.tell(ng.p.MultiobjectiveReference(), [10.0, 10.0])
    
    pareto_results = []
    
    for i in range(phase2_budget):
        cand = opt.ask()
        x = cand.value
        merged = {**best_params, **x}
        
        # Calculate CE
        try:
            arr_ce = train_evaluate_model(merged, objective="ce")
            ce = float(arr_ce[0][0])
        except:
            ce = float("inf")
            
        # Calculate EFE
        try:
            arr_efe = train_evaluate_model(merged, objective="efe")
            efe = float(arr_efe[0][0])
        except:
            efe = float("inf")

        print(f"MO-DE Trial {i}: CE={ce:.4f}, |EFE|={abs(efe):.4f}")
        opt.tell(cand, [ce, abs(efe)])
        
    pareto = sorted(opt.pareto_front(), key=lambda c: c.losses)
    print("\nPareto front (params and losses):")
    for p in pareto:
        print({"params": p.value, "losses": p.losses})
        pareto_results.append({"params": p.value, "losses": p.losses})
        
    return pareto_results

# ==============================================================================
# 5. ADVANCED ORCHESTRATOR
# ==============================================================================

def run_advanced(phase1_budget=30, phase2_budget=40, num_workers=4):
    """
    Runs all advanced strategies sequentially (without try-except suppression)
    to find the absolute best combination of speed and results.
    """
    results = []

    # 1. Run Parallel (Fastest)
    res_par = run_two_phase_parallel(phase1_budget, phase2_budget, num_workers)
    results.append(res_par)
    
    # 2. Run Portfolio (Best for Discrete/Choices)
    res_port = run_two_phase_with_portfolio(phase1_budget, phase2_budget)
    results.append(res_port)

    # 3. Run Chaining (Best for escaping local minima)
    res_chain = run_two_phase_with_chaining(phase1_budget, phase2_budget)
    results.append(res_chain)

    # Filter out failed runs (loss is None or inf)
    valid_results = [r for r in results if r["phase2_loss"] is not None and r["phase2_loss"] != float("inf")]

    if not valid_results:
        print("All strategies failed to produce a valid loss.")
        return

    # Select best strategy
    best_result = min(valid_results, key=lambda r: r["phase2_loss"])
    
    print("\n=============================================")
    print("WINNING STRATEGY:", best_result["label"].upper())
    print(f"Final Loss (CE): {best_result['phase2_loss']:.5f}")
    print("=============================================")

    # 4. Final Polish: Run Multi-Objective optimization on the winning parameters
    # This combines the 'speed' of previous runs with the 'best result' of MO-DE
    run_phase2_multiobjective_de(best_result["phase2_params"], phase2_budget)

if __name__ == "__main__":
    # Basic logic to switch between standard and advanced mode
    default_case = getattr(config, "case_nevergrad", 2) # Default to 2 for this script
    case = int(os.environ.get("HPO_CASE", str(default_case)))
    
    p1_b = getattr(config, "p1_budget", 30)
    p2_b = getattr(config, "p2_budget", 40)
    
    if case == 1:
        # Standard run (not requested, but kept for legacy)
        run_phase(ng.optimizers.NGOpt(parametrization=phase1_space(), budget=p1_b), "efe")
    elif case == 2:
        # The requested Advanced Function
        run_advanced(phase1_budget=p1_b, phase2_budget=p2_b, num_workers=4)
    else:
        print(f"Unknown case {case}")