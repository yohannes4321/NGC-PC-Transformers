import os
import gc
import sys
import jax
from train_tuning import run_training, PruningError

# Global state tracker
class StudyState:
    best_efe = float('inf')
    best_ce = float('inf')
    trial_count = 0

state = StudyState()

def clean_memory():
    gc.collect()
    try: jax.clear_caches()
    except: pass

def _prepare_params(args, kwargs):
    raw = args[0] if (args and isinstance(args[0], dict)) else kwargs
    params = dict(raw)
    int_keys = {"n_layers", "n_heads", "embed_mult", "batch_size", "seq_len", "tau_m", "n_iter"}
    for k in int_keys:
        if k in params: params[k] = int(params[k])
    if "n_heads" in params and "embed_mult" in params:
        params["n_embed"] = params["n_heads"] * params["embed_mult"]
    return params

def _run_trial_internal(args, kwargs, objective_type="efe"):
    params = _prepare_params(args, kwargs)
    state.trial_count += 1
    print(f"\n[TRIAL {state.trial_count}] Params: {params}", flush=True)
    
    # Logic for pruning
    threshold = state.best_efe if objective_type == "efe" else state.best_ce
    if threshold > 1e9: threshold = None 

    try:
        clean_memory()
        efe, ce, ppl = run_training(
            params_override=params,
            pruning_threshold=threshold
        )

        if objective_type == "efe" and efe < state.best_efe:
            state.best_efe = efe
        if objective_type == "ce" and ce < state.best_ce:
            state.best_ce = ce

        return efe, ce, ppl

    except PruningError as e:
        print(f"Trial {state.trial_count} Pruned: {e}")
        return 1e12, 1e12, 1e12

    except Exception as e:
        print(f"[TRIAL {state.trial_count} FAILURE] {repr(e)}")
        return 1e12, 1e12, 1e12
    finally:
        clean_memory()

def evaluate_objective_efe(*args, **kwargs):
    efe, _, _ = _run_trial_internal(args, kwargs, "efe")
    return efe

def evaluate_objective_ce(*args, **kwargs):
    _, ce, _ = _run_trial_internal(args, kwargs, "ce")
    return ce