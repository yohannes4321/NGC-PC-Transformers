import osz
import gc
import sys
import time
import jax
import numpy as np
from experiment_logger import DualLogger, LOG_DIR
from train_tuning import run_training, PruningError

# State tracker for pruning
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
    trial_label = state.trial_count
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"trial_{trial_label}.txt")

    print(f"[TRIAL {trial_label}] Objective: {objective_type}")
    print(f"[TRIAL {trial_label}] Params: {params}")

    # Set threshold based on current objective
    threshold = state.best_efe if objective_type == "efe" else state.best_ce
    # Don't prune if we haven't found a good best yet
    if threshold > 1e9: threshold = None 

    original_stdout = sys.stdout
    logger = DualLogger(log_path)
    sys.stdout = logger

    try:
        clean_memory()
        efe, ce, ppl = run_training(
            params_override=params,
            pruning_threshold=threshold
        )

        # Update global bests if not pruned
        if objective_type == "efe" and efe < state.best_efe:
            state.best_efe = efe
        if objective_type == "ce" and ce < state.best_ce:
            state.best_ce = ce

        return efe, ce, ppl, False

    except PruningError as e:
        print(f"Stopping Trial: {e}")
        return 1e12, 1e12, 1e12, True # Penalty score

    except Exception as e:
        print(f"[SYSTEM FAILURE] {repr(e)}")
        return 1e12, 1e12, 1e12, True

    finally:
        logger.close()
        sys.stdout = original_stdout
        clean_memory()

def evaluate_objective_efe(*args, **kwargs):
    efe, ce, ppl, unstable = _run_trial_internal(args, kwargs, "efe")
    return efe

def evaluate_objective_ce(*args, **kwargs):
    efe, ce, ppl, unstable = _run_trial_internal(args, kwargs, "ce")
    return ce