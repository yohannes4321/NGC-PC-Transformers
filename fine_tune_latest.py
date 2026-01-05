import csv
import jax
import jax.numpy as jnp
from jax import random, clear_caches
import numpy as np
import sys
import gc
import os
import time
import traceback
import random as py_random
import copy
import math
from itertools import count
from pathlib import Path

# --- User Imports ---
from config import Config as config
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from eval import eval_model

# --- 1. Environment & Setup ---
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

LOG_FILE = "search_progress.log"
RESULT_FILE = "tuning_results.txt"
RESULT_CSV = "trial_metrics.csv"
BEST_FILE = "best_trial.txt"
TRIAL_COUNTER = count(1)
BEST_RECORD = {"val_ce": float("inf"), "avg_ppl": None, "avg_efe": None, "avg_ce": None, "params": None, "trial_id": None}

def log_message(message, end="\n"):
    print(message, end=end, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + end)

def clean_memory():
    """Aggressively free JAX/device and host allocations."""
    try:
        clear_caches()
        jax.block_until_ready(jnp.array(0))
    except Exception:
        pass
    gc.collect()

def set_model_eta(model, eta_value):
    """Best-effort setter to propagate eta to major submodules."""
    def _try_set(obj):
        if obj is not None and hasattr(obj, "eta"):
            try:
                obj.eta = eta_value
            except Exception:
                pass

    _try_set(model)
    for attr in ["embedding", "output", "projection"]:
        _try_set(getattr(model, attr, None))
    blocks = getattr(model, "blocks", []) or []
    for blk in blocks:
        _try_set(blk)
        _try_set(getattr(blk, "attention", None))
        _try_set(getattr(blk, "mlp", None))

# --- 2. Custom Evolutionary Algorithm (The "Shim" to replace Alogos/Grammar) ---

class EvoTrial:
    """Mimics Optuna's Trial object but for our Custom GA."""
    def __init__(self, params=None, recording_mode=False):
        self.params = params if params is not None else {}
        self.recording_mode = recording_mode
        self.definitions = {} # Used only during recording phase

    def suggest_int(self, name, low, high, step=1, log=False):
        if self.recording_mode:
            self.definitions[name] = {"type": "int", "low": low, "high": high, "step": step, "log": log}
            return low # Return dummy value during definition
        return self.params[name]

    def suggest_float(self, name, low, high, step=None, log=False):
        if self.recording_mode:
            self.definitions[name] = {"type": "float", "low": low, "high": high, "step": step, "log": log}
            return low
        return self.params[name]

    def suggest_categorical(self, name, choices):
        if self.recording_mode:
            self.definitions[name] = {"type": "cat", "choices": choices}
            return choices[0]
        return self.params[name]

class EvoOptimizer:
    """
    A custom Genetic Algorithm that understands the 'suggest_' API.
    It replaces the need for a Grammar string.
    """
    def __init__(self, objective_func, population_size=6, max_generations=3, mutation_rate=0.3):
        self.objective_func = objective_func
        self.pop_size = population_size
        self.generations = max_generations
        self.mutation_rate = mutation_rate
        self.definitions = {}
        self.population = [] # List of dicts (individual params)
        self.fitness_scores = []

    def _get_random_value(self, name):
        """Generates a random value based on the parameter definition."""
        d = self.definitions[name]
        
        if d['type'] == 'int':
            if d['log']:
                # Log sampling for int
                val = int(math.exp(py_random.uniform(math.log(d['low']), math.log(d['high']))))
            else:
                val = py_random.randint(d['low'], d['high'])
            
            if d['step'] > 1:
                val = d['low'] + round((val - d['low']) / d['step']) * d['step']
            return max(d['low'], min(d['high'], val))

        elif d['type'] == 'float':
            if d['log']:
                 val = math.exp(py_random.uniform(math.log(d['low']), math.log(d['high'])))
            else:
                val = py_random.uniform(d['low'], d['high'])
            
            if d['step'] is not None:
                val = d['low'] + round((val - d['low']) / d['step']) * d['step']
            return max(d['low'], min(d['high'], val))

        elif d['type'] == 'cat':
            return py_random.choice(d['choices'])

    def _mutate(self, params):
        """Mutates a parameter dictionary."""
        new_params = params.copy()
        for name in params:
            if py_random.random() < self.mutation_rate:
                # Re-sample this specific parameter
                new_params[name] = self._get_random_value(name)
        return new_params

    def _crossover(self, p1, p2):
        """Uniform Crossover between two parameter dictionaries."""
        child = {}
        for name in p1:
            if py_random.random() < 0.5:
                child[name] = p1[name]
            else:
                child[name] = p2[name]
        return child

    def run(self):
        # 1. Discovery Phase: Run objective once to learn the search space
        log_message("[EvoOptimizer] Discovering Search Space...")
        probe_trial = EvoTrial(recording_mode=True)
        try:
            self.objective_func(probe_trial)
        except Exception:
            # We expect it might fail because values are dummy, but we only need the definitions
            pass
        
        self.definitions = probe_trial.definitions
        log_message(f"[EvoOptimizer] Found {len(self.definitions)} parameters: {list(self.definitions.keys())}")

        # 2. Initialize Population
        for _ in range(self.pop_size):
            ind = {name: self._get_random_value(name) for name in self.definitions}
            self.population.append(ind)

        # 3. Evolution Loop
        best_overall_params = None
        best_overall_score = float('inf')

        for gen in range(self.generations):
            log_message(f"\n=== GENERATION {gen+1}/{self.generations} ===")
            gen_scores = []
            
            # Evaluate
            for i, individual in enumerate(self.population):
                trial = EvoTrial(params=individual)
                score = self.objective_func(trial)
                gen_scores.append(score)
                
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_params = individual

            # Selection (Tournament)
            new_population = []
            # Elitism: Keep best 1
            best_idx = np.argmin(gen_scores)
            new_population.append(self.population[best_idx])

            while len(new_population) < self.pop_size:
                # Tournament size 3
                candidates = py_random.sample(list(enumerate(self.population)), 3)
                # Sort by score (lower is better for minimization)
                candidates.sort(key=lambda x: gen_scores[x[0]])
                parent1 = candidates[0][1]

                candidates = py_random.sample(list(enumerate(self.population)), 3)
                candidates.sort(key=lambda x: gen_scores[x[0]])
                parent2 = candidates[0][1]
                
                # Crossover
                child = self._crossover(parent1, parent2)
                # Mutation
                child = self._mutate(child)
                new_population.append(child)
            
            self.population = new_population

        return best_overall_params, best_overall_score


# --- 3. Helper Logic ---

def get_dynamic_batch_size(n_embed, block_size):
    """Calculates batch size to maintain constant memory usage."""
    complexity_score = n_embed * block_size
    if complexity_score >= 32000: return 4
    elif complexity_score >= 20000: return 8
    elif complexity_score >= 12000: return 12
    elif complexity_score >= 6000: return 16
    else: return 24

# --- 4. The Objective Function (Now uses Trial object) ---

def objective_function(trial):
    start_time = time.time()
    trial_id = next(TRIAL_COUNTER)

    # --- DEFINE SEARCH SPACE HERE (Optuna Style) ---
    # Model / architecture
    n_embed = trial.suggest_int("n_embed", 64, 512, step=64)
    n_heads = trial.suggest_int("n_heads", 1, 8) # Reduced max to prevent divisibility issues mostly
    n_layers = trial.suggest_int("n_layers", 1, 6)
    block_size = trial.suggest_int("block_size", 64, 256, step=64)
    T = trial.suggest_int("T", 4, 10)

    # Optimization
    eta = trial.suggest_float("eta", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Regularization
    wlb = trial.suggest_float("wlb", -0.05, -0.01)
    wub = trial.suggest_float("wub", 0.01, 0.05)
    
    # Training schedule
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)

    # Predictive coding / dynamics
    tau_m = trial.suggest_float("tau_m", 5.0, 50.0)

    # Activation function
    act_fx = trial.suggest_categorical("act_fx", ["identity", "tanh"])

    # --- LOGIC START ---
    
    # Validation constraints
    if n_embed % n_heads != 0:
        # Auto-correct instead of failing to keep evolution smooth
        n_heads = 1 
        # Or return high penalty:
        # return 1e9

    params_str = str(trial.params) if hasattr(trial, 'params') else "Discovery_Mode"

    log_message(f"\n" + "="*60)
    log_message(f"[Trial {trial_id}] Params: {params_str}")

    curr_batch_size = get_dynamic_batch_size(n_embed, block_size)
    
    # Sync global config
    config.seq_len = block_size
    config.batch_size = curr_batch_size
    config.n_embed = n_embed
    config.n_heads = n_heads
    config.tau_m = tau_m
    config.act_fx = act_fx

    model = None
    try:
        # Reload Data with new Block Size
        loader = DataLoader(seq_len=block_size, batch_size=curr_batch_size)
        train_loader, valid_loader, _ = loader.load_and_prepare_data()
        clean_memory()
        
        # Initialize Model
        dkey = random.PRNGKey(int(time.time()))
        model = NGCTransformer(
            dkey, 
            batch_size=curr_batch_size, 
            seq_len=block_size, 
            n_embed=n_embed,
            vocab_size=config.vocab_size, 
            n_layers=n_layers, 
            n_heads=n_heads,
            T=T, 
            dt=1.0, 
            tau_m=tau_m, 
            act_fx=act_fx, 
            eta=eta,
            dropout_rate=dropout, 
            pos_learnable=config.pos_learnable,
            optim_type=config.optim_type,
            wub=wub, 
            wlb=wlb, 
            exp_dir="exp",
            model_name="ngc_evo_temp"
        )

        # Training Loop
        total_efe = 0.0
        total_ppl = 0.0
        total_ce = 0.0
        batch_count = 0
        max_batches_per_trial = 10
        warmup_steps = max(1, warmup_epochs)
        warmup_factors = np.linspace(0.1, 1.0, warmup_steps)
        
        train_iter = iter(train_loader)
        
        for i in range(max_batches_per_trial):
            try:
                batch = next(train_iter)
            except StopIteration:
                break

            # Linear warmup on eta
            if i < warmup_steps:
                eta_scale = float(warmup_factors[i])
            else:
                eta_scale = 1.0
            effective_eta = eta * eta_scale
            set_model_eta(model, effective_eta)

            inputs = batch[0][1] 
            targets = batch[1][1]
            targets_onehot = jnp.eye(config.vocab_size)[targets]
            targets_flat = targets_onehot.reshape(-1, config.vocab_size)
            
            # Forward + Update
            yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            y_pred = yMu_inf.reshape(-1, config.vocab_size)
            
            # NaN Check
            if jnp.isnan(_EFE) or jnp.isnan(y_pred).any():
                log_message(f" [!] Trial Killed: NaN detected at batch {i}")
                return 1e9 
                
            batch_ce = float(measure_CatNLL(y_pred, targets_flat).mean())
            batch_ppl = float(np.exp(batch_ce))
            batch_efe = float(_EFE)
            
            total_efe += batch_efe
            total_ppl += batch_ppl
            total_ce += batch_ce
            batch_count += 1
            clean_memory()

        # Calculate Averages
        avg_efe = total_efe / batch_count if batch_count > 0 else 0
        avg_ppl = total_ppl / batch_count if batch_count > 0 else 0
        avg_ce = total_ce / batch_count if batch_count > 0 else 0
        eval_time = time.time() - start_time

        # Validation (The Fitness Function)
        val_ce, _ = eval_model(model, valid_loader, config.vocab_size)
        log_message(f"[Result] Validation Score (CE): {val_ce:.4f}")

        # Record CSV
        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_id, "ok", params_str, 
                f"{avg_efe:.4f}", f"{avg_ppl:.4f}", f"{avg_ce:.4f}", 
                f"{val_ce:.4f}", f"{eval_time:.2f}"
            ])

        # Track Best
        if val_ce < BEST_RECORD["val_ce"]:
            BEST_RECORD.update({
                "val_ce": val_ce,
                "avg_ppl": avg_ppl,
                "avg_efe": avg_efe,
                "avg_ce": avg_ce,
                "params": params_str,
                "trial_id": trial_id
            })
            with open(BEST_FILE, "w") as f:
                f.write(f"Best Trial ID: {trial_id}\nParams: {params_str}\nVal CE: {val_ce}")

        return float(val_ce)

    except KeyboardInterrupt:
        return 1e9
    except Exception as e:
        log_message(f"\n[!] ERROR in Trial: {e}")
        return 1e9 
    finally:
        if model: del model
        clean_memory()

# --- 5. Execution ---
def main():
    # Clear logs
    with open(LOG_FILE, "w") as f: f.write("=== New Custom Evo Session ===\n")
    if not Path(RESULT_CSV).exists():
        with open(RESULT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_id", "status", "params", "avg_efe", "avg_ppl", "avg_ce", "val_ce", "eval_time"])

    # Instantiate Custom Evolutionary Optimizer
    optimizer = EvoOptimizer(
        objective_func=objective_function,
        population_size=6,
        max_generations=3,
        mutation_rate=0.3
    )

    log_message("\n" + "#"*50)
    log_message("STARTING EVOLUTIONARY OPTIMIZATION (OPTUNA STYLE)")
    log_message("#"*50)

    try:
        best_params, best_score = optimizer.run()
        
        log_message("\n" + "#"*50)
        log_message("OPTIMIZATION COMPLETED")
        log_message(f"Best Params: {best_params}")
        log_message(f"Best Score: {best_score}")
        log_message("#"*50)
            
    except KeyboardInterrupt:
        print("\nOptimization stopped manually.")

if __name__ == "__main__":
    main()