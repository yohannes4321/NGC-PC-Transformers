import os
import sys
import warnings
import logging
import optuna

warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('optuna').setLevel(logging.WARNING)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3' 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


import time
import jax
import jax.numpy as jnp
import jax.random as random
from pathlib import Path
from model import NGCTransformer
from data_preprocess.data_loader import DataLoader
from config import Config as base_config
import gc
from eval import eval_model

EFE_STABILITY_THRESHOLD = 2e1


def define_search_space(trial):
    # Include smaller hyperparameter ranges to explore lower-EFE regimes.
    n_heads = trial.suggest_int("n_heads", 1, 12)
    embed_mult = trial.suggest_int("embed_mult", 1, 16)
    n_embed = n_heads * embed_mult
    batch_size = trial.suggest_int("batch_size", 1, 16)
    seq_len = trial.suggest_int("seq_len", 4, 64)

    return {
        "n_layers": trial.suggest_int("n_layers", 1, 6),
        "pos_learnable": trial.suggest_categorical("pos_learnable", [True, False]),
        "eta": trial.suggest_float("eta", 1e-7, 5e-4, log=True),
        "tau_m": trial.suggest_int("tau_m", 5, 40),
        "n_iter": trial.suggest_int("n_iter", 1, 50),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.25),
        "wub": trial.suggest_float("wub", 0.0, 0.2),
        "wlb": trial.suggest_float("wlb", -0.2, -0.001),
        "optim_type": trial.suggest_categorical("optim_type", ["adam", "sgd"]),
        "act_fx": trial.suggest_categorical("act_fx", ["identity", "relu"]),
        "n_heads": n_heads,
        "n_embed": n_embed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_mult": embed_mult
    }
    

def create_model_with_all_params(trial_number, params, cfg):
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()
    dkey = random.PRNGKey(trial_number * 1000 + 42)

    model_args = {
        "dkey": dkey,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "n_embed": cfg.n_embed,
        "vocab_size": cfg.vocab_size,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "T": cfg.n_iter,
        "dt": 1.0,
        "tau_m": cfg.tau_m,
        "act_fx": cfg.act_fx,
        "eta": cfg.eta,
        "dropout_rate": cfg.dropout_rate,
        "exp_dir": None,
        "loadDir": None,
        "pos_learnable": cfg.pos_learnable,
        "optim_type": cfg.optim_type,
        "wub": cfg.wub,
        "wlb": cfg.wlb,
        "model_name": f"trial_{trial_number}"
    }

    model = NGCTransformer(**model_args)
    return model, train_loader, valid_loader
def run_single_trial_efe(trial):
    try:
        params = define_search_space(trial)
        print(f"[EFE Phase] Trial {trial.number} | params: {params}")

        cfg = type('Config', (), {})()
        for key, value in base_config.__dict__.items():
            if not key.startswith('_'):
                setattr(cfg, key, value)
        for key, value in params.items():
            setattr(cfg, key, value)
        if not hasattr(cfg, 'vocab_size'):
            cfg.vocab_size = base_config.vocab_size

        try:
            model, train_loader, valid_loader = create_model_with_all_params(trial.number, params, cfg)
        except Exception as e:
            reason = f"Failed to create model: {e}"
            trial.set_user_attr("prune_reason", reason)
            print(reason)
            raise optuna.TrialPruned()

        total_EFE = 0.0
        batches_processed = 0
        start_time = time.time()
        max_batches = 1
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            inputs = batch[0][1]
            targets = batch[1][1]
            targets_flat = jax.nn.one_hot(targets.flatten(), cfg.vocab_size)


            try:
                _, _, EFE, *_ = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
                EFE = abs(float(EFE))
            except Exception as e:
                reason = f"model.process failed: {e}"
                trial.set_user_attr("prune_reason", reason)
                print(reason)
                raise optuna.TrialPruned()

            if jnp.isnan(EFE) or jnp.isinf(EFE) or EFE > EFE_STABILITY_THRESHOLD:
                reason = f"Unstable EFE: {EFE}"
                trial.set_user_attr("prune_reason", reason)
                print(reason)
                raise optuna.TrialPruned()

            total_EFE += EFE
            batches_processed += 1
            current_efe = total_EFE / batches_processed

            trial.report(current_efe, 0)
            if trial.should_prune():
                reason = f"TPE pruned at batch {batch_idx} | current EFE={current_efe:.4f}"
                trial.set_user_attr("prune_reason", reason)
                print(reason)
                raise optuna.TrialPruned()

            elapsed = time.time() - start_time
            print(f"Batch {batch_idx} | EFE={EFE:.4f} | Avg EFE={current_efe:.4f} | Time={elapsed:.1f}s")

        final_efe = total_EFE / batches_processed if batches_processed > 0 else 1000.0
        total_time = time.time() - start_time

        trial.set_user_attr("time", total_time)

        for key, value in params.items():
            trial.set_user_attr(f"param_{key}", value)

        print(f"Trial {trial.number} Complete | EFE={final_efe:.4f} | Time={total_time:.1f}s")
        return float(final_efe)
    finally:
        
        # Delete Python objects
        for obj_name in ['model', 'train_loader', 'valid_loader']:
            if obj_name in locals() and locals()[obj_name] is not None:
                del locals()[obj_name]
        
        # Force garbage collection
        for _ in range(2):
            gc.collect()
        
        # Clear JAX caches
        try:
            jax.clear_caches()
        except:
            pass

def case1_efe_only_complete():
    Path("tuning").mkdir(exist_ok=True)

    print("TPE optimizing EFE only (all parameters)")
    study_efe = optuna.create_study(
        study_name="case1_efe_only",
        storage="sqlite:///tuning/case1_efe_only.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=1, reduction_factor=2)
    )

    study_efe.optimize(run_single_trial_efe, n_trials=30, n_jobs=1, show_progress_bar=False)

    if study_efe.best_trial:
        best_efe = study_efe.best_value
        best_params = study_efe.best_trial.params
        best_ce = study_efe.best_trial.user_attrs.get("ce", float("inf"))
        best_ppl = study_efe.best_trial.user_attrs.get("ppl", float("inf"))
        
        print(f"\n{'='*60}")
        print("EFE TUNING COMPLETE")
        print(f"{'='*60}")
        print(f"Best EFE: {best_efe:.4f}")
        print(f"Best CE: {best_ce:.4f}")
        print(f"Best PPL: {best_ppl:.4f}")
        print(f"\nBest Parameters:")
        for key in ['n_layers', 'n_heads', 'n_embed', 'tau_m', 'n_iter',
                   'batch_size', 'seq_len', 'pos_learnable', 'optim_type', 'act_fx', 'eta', 'dropout_rate', 'wub', 'wlb']:
            print(f"  {key}: {best_params.get(key)}")
    else:
        return None

    with open("tuning/best_hyperparameters.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("BEST HYPERPARAMETERS\n")
        f.write("="*60 + "\n\n")
        f.write("EFE-ONLY SEARCH\n")
        f.write("-"*40 + "\n")
        f.write(f"Best EFE: {best_efe:.6f}\n")
        f.write("-"*40 + "\n")
        f.write(f"Best CE: {study_efe.best_trial.user_attrs.get('ce', float('inf')):.6f}\n")
        f.write(f"Best PPL: {study_efe.best_trial.user_attrs.get('ppl', float('inf')):.6f}\n")
        f.write("\n")
        for key, value in best_params.items():
            f.write(f"{key} = {value}\n")

    print(f"\n✓ Best hyperparameters saved to: tuning/best_hyperparameters.txt")

    return {
        "best_efe": best_efe,
        "best_ce": best_ce,
        "best_ppl": best_ppl,
        "parameters": best_params,
    }

def main():
    print("PC TRANSFORMER - EFE HYPERPARAMETER TUNING")
    print("="*60)
    print("Optimize only EFE")
    print("="*60)

    try:
        results = case1_efe_only_complete()
        if results:
            print(f"\n{'='*60}")
            print("TUNING COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Final Results:")
            print(f"- Best EFE: {results['best_efe']:.4f}")
            print(f"- Best CE: {results['best_ce']:.4f}")
            print(f"- Best PPL: {results['best_ppl']:.4f}")
            print(f"\n Parameters saved to: tuning/best_hyperparameters.txt")
        else:
            print("Tuning failed or was interrupted.")
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")
    except Exception as e:
        print(f"Error during tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()