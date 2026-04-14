import subprocess
import sys
import os
import time

def run_tuning(gpu_id, eta_start, eta_end, study_name, n_trials=10):
    cmd = [
        sys.executable, "tuning.py",
        "--gpu", str(gpu_id),
        "--eta_start", str(eta_start),
        "--eta_end", str(eta_end),
        "--study_name", study_name,
        "--n_trials", str(n_trials)
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting: GPU {gpu_id} | Eta: {eta_start} - {eta_end}")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    result = subprocess.run(cmd, env=env)
    return result.returncode

def main():
    print("Multi-GPU Hyperparameter Tuning Launcher")
    print("=" * 60)
    
    gpu_configs = [
        {"gpu": 0, "eta_start": 1e-5, "eta_end": 1e-3, "study": "eta_1_10"},
        {"gpu": 1, "eta_start": 1e-3, "eta_end": 1e-1, "study": "eta_11_20"},
    ]
    
    n_trials = 10
    
    processes = []
    for config in gpu_configs:
        p = run_tuning(
            config["gpu"],
            config["eta_start"],
            config["eta_end"],
            config["study"],
            n_trials
        )
        processes.append(p)
    
    for i, p in enumerate(processes):
        gpu = gpu_configs[i]
        print(f"\nGPU {gpu['gpu']} (eta {gpu['eta_start']}-{gpu['eta_end']}) completed with code {p}")
    
    print("\n" + "="*60)
    print("All tuning jobs complete!")
    print("="*60)

if __name__ == "__main__":
    main()