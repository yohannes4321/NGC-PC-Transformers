# filename: main_hebo.py
os.environ["HEBO_FORCE_CPU"] = "true" 
# Ensure JAX doesn't pre-allocate 90% of your VRAM immediately (prevents OOM)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
import numpy as np

import jax
import os
# Force JAX to use the GPU for all operations
jax.config.update("jax_default_device", jax.devices("gpu")[0])

# Force HEBO to CPU (all arrays inside HEBO will be NumPy)


import matplotlib.pyplot as plt
import math
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

# Import the trainer wrapper
from trainer_wrapper import train_evaluate_model

def run_optimization():
    # 1. DEFINE DESIGN SPACE
    # Ranges adapted for Transformer optimization
    params_spec = [
        {'name': 'n_embed',      'type': 'int',  'lb': 32,   'ub': 512},
        {'name': 'n_heads',      'type': 'cat',  'categories': [2, 4, 8]},
        {'name': 'n_layers',     'type': 'int',  'lb': 1,    'ub': 6},
        {'name': 'block_size',   'type': 'int',  'lb': 32,   'ub': 128},
        {'name': 'batch_size',   'type': 'int',  'lb': 8,    'ub': 64},
        {'name': 'T',            'type': 'int',  'lb': 1,    'ub': 10},
        {'name': 'eta',          'type': 'pow',  'lb': 1e-4, 'ub': 1e-1, 'base': 10},
        {'name': 'dropout',      'type': 'num',  'lb': 0.0,  'ub': 0.5},
        {'name': 'wlb',          'type': 'num',  'lb': -0.1, 'ub': 0.0},
        {'name': 'wub',          'type': 'num',  'lb': 0.0,  'ub': 0.1},
        {'name': 'tau_m',        'type': 'num',  'lb': 1.0,  'ub': 20.0},
        {'name': 'act_fx',       'type': 'cat',  'categories': ['relu', 'gelu', 'tanh']}
    ]

    space = DesignSpace().parse(params_spec)

    # 2. INITIALIZE OPTIMIZER
    opt = HEBO(space, model_name='gpyopt', rand_sample=4)
    
    n_iterations = 20  # Set how many trials you want
    history_loss = []
    best_loss = float('inf')

    print(f"Starting HEBO Optimization for {n_iterations} iterations...")
    
    # 3. OPTIMIZATION LOOP
    for i in range(n_iterations):
        print(f"\n--- HEBO Iteration {i+1}/{n_iterations} ---")
        
        # Suggest parameters
        rec_x = opt.suggest(n_suggestions=1)
        
        # Evaluate (runs training in trainer_wrapper.py)
        y = train_evaluate_model(rec_x)
        
        # Observe
        opt.observe(rec_x, y)
        
        # Track Progress
        current_loss = y[0][0]
        history_loss.append(current_loss)
        
        if current_loss < best_loss:
            best_loss = current_loss
            print(f">>> NEW BEST LOSS: {best_loss:.4f} <<<")

    # 4. RESULTS & VISUALIZATION
    print("\noptimization Finished.")
    best_idx = np.argmin(opt.y)
    best_params = opt.X.iloc[best_idx]
    best_score = opt.y[best_idx][0]
    
    print(f"Best Loss (CL): {best_score}")
    print(f"Best PPL:       {math.exp(best_score):.2f}")
    print("Best Parameters:")
    print(best_params)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_loss, marker='o', label='Trial Loss')
    plt.plot(np.minimum.accumulate(history_loss), 'r--', linewidth=2, label='Best So Far')
    plt.title('HEBO Optimization Convergence')
    plt.xlabel('Trial')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('hebo_optimization_log.png')
    print("Visualization saved to hebo_optimization_log.png")

if __name__ == "__main__":
    run_optimization()