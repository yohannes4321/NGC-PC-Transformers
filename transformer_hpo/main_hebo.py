# filename: main_nevergrad.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import jax
# Force JAX to use GPU
jax.config.update("jax_default_device", jax.devices("gpu")[0])

from trainer_wrapper import train_evaluate_model

import nevergrad as ng

def run_optimization():
    # 1. DEFINE PARAMETER SPACE
    # Mirrors your HEBO design space
    param_instrumentation = ng.p.Instrumentation(
        n_embed=ng.p.Scalar(lower=32, upper=512).set_integer_casting(),
        n_heads=ng.p.Choice([2, 4, 8]),
        n_layers=ng.p.Scalar(lower=1, upper=6).set_integer_casting(),
        block_size=ng.p.Scalar(lower=32, upper=128).set_integer_casting(),
        batch_size=ng.p.Scalar(lower=8, upper=64).set_integer_casting(),
        T=ng.p.Scalar(lower=1, upper=10).set_integer_casting(),
        eta=ng.p.Log(lower=1e-4, upper=1e-1),
        dropout=ng.p.Scalar(lower=0.0, upper=0.5),
        wlb=ng.p.Scalar(lower=-0.1, upper=0.0),
        wub=ng.p.Scalar(lower=0.0, upper=0.1),
        tau_m=ng.p.Scalar(lower=1.0, upper=20.0),
        act_fx=ng.p.Choice(['relu', 'gelu', 'tanh'])
    )

    # 2. CREATE NEVERGRAD OPTIMIZER
    optimizer = ng.optimizers.NGOpt(param_instrumentation, budget=20)

    history_loss = []
    best_loss = float('inf')
    best_params = None

    print(f"Starting Nevergrad Optimization (budget={optimizer.budget})...")

    # 3. OPTIMIZATION LOOP
    for iteration in range(optimizer.budget):
        # Ask for a suggestion
        candidate = optimizer.ask()
        x_dict = candidate.value  # Dictionary of suggested hyperparams

        # Nevergrad uses nested dictionaries; convert to DataFrame for trainer
        import pandas as pd
        params_df = pd.DataFrame([x_dict])

        # Evaluate the model
        loss_array = train_evaluate_model(params_df)
        loss_value = loss_array[0][0]

        # Tell optimizer the result
        optimizer.tell(candidate, loss_value)

        # Track history
        history_loss.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = x_dict
            print(f">>> Iteration {iteration+1}: NEW BEST LOSS = {best_loss:.4f}")

    # 4. REPORT RESULTS
    print("\n--- Optimization Finished ---")
    print(f"Best Loss (CL): {best_loss:.4f}")
    try:
        print(f"Best PPL:       {math.exp(best_loss):.2f}")
    except OverflowError:
        print("Best PPL:       inf")
    print("Best Parameters:")
    print(best_params)

    # 5. VISUALIZATION
    plt.figure(figsize=(10,6))
    plt.plot(history_loss, marker='o', label='Trial Loss')
    plt.plot(np.minimum.accumulate(history_loss), 'r--', linewidth=2, label='Best So Far')
    plt.title('Nevergrad Optimization Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('nevergrad_optimization_log.png')
    print("Visualization saved to nevergrad_optimization_log.png")

if __name__ == "__main__":
    run_optimization()
