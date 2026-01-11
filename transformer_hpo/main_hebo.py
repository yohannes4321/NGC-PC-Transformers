# filename: main_nevergrad.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nevergrad as ng
from trainer_wrapper import train_evaluate_model

def run_optimization():
    # 1. DEFINE PARAMETER SPACE
    # Using ng.p.Dict ensures candidate.value is a clean dictionary
    # Using Choice for dimensions prevents the "prime number" shape issue
    param_space = ng.p.Dict(
        n_embed=ng.p.Choice([128, 256, 384]),
        n_heads=ng.p.Choice([2, 4]),
        n_layers=ng.p.Scalar(lower=1, upper=5).set_integer_casting(),
        block_size=ng.p.Choice([32, 64, 96]), 
        batch_size=ng.p.Choice([8, 16, 32]),
        T=ng.p.Scalar(lower=1, upper=10).set_integer_casting(),
        eta=ng.p.Log(lower=1e-4, upper=1e-1),
        dropout=ng.p.Scalar(lower=0.0, upper=0.5),
        wlb=ng.p.Scalar(lower=-0.1, upper=0.0),
        wub=ng.p.Scalar(lower=0.0, upper=0.1),
        tau_m=ng.p.Scalar(lower=1.0, upper=20.0),
        act_fx=ng.p.Choice(['relu', 'gelu', 'tanh'])
    )

    # 2. CREATE OPTIMIZER
    optimizer = ng.optimizers.NGOpt(parametrization=param_space, budget=20)

    history_loss = []
    best_loss = float('inf')
    best_params = None

    print(f"Starting Nevergrad Optimization (budget={optimizer.budget})...")

    # 3. OPTIMIZATION LOOP
    for iteration in range(optimizer.budget):
        candidate = optimizer.ask()
        x_dict = candidate.value 

        # Evaluate the model
        loss_array = train_evaluate_model(x_dict)
        loss_value = float(loss_array[0][0])

        optimizer.tell(candidate, loss_value)

        history_loss.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = x_dict
            print(f">>> Iteration {iteration+1}: NEW BEST LOSS = {best_loss:.4f}")
        else:
            print(f"Iteration {iteration+1}: Loss = {loss_value:.4f}")

    # 4. REPORT RESULTS
    print("\n--- Optimization Finished ---")
    print(f"Best Loss (CL): {best_loss:.4f}")
    if best_loss < float('inf'):
        print(f"Best PPL:       {math.exp(best_loss):.2f}")
    print("Best Parameters:", best_params)

    # 5. VISUALIZATION
    plt.figure(figsize=(10,6))
    # Filter out inf for the plot
    plot_data = [l if l != float('inf') else 10.0 for l in history_loss]
    plt.plot(plot_data, marker='o', label='Trial Loss')
    plt.plot(np.minimum.accumulate(plot_data), 'r--', label='Best So Far')
    plt.title('Nevergrad Optimization Convergence')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig('nevergrad_optimization_log.png')

if __name__ == "__main__":
    run_optimization()