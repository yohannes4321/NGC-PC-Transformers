import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nevergrad as ng
from trainer_wrapper import train_evaluate_model

def run_optimization():
    # 1. DEFINE PARAMETER SPACE using ng.p.Dict
    # This ensures candidate.value returns a clean dictionary
    param_space = ng.p.Dict(
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

    # 2. CREATE OPTIMIZER
    # NGOpt is a powerful meta-optimizer that chooses the best algorithm for your space
    optimizer = ng.optimizers.NGOpt(parametrization=param_space, budget=20)

    history_loss = []
    best_loss = float('inf')
    best_params = None

    print(f"Starting Nevergrad Optimization (budget={optimizer.budget})...")

    # 3. OPTIMIZATION LOOP
    for iteration in range(optimizer.budget):
        candidate = optimizer.ask()
        
        # Because we used ng.p.Dict, x_dict is now exactly what we need
        x_dict = candidate.value 

        # Convert to DataFrame to maintain compatibility with your trainer
        params_df = pd.DataFrame([x_dict])

        # Evaluate the model
        # train_evaluate_model returns a 2D array [[loss]]
        loss_array = train_evaluate_model(params_df)
        loss_value = float(loss_array[0][0])

        # Tell optimizer the result
        optimizer.tell(candidate, loss_value)

        # Track history
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
    print("Best Parameters:")
    print(best_params)

    # 5. VISUALIZATION
    plt.figure(figsize=(10,6))
    valid_losses = [l if l != float('inf') else 10.0 for l in history_loss] # Cap inf for plotting
    plt.plot(valid_losses, marker='o', label='Trial Loss')
    plt.plot(np.minimum.accumulate(valid_losses), 'r--', linewidth=2, label='Best So Far')
    plt.title('Nevergrad Optimization Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('nevergrad_optimization_log.png')
    print("Visualization saved to nevergrad_optimization_log.png")

if __name__ == "__main__":
    run_optimization()