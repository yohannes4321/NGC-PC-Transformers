import matplotlib.pyplot as plt
import numpy as np

# Trials
trials = np.arange(18)

# EFE values (absolute value)
efe = [np.nan, 6.7293, 3.8622, 12.3669, 0.5269, 5.3588, 3.7878, 34.4264,
       7.4638, np.nan, 17.0206, 0.5535, 0.5253, 0.5110, 0.5142, 0.5095, 0.5093, 0.5095]

# Hyperparameters per trial
n_heads    = [8, 4, 2, 4, 4, 8, 2, 8, 4, 8, 8, 8, 4, 2, 2, 8, 4, 2]
embed_mult = [8,16,16,32,32,8,32,32,16,16,16,16,32,32,32,8,16,32]
batch_size = [64,16,32,16,32,16,64,32,64,64,16,32,32,32,32,32,16,16]
seq_len    = [64,16,16,32,16,16,16,16,32,64,32,32,16,32,16,16,64,16]

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14,10))
param_names = ['n_heads','embed_mult','batch_size','seq_len']
param_values = [n_heads, embed_mult, batch_size, seq_len]

for ax, name, values in zip(axs.flatten(), param_names, param_values):
    ax.scatter(values, efe, color='blue')
    for i, txt in enumerate(trials):
        ax.annotate(txt, (values[i], efe[i]), fontsize=8, alpha=0.7)
    ax.set_xlabel(name)
    ax.set_ylabel('EFE')
    ax.set_title(f'EFE vs {name}')
    ax.grid(True)

plt.tight_layout()
plt.show()
