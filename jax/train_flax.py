import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from flax_model import FlaxTransformer, compute_metrics

import numpy as np

# Dummy data loader for demonstration
# Replace with your real data pipeline

def get_batch(batch_size, seq_len, vocab_size):
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    y = np.random.randint(0, vocab_size, (batch_size, seq_len))
    return x, y

def create_train_state(rng, config):
    model = FlaxTransformer(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        n_embed=config['n_embed'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout_rate=config['dropout_rate']
    )
    params = model.init(rng, jnp.ones([config['batch_size'], config['seq_len']], jnp.int32))['params']
    tx = optax.adam(config['learning_rate'])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    x, y = batch
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, y)
    return state, metrics

def main():
    config = {
        'vocab_size': 1000,
        'seq_len': 32,
        'n_embed': 128,
        'n_layers': 2,
        'n_heads': 4,
        'dropout_rate': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_steps': 100
    }
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config)
    for step in range(config['num_steps']):
        batch = get_batch(config['batch_size'], config['seq_len'], config['vocab_size'])
        state, metrics = train_step(state, batch)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
