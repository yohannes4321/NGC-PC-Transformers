from jax import random


class RandomInitState:
    """Small stateful random initializer with advance_state/run API."""

    def __init__(self, key=None, scale=1e-3):
        self.key = random.PRNGKey(0) if key is None else key
        self.scale = scale

    def advance_state(self, template):
        """Generate a random tensor using template shape and dtype."""
        self.key, subkey = random.split(self.key)
        return random.normal(subkey, shape=template.shape, dtype=template.dtype) * self.scale

    def run(self, template):
        return self.advance_state(template)