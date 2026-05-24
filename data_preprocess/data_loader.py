import jax.numpy as jnp
from pathlib import Path
from ngclearn.utils.data_loader import DataLoader as NGCDataLoader
import sys
import numpy as np

DIR = Path(__file__).parent
sys.path.append(str(DIR.parent))


class DataLoader:
    def __init__(
        self,
        seq_len,
        batch_size,
        data_dir=DIR / "outputs" / "tokenized_data"
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token = -1

    def load_and_prepare_data(self, max_samples=None, train_ratio=None):
        """
        Load tokenized data and optionally limit dataset size.

        Args:
            max_samples (int): take first N tokens
            train_ratio (float): take percentage of training data (0.0 - 1.0)
        """

        train_tokens = jnp.load(self.data_dir / "train_tokens.npy")
        valid_tokens = jnp.load(self.data_dir / "valid_tokens.npy")
        test_tokens = jnp.load(self.data_dir / "test_tokens.npy")

        # -----------------------------
        # 🔥 LIMIT TRAIN DATA (DEBUG MODE)
        # -----------------------------
        if max_samples is not None:
            train_tokens = train_tokens[:max_samples]

        # -----------------------------
        # 🔥 LIMIT BY RATIO (EXPERIMENT MODE)
        # -----------------------------
        if train_ratio is not None:
            cut = int(len(train_tokens) * train_ratio)
            train_tokens = train_tokens[:cut]

        # Create loaders
        train_loader = self._create_data_loader(train_tokens, shuffle=True)
        valid_loader = self._create_data_loader(valid_tokens, shuffle=False)
        test_loader = self._create_data_loader(test_tokens, shuffle=False)

        return train_loader, valid_loader, test_loader

    def _create_data_loader(self, tokens, shuffle):
        """Create sequences and return NGC DataLoader"""

        window_size = self.seq_len + 1
        stride = self.seq_len
        n_tokens = len(tokens)

        num_sequences = (n_tokens - window_size) // stride + 1

        # -----------------------------
        # Handle small dataset case
        # -----------------------------
        if num_sequences <= 0:
            padded_tokens = jnp.concatenate([
                tokens,
                jnp.full((window_size - len(tokens),), self.pad_token)
            ])
            sequences = padded_tokens.reshape(1, -1)
        else:
            indices = np.arange(num_sequences) * stride
            sequences = np.array([
                tokens[i:i + window_size] for i in indices
            ])

        # inputs and targets
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]

        # mask padding tokens
        mask = (targets != self.pad_token).astype(jnp.float32)

        return NGCDataLoader(
            design_matrices=[
                ("inputs", inputs),
                ("targets", targets),
                ("mask", mask),
            ],
            batch_size=self.batch_size,
            disable_shuffle=not shuffle,
            ensure_equal_batches=True
        )