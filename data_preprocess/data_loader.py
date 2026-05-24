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
        data_dir=DIR / "outputs" / "tokenized_data",
        max_samples=50  # NEW
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token = -1
        self.max_samples = max_samples

    def load_and_prepare_data(self):
        """Load tokenized data and prepare small subset"""

        train_tokens = jnp.load(self.data_dir / "train_tokens.npy")
        valid_tokens = jnp.load(self.data_dir / "valid_tokens.npy")
        test_tokens = jnp.load(self.data_dir / "test_tokens.npy")

        # =========================
        # USE SMALL DATA ONLY
        # =========================
        train_tokens = train_tokens[:self.max_samples]
        valid_tokens = valid_tokens[: max(200, self.max_samples // 10)]
        test_tokens = test_tokens[: max(200, self.max_samples // 10)]

        print(f"Using small dataset:")
        print(f"Train tokens: {len(train_tokens)}")
        print(f"Valid tokens: {len(valid_tokens)}")
        print(f"Test tokens : {len(test_tokens)}")

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

        if num_sequences <= 0:
            padded_tokens = jnp.concatenate([
                tokens,
                jnp.full((window_size - len(tokens),), self.pad_token)
            ])

            sequences = padded_tokens.reshape(1, -1)

        else:
            indices = np.arange(num_sequences) * stride

            sequences = np.array([
                tokens[i:i + window_size]
                for i in indices
            ])

        # =========================
        # INPUT / TARGET
        # =========================
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]

        # =========================
        # MASK
        # =========================
        mask = (targets != self.pad_token).astype(jnp.float32)

        print(f"Created {len(inputs)} sequences")

        return NGCDataLoader(
            design_matrices=[
                ("inputs", inputs),
                ("targets", targets),
                ("mask", mask)
            ],
            batch_size=self.batch_size,
            disable_shuffle=not shuffle,
            ensure_equal_batches=True
        )