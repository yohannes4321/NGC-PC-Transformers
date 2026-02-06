# dataloader_benchmark.py

import time
import sys
from pathlib import Path

import numpy as np
from ngclearn.utils.data_loader import DataLoader as NGCDataLoader

DIR = Path(__file__).parent
sys.path.append(str(DIR.parent))


class DataLoader:
    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        data_dir: Path = DIR / "outputs" / "tokenized_data",
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token = 0

    # LOAD DATA (CPU ONLY)
    def load_and_prepare_data(self):
        """
        Load token arrays into CPU RAM.
        Never load full datasets into GPU memory.
        """

        train_tokens = np.load(self.data_dir / "train_tokens.npy")
        valid_tokens = np.load(self.data_dir / "valid_tokens.npy")
        test_tokens  = np.load(self.data_dir / "test_tokens.npy")

        train_loader = self._create_data_loader(
            train_tokens,
            shuffle=True,
            max_samples=1000,   # ✅ limit train
        )

        valid_loader = self._create_data_loader(
            valid_tokens,
            shuffle=False,
            max_samples=800,    # ✅ limit validation
        )

        test_loader = self._create_data_loader(
            test_tokens,
            shuffle=False,
            max_samples=None,   # use all test windows
        )

        return train_loader, valid_loader, test_loader

    # WINDOW CREATION (ZERO-COPY)
    def _create_data_loader(self, tokens, shuffle, max_samples=None):
        """
        O(1) window creation using NumPy stride tricks.
        Zero-copy. No Python loops. No VRAM usage.
        """

        window_size = self.seq_len + 1

        # Pad only if required
        if len(tokens) < window_size:
            tokens = np.pad(
                tokens,
                (0, window_size - len(tokens)),
                constant_values=self.pad_token,
            )

        # Sliding window (zero-copy)
        sequences = np.lib.stride_tricks.sliding_window_view(
            tokens, window_size
        )

        # Optional cap on number of sequences
        if max_samples is not None:
            sequences = sequences[:max_samples]

        # Split inputs / targets
        inputs  = sequences[:, :-1]
        targets = sequences[:, 1:]

        loader = NGCDataLoader(
            design_matrices=[
                ("inputs", inputs),
                ("targets", targets),
            ],
            batch_size=self.batch_size,
            disable_shuffle=not shuffle,
            ensure_equal_batches=True,
        )

        return loader
