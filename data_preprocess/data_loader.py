# dataloader_benchmark.py

import time
import os
import psutil
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from ngclearn.utils.data_loader import DataLoader as NGCDataLoader


# ------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------
DIR = Path(__file__).parent
sys.path.append(str(DIR.parent))


# ------------------------------------------------------------------
# RAM MONITOR
# ------------------------------------------------------------------
def ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# ------------------------------------------------------------------
# DATA LOADER
# ------------------------------------------------------------------
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

    # --------------------------------------------------------------
    # LOAD DATA (CPU ONLY)
    # --------------------------------------------------------------
    def load_and_prepare_data(self):
        """
        Load token arrays into CPU RAM.
        Never load full datasets into GPU memory.
        """

        print(">>> Loading token files (CPU RAM only)")
        print("RAM before loading:", ram_mb(), "MB")

        train_tokens = np.load(self.data_dir / "train_tokens.npy")
        valid_tokens = np.load(self.data_dir / "valid_tokens.npy")
        test_tokens  = np.load(self.data_dir / "test_tokens.npy")

        print("RAM after loading:", ram_mb(), "MB")

        train_loader = self._create_data_loader(train_tokens, shuffle=True, tag="TRAIN")
        valid_loader = self._create_data_loader(valid_tokens, shuffle=False, tag="VALID")
        test_loader  = self._create_data_loader(test_tokens,  shuffle=False, tag="TEST")

        return train_loader, valid_loader, test_loader

    # --------------------------------------------------------------
    # WINDOW CREATION (ZERO-COPY)
    # --------------------------------------------------------------
    def _create_data_loader(self, tokens, shuffle: bool, tag: str):
        """
        O(1) window creation using NumPy stride tricks.
        Zero-copy. No Python loops. No VRAM usage.
        """

        window_size = self.seq_len + 1

        print(f"\n[{tag}] Creating windows")
        print("RAM before windowing:", ram_mb(), "MB")

        # Pad only if required
        if len(tokens) < window_size:
            tokens = np.pad(
                tokens,
                (0, window_size - len(tokens)),
                constant_values=self.pad_token,
            )

        # ---------------- TIMING ----------------
        start = time.time()

        sequences = np.lib.stride_tricks.sliding_window_view(
            tokens, window_size
        )

        elapsed = time.time() - start
        print(f"[{tag}] Windowing time: {elapsed:.6f} seconds")

        print("RAM after windowing:", ram_mb(), "MB")

        # Split inputs / targets
        inputs  = sequences[:, :-1]
        targets = sequences[:, 1:]

        # ----------------------------------------------------------
        # IMPORTANT: Data stays on CPU until batch transfer
        # ----------------------------------------------------------
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


# ------------------------------------------------------------------
# OPTIONAL: QUICK SANITY TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== DataLoader Benchmark ===")

    loader = DataLoader(
        seq_len=256,
        batch_size=64,
    )

    train_loader, valid_loader, test_loader = loader.load_and_prepare_data()

    print("\n>>> Starting training loop")
    for batch in train_loader:
        x = batch["inputs"]
        y = batch["targets"]
        print("Batch shapes:", x.shape, y.shape)
        break
