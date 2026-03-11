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
        self.pad_token = -1

    def load_and_prepare_data(self):
        """
        Load token arrays using memory mapping.
        Prevents loading entire dataset into RAM.
        """

        train_tokens = np.load(self.data_dir / "train_tokens.npy", mmap_mode="r")
        valid_tokens = np.load(self.data_dir / "valid_tokens.npy", mmap_mode="r")
        test_tokens  = np.load(self.data_dir / "test_tokens.npy", mmap_mode="r")

        train_loader = self._create_data_loader(train_tokens, shuffle=True)
        valid_loader = self._create_data_loader(valid_tokens, shuffle=False)
        test_loader  = self._create_data_loader(test_tokens, shuffle=False)

        return train_loader, valid_loader, test_loader

    def _create_data_loader(self, tokens, shuffle):

        window_size = self.seq_len + 1

        if len(tokens) < window_size:
            tokens = np.pad(
                tokens,
                (0, window_size - len(tokens)),
                constant_values=self.pad_token,
            )

        sequences = np.lib.stride_tricks.sliding_window_view(tokens, window_size)

        inputs  = sequences[:, :-1]
        targets = sequences[:, 1:]

        mask = (targets != self.pad_token).astype(np.float32)

        loader = NGCDataLoader(
            design_matrices=[
                ("inputs", inputs),
                ("targets", targets),
                ("mask", mask),
            ],
            batch_size=self.batch_size,
            disable_shuffle=not shuffle,
            ensure_equal_batches=True,
        )

        return loader
