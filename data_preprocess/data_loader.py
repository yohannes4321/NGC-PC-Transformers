import numpy as np # Use CPU NumPy for the "View" logic
import jax.numpy as jnp
from pathlib import Path
from ngclearn.utils.data_loader import DataLoader as NGCDataLoader
import sys

DIR = Path(__file__).parent
sys.path.append(str(DIR.parent))

class DataLoader:
    def __init__(self, seq_len, batch_size, data_dir= DIR / "outputs" / "tokenized_data"):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token = 0

    def load_and_prepare_data(self):
        """Load data into CPU RAM first to avoid VRAM congestion"""
        # Load using standard numpy. This is crucial for big data.
        train_tokens = np.load(self.data_dir / "train_tokens.npy")
        valid_tokens = np.load(self.data_dir / "valid_tokens.npy")
        test_tokens = np.load(self.data_dir / "test_tokens.npy")

        train_loader = self._create_data_loader(train_tokens, shuffle=True)
        valid_loader = self._create_data_loader(valid_tokens, shuffle=False)
        test_loader = self._create_data_loader(test_tokens, shuffle=False)

        return train_loader, valid_loader, test_loader

    def _create_data_loader(self, tokens, shuffle):
        """Vectorized Windowing without Python loops"""
        window_size = self.seq_len + 1 
        
        # 1. Handle short data sequences with padding
        if len(tokens) < window_size:
            tokens = np.pad(tokens, (0, window_size - len(tokens)), 
                            constant_values=self.pad_token)
        
        # 2. THE LOOP REPLACEMENT: Sliding Window View
        # This creates a 'virtual' array of shape (num_sequences, window_size)
        # It takes 0.0001 seconds even for millions of tokens.
        sequences = np.lib.stride_tricks.sliding_window_view(tokens, window_size)
        
        # 3. Slice the virtual array
        # No actual data is moved or copied here yet
        inputs = sequences[:, :-1]    
        targets = sequences[:, 1:]    
                
        # 4. NGCDataLoader handles the batching
        return NGCDataLoader(
            design_matrices=[("inputs", inputs), ("targets", targets)],
            batch_size=self.batch_size,
            disable_shuffle=not shuffle,
            ensure_equal_batches=True
        )