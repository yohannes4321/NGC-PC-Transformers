import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from ngclearn.utils.data_loader import DataLoader as NGCDataLoader
import sys

# Setup directory paths
DIR = Path(__file__).parent
sys.path.append(str(DIR.parent))

class DataLoader:
    def __init__(self, seq_len, batch_size, data_dir=None):
        if data_dir is None:
            self.data_dir = DIR / "outputs" / "tokenized_data"
        else:
            self.data_dir = Path(data_dir)
            
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token = 0
        
        # Verify GPU detection
        devices = jax.devices()
        print(f"--- Data Loader Initialized ---")
        print(f"JAX Devices: {devices}")

    def load_and_prepare_data(self):
        """Load tokenized data and prepare for training using Zero-Copy views"""
        print(f"Loading data from: {self.data_dir}")
        
        # Load as standard NumPy (CPU) to avoid VRAM bloat immediately
        train_tokens = np.load(self.data_dir / "train_tokens.npy")
        valid_tokens = np.load(self.data_dir / "valid_tokens.npy")
        test_tokens = np.load(self.data_dir / "test_tokens.npy")

        train_loader = self._create_data_loader(train_tokens, shuffle=True)
        valid_loader = self._create_data_loader(valid_tokens, shuffle=False)
        test_loader = self._create_data_loader(test_tokens, shuffle=False)

        return train_loader, valid_loader, test_loader

    def _create_data_loader(self, tokens, shuffle):
        """Create sequences using efficient sliding window views"""
        window_size = self.seq_len + 1 
        
        # 1. Handle edge case for short data
        if len(tokens) < window_size:
            padding = np.full((window_size - len(tokens),), self.pad_token)
            tokens = np.concatenate([tokens, padding])
        
        # 2. INSTANT WINDOWING (Zero-Copy)
        # Instead of a loop, we create a 'view' of the memory.
        # This takes 0.001s regardless of how big the data is.
        sequences = np.lib.stride_tricks.sliding_window_view(tokens, window_size)
        
        # 3. Slice into Inputs and Targets
        # inputs: all tokens except the last in each window
        # targets: all tokens except the first in each window (shifted right)
        inputs = sequences[:, :-1]    
        targets = sequences[:, 1:]    
        
        print(f"Created loader: {inputs.shape[0]} sequences of length {self.seq_len}")
                
        # NGCDataLoader will fetch batches from CPU RAM and 
        # move them to GPU VRAM only when needed for training.
        return NGCDataLoader(
            design_matrices=[("inputs", inputs), ("targets", targets)],
            batch_size=self.batch_size,
            disable_shuffle=not shuffle,
            ensure_equal_batches=True
        )

# --- Usage Example ---
if __name__ == "__main__":
    # Example parameters
    loader_tool = DataLoader(seq_len=128, batch_size=32)
    train_ld, valid_ld, test_ld = loader_tool.load_and_prepare_data()
    
    # Check first batch
    for batch in train_ld:
        x_name, x_val = batch[0]
        y_name, y_val = batch[1]
        print(f"Batch loaded to GPU: {x_name} shape {x_val.shape}")
        break # Just checking the first batch