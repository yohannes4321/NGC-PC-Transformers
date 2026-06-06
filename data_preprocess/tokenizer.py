from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import sys
import tiktoken

""" to run: python -m data_preprocess.tokenizer """

DIR = Path(__file__).parent

try:
    sys.path.append(str(DIR.parent))
    from config import Config as config
    VOCAB_SIZE = config.vocab_size
except ImportError:
    VOCAB_SIZE = 12000
    print("Using default vocab_size: 12000")


# ---------------------------------------------------------------------------
# BPE Tokenizer (custom, trained on your data)
# ---------------------------------------------------------------------------

class BPETokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def load_data(self, data_dir: str = None):
        if data_dir is None:
            data_dir = DIR / "data"
        else:
            data_dir = DIR / data_dir

        data_dir = Path(data_dir)

        with open(data_dir / "train.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(data_dir / "valid.txt", "r", encoding="utf-8") as f:
            valid_text = f.read()
        with open(data_dir / "test.txt", "r", encoding="utf-8") as f:
            test_text = f.read()

        all_text = train_text + valid_text + test_text
        return train_text, valid_text, test_text, all_text

    def train_tokenizer(self, all_text: str):
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=2
        )

        self.tokenizer.train_from_iterator([all_text], trainer=trainer)

    def load_tokenizer(self, path: str):
        """
        Load a saved tokenizers Tokenizer JSON file
        (e.g. outputs/tokenizer/bpe_tokenizer.json).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> jnp.ndarray:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained/loaded.")
        encoded = self.tokenizer.encode(text)
        return jnp.array(encoded.ids, dtype=jnp.int32)

    def decode(self, tokens) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained/loaded.")
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def tokenize_splits(self, train_text: str, valid_text: str, test_text: str):
        train_tokens = self.encode(train_text)
        valid_tokens = self.encode(valid_text)
        test_tokens = self.encode(test_text)
        return train_tokens, valid_tokens, test_tokens

    def get_vocab_size(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained/loaded.")
        return self.tokenizer.get_vocab_size()

    def save_tokenizer(self, save_path: str = None):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained/loaded.")
        if save_path is None:
            save_path = DIR / "outputs" / "tokenizer"
        else:
            save_path = DIR / save_path

        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(f"{save_path}/bpe_tokenizer.json")

    def save_data(self, train_tokens, valid_tokens, test_tokens):
        save_dir = DIR / "outputs" / "tokenized_data"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{save_dir}/train_tokens.npy", np.array(train_tokens))
        np.save(f"{save_dir}/valid_tokens.npy", np.array(valid_tokens))
        np.save(f"{save_dir}/test_tokens.npy", np.array(test_tokens))


# ---------------------------------------------------------------------------
# Tiktoken Tokenizer  (uses OpenAI's real tiktoken library)
#
# Encoding guide (pick the best one for your use case):
#   "o200k_base"  – GPT-4o / GPT-4o-mini  (200k vocab, most recent)  ← default
#   "cl100k_base" – GPT-4 / GPT-3.5-turbo  (100k vocab)
#   "p50k_base"   – text-davinci-003 / Codex  (50k vocab)
#   "gpt2"        – GPT-2  (50k vocab, oldest)
# ---------------------------------------------------------------------------

# Best / latest encoding available in tiktoken as of 2025
_BEST_ENCODING = "o200k_base"


class TiktokenTokenizer:
    """
    Tokenizer backed by OpenAI's tiktoken library.
    Default encoding is 'o200k_base' (GPT-4o), the most recent and largest
    vocabulary available in tiktoken.

    Drop-in replacement for BPETokenizer: same encode / decode /
    get_vocab_size / tokenize_splits / save_data API.
    """

    def __init__(self, encoding: str = _BEST_ENCODING):
        self.encoding = encoding
        self._enc = tiktoken.get_encoding(encoding)
        print(
            f"[TiktokenTokenizer] encoding='{encoding}'  "
            f"vocab_size={self._enc.n_vocab}"
        )

    # ------------------------------------------------------------------
    def encode(self, text: str) -> jnp.ndarray:
        ids = self._enc.encode(text)
        return jnp.array(ids, dtype=jnp.int32)

    def decode(self, tokens) -> str:
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return self._enc.decode(tokens)

    def get_vocab_size(self) -> int:
        return self._enc.n_vocab

    def tokenize_splits(self, train_text: str, valid_text: str, test_text: str):
        train_tokens = self.encode(train_text)
        valid_tokens = self.encode(valid_text)
        test_tokens = self.encode(test_text)
        return train_tokens, valid_tokens, test_tokens

    def save_data(self, train_tokens, valid_tokens, test_tokens):
        save_dir = DIR / "outputs" / "tokenized_data"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{save_dir}/train_tokens.npy", np.array(train_tokens))
        np.save(f"{save_dir}/valid_tokens.npy", np.array(valid_tokens))
        np.save(f"{save_dir}/test_tokens.npy", np.array(test_tokens))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_tokenizer(cfg=None):
    """
    Returns a tokenizer instance based on cfg.tokenizer:

      cfg.tokenizer = "BPE"      → custom BPETokenizer (default)
      cfg.tokenizer = "tiktoken" → TiktokenTokenizer

    For tiktoken, the encoding is read from cfg.tokenizer_encoding
    (default: "o200k_base", i.e. GPT-4o – the latest available).

    For BPE, an optional cfg.tokenizer_vocab_file can point to a saved
    bpe_tokenizer.json to skip re-training.
    """
    if cfg is None:
        cfg = config

    backend = getattr(cfg, "tokenizer", "BPE")

    if isinstance(backend, str) and backend.lower() == "tiktoken":
        encoding = getattr(cfg, "tokenizer_encoding", _BEST_ENCODING)
        print(f"[get_tokenizer] backend=tiktoken  encoding='{encoding}'")
        return TiktokenTokenizer(encoding=encoding)

    # Default: custom BPE
    print("[get_tokenizer] backend=BPE (custom)")
    bpe = BPETokenizer(vocab_size=getattr(cfg, "vocab_size", VOCAB_SIZE))
    vocab_file = getattr(cfg, "tokenizer_vocab_file", None)
    if vocab_file:
        try:
            bpe.load_tokenizer(vocab_file)
        except Exception as e:
            print(f"[get_tokenizer] Could not load vocab file: {e}")
    return bpe


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    cfg = config
    tokenizer = get_tokenizer(cfg)

    # Always use BPETokenizer.load_data() to read the raw text splits
    loader = BPETokenizer()
    train_text, valid_text, test_text, all_text = loader.load_data()

    if isinstance(tokenizer, BPETokenizer):
        print("Training custom BPE tokenizer …")
        tokenizer.train_tokenizer(all_text)
        train_tokens, valid_tokens, test_tokens = tokenizer.tokenize_splits(
            train_text, valid_text, test_text
        )
        tokenizer.save_tokenizer()
        tokenizer.save_data(train_tokens, valid_tokens, test_tokens)

    elif isinstance(tokenizer, TiktokenTokenizer):
        print(f"Tokenizing with tiktoken (encoding='{tokenizer.encoding}') …")
        train_tokens, valid_tokens, test_tokens = tokenizer.tokenize_splits(
            train_text, valid_text, test_text
        )
        tokenizer.save_data(train_tokens, valid_tokens, test_tokens)

    print("Done.")


if __name__ == "__main__":
    main()
