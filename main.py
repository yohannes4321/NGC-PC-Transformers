import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
import pickle
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bertscore
import time
import tiktoken
from contextlib import nullcontext
from pathlib import Path
import requests
import numpy as np
nltk.download('punkt')

from config import Config
batch_size = Config.batch_size
block_size = Config.seq_len
MAX_LENGTH = Config.seq_len
learning_rate = Config.eta_o
n_embd = Config.n_embed
n_head = Config.n_heads
n_layer = Config.n_layers
dropout = Config.dropout_rate
max_epochs = Config.epoch
max_new_tokens = 200
temperature = 0.9
class BPETokenizer:
    def __init__(self, vocab_size: int = 11710):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train_tokenizer(self, all_text: str):
        """Trains a Byte-Pair Encoding (BPE) tokenizer from scratch using a raw text string."""
        # Initialize a BPE model with a standard unknown token handler
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        
        # Pre-tokenize text splitting on spaces before looking up subwords
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Configure the trainer with special tokens and minimum frequency thresholds
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=2
        )

        # Train directly using the in-memory text string iteration
        self.tokenizer.train_from_iterator([all_text], trainer=trainer)

    def load_tokenizer(self, path: str):
        """Loads a pre-trained bpe_tokenizer.json file configuration."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list:
        """Encodes a string into a list of token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded yet.")
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def decode(self, tokens) -> str:
        """Decodes an iterable of token IDs back into string text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded yet.")
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def get_vocab_size(self) -> int:
        """Returns the dynamic vocabulary count calculated during training."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded yet.")
        return self.tokenizer.get_vocab_size()

    def save_tokenizer(self, save_path: str = None):
        """Saves the current trained configuration back down onto disk."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be initialized and trained before saving.")
        if save_path is None:
            save_path = "checkpoints"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(f"{save_path}/bpe_tokenizer.json")


def get_tokenizer(cfg):
    """Factory function: returns (encode_fn, decode_fn, vocab_size) based on Config.tokenizer."""
    tokenizer_type = getattr(cfg, "tokenizer", "BPE").strip().lower()

    if tokenizer_type == "tiktoken":
        # ---- tiktoken path ----
        encoding_name = getattr(cfg, "tokenizer_name", "gpt2")
        print(f"Using tokenizer backend: tiktoken (encoding={encoding_name})")
        enc = tiktoken.get_encoding(encoding_name)
        encode_fn = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
        decode_fn = lambda t: enc.decode(t)
        v_size = enc.n_vocab
        return encode_fn, decode_fn, v_size, None  # None = no trainable tokenizer object

    elif tokenizer_type == "bpe":
        # ---- custom BPE path ----
        print("Using tokenizer backend: custom BPE")
        bpe = BPETokenizer(vocab_size=getattr(cfg, "vocab_size", 11710))

        # Check config parameter paths or default checkpoints directories
        vocab_file = getattr(cfg, "tokenizer_vocab_file", None)
        if not vocab_file:
            default_path = "checkpoints/bpe_tokenizer.json"
            if os.path.exists(default_path):
                vocab_file = default_path

        if vocab_file:
            try:
                bpe.load_tokenizer(vocab_file)
                print(f"Loaded existing BPE tokenizer configuration from {vocab_file}")
            except Exception:
                print("Failed to load saved checkpoint. Will require fresh training.")

        encode_fn = lambda s: bpe.encode(s)
        decode_fn = lambda t: bpe.decode(t)
        v_size = bpe.get_vocab_size() if bpe.tokenizer is not None else getattr(cfg, "vocab_size", 11710)
        return encode_fn, decode_fn, v_size, bpe

    else:
        raise ValueError(f"Unknown tokenizer type '{cfg.tokenizer}'. Choose 'BPE' or 'tiktoken' in Config.")


# ==========================================
# Execution / Training Pipeline Example
# ==========================================

# 0. Load raw text data and split into train/val
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# 1. Initialize tokenizer based on Config.tokenizer choice
encode, decode, vocab_size, _bpe_tokenizer = get_tokenizer(Config)

# 2. For BPE: if no checkpoint was found, train from scratch on raw text
if _bpe_tokenizer is not None and _bpe_tokenizer.tokenizer is None:
    print("No checkpoint found. Training BPE tokenizer on training dataset...")
    _bpe_tokenizer.train_tokenizer(train_data)
    _bpe_tokenizer.save_tokenizer("checkpoints")
    # Refresh encode/decode after training
    encode = lambda s: _bpe_tokenizer.encode(s)
    decode = lambda t: _bpe_tokenizer.decode(t)
    vocab_size = _bpe_tokenizer.get_vocab_size()

# 3. Transform textual dataset segments to standardized lists of IDs
train_ids = encode(train_data)
val_ids = encode(val_data)

# export to bin files in 'data' directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Model architecture=casual self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):#mlp
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Training function
def train(model, optimizer, epoch, num_batches=100):
    model.train()
    total_loss = 0
    total_batches = 0
    for batch_idx in range(num_batches):
        input_ids,targets = get_batch('train')
        logits, loss = model(input_ids, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} | train_loss {loss.item():.4f} | train_perplexity {torch.exp(loss).item():.4f}", flush=True)
    avg_loss = total_loss / total_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, avg_perplexity


def compute_text_metrics(predictions, targets):
    print("\nComputing BERTScore and BLEU...")
    P, R, F1 = bertscore(
        predictions,
        targets,
        lang="en",
        model_type="roberta-base",
        rescale_with_baseline=True,
    )
    print(f"BERTScore (F1): {F1.mean().item():.4f}")

    smooth_fn = SmoothingFunction().method4
    tokenized_targets = [[target.split()] for target in targets]
    tokenized_pred = [pred.split() for pred in predictions]
    bleu = corpus_bleu(tokenized_targets, tokenized_pred, smoothing_function=smooth_fn)
    print(f"BLEU Score: {bleu:.4f}")

@torch.no_grad()
def evaluate(model,num_batches=100):
    start_time = time.time()
    model.eval()
    total_loss = 0
    total_batches = 0
    decoded_targets, decoded_predictions = [], []
    # if max_batches is None:
    #     print(f"Evaluating on the full test set...")
    # else:
    #     print(f"Evaluating on up to {max_batches} batches...")

    for batch_idx in range(num_batches):
        # if max_batches is not None and batch_idx >= max_batches:
        #     break

        input_ids, targets = get_batch('val')

        # Compute loss
        logits, loss = model(input_ids, targets)
        total_loss += loss.item()
        total_batches += 1

        # if compute_metrics:
        #      preds = torch.argmax(logits, dim=-1)
        #      mask = targets != pad_token_id
        #      for i in range(preds.size(0)):
        #         pred_str = decode_ids(tokenizer, preds[i][mask[i]].tolist(), stop_at_eos=True)
        #         tgt_str = decode_ids(tokenizer, targets[i][mask[i]].tolist(), stop_at_eos=True)
        #         decoded_predictions.append(pred_str)
        #         decoded_targets.append(tgt_str)

           
    # if compute_metrics and decoded_predictions and decoded_targets:
    #     compute_text_metrics(decoded_predictions, decoded_targets)
           

    # Compute average loss and perplexity
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss != float('inf') else float('inf')
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    print(f"Total Batches Processed: {batch_idx + 1}")
    print(f"Avg Test CE Loss: {avg_loss:.4f} | Avg Test Perplexity: {avg_perplexity:.4f}")
    return avg_loss,avg_perplexity



# Training phase
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageModel()
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
start_training_time = time.time()
for epoch in range(max_epochs):
    print(f"\nEpoch {epoch + 1}/{max_epochs}")
    avg_loss, avg_perplexity = train(model, optimizer, epoch, num_batches=100)
    print(f"Epoch {epoch + 1} completed | avg_train_loss {avg_loss:.4f} | avg_train_perplexity {avg_perplexity:.4f}")
total_training_time = time.time() - start_training_time
print(f"Total Training Time: {total_training_time:.2f} seconds", flush=True)
print("========== Training completed ==========", flush=True)
# Save model
save_path = "checkpoints/gpt_backprop.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
if os.path.exists(save_path):
    os.remove(save_path)
torch.save({"model_state": model.state_dict()}, save_path)
print("Model saved.")



# Evaluate with metrics
print("starting evaluation")
test_loss, test_perplexity = evaluate(model,num_batches=100)

@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = self(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
num_samples = 10
# Generation setup
top_k = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = torch.float32
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# Use the globally configured encode and decode functions
# (defined based on Config.tokenizer selection at the top of the file)
# encode the beginning of the prompt
start = "\n"
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
print("start generation")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')





