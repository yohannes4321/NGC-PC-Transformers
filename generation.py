from model import NGCTransformer
import jax
import jax.numpy as jnp
import numpy as np
from config import Config as config
from data_preprocess.data_loader import DataLoader
from data_preprocess.tokenizer import get_tokenizer, BPETokenizer
from pathlib import Path


# Initialize the model and tokenizer only when run as a script
if __name__ == "__main__":
    # Initialize the model
    dkey = jax.random.PRNGKey(0)
    model = NGCTransformer(
        dkey, 
        batch_size=1, # Standard for single-sequence generation
        seq_len=config.seq_len, 
        n_embed=config.n_embed, 
        vocab_size=config.vocab_size, 
        n_layers=config.n_layers, 
        n_heads=config.n_heads,
        T=config.n_iter, 
        dt=1., 
        tau_m=config.tau_m, 
        act_fx=config.act_fx, 
        eta=config.eta, 
        dropout_rate=config.dropout_rate, 
        exp_dir="exp",
        loadDir="exp", # Ensure model is loaded from trained exp/ directory
        pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, 
        wub=config.wub, 
        wlb=config.wlb, 
        model_name="ngc_transformer"
    )

    # Call weight stats once
    weight_stats(model)

    tokenizer = get_tokenizer(config)

    if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is None:
        vocab_file = getattr(config, "tokenizer_vocab_file", None)
        if vocab_file is None:
            default_path = Path(__file__).parent / "data_preprocess" / "outputs" / "tokenizer" / "bpe_tokenizer.json"
            if default_path.exists():
                vocab_file = str(default_path)
                print(f"Auto-loading BPE tokenizer from default path: {vocab_file}")
        
        # Attempt to load
        if vocab_file and Path(vocab_file).exists():
            tokenizer.load_tokenizer(vocab_file)
            print(f"Loaded BPE tokenizer (vocab size: {tokenizer.get_vocab_size()})")
        else:
            raise RuntimeError(
                "BPE tokenizer not trained or loaded!\n\n"
            )


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    seq_len: int = config.seq_len,
    temperature: float = 1.0,
    top_k: int = 0,
    key=None,
    pad_token_id: int = None
):
    """
    Generate text using the model and provided tokenizer.
    Works with both custom BPE and tiktoken backends.
    """
    # Encode prompt - returns jnp.ndarray for both backends
    prompt_ids = tokenizer.encode(prompt)
    
    # Ensure batch dimension: (1, sequence_length)
    if prompt_ids.ndim == 1:
        prompt_tensor = prompt_ids[None, :]
    else:
        prompt_tensor = prompt_ids

    current_tokens = prompt_tensor
    current_key = key

    if pad_token_id is None:
        if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is not None:
            pad_token_id = tokenizer.tokenizer.token_to_id("<pad>")
        else:
            pad_token_id = 0

    for _ in range(max_new_tokens):
        # Truncate context to fit model's seq_len
        if current_tokens.shape[1] > seq_len:
            input_seq = current_tokens[:, -seq_len:]
        else:
            input_seq = current_tokens

        # Pad to exactly seq_len if needed
        if input_seq.shape[1] < seq_len:
            pad_len = seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
        
        # Forward pass (no target clamping during inference)
        y_mu_inf, y_mu, _ = model.process(input_seq, lab=None, adapt_synapses=False)
        logits = y_mu.reshape(model.batch_size, seq_len, config.vocab_size)

        # Get logits for the last *real* token (excluding padding)
        if current_tokens.shape[1] > seq_len:
            last_pos = seq_len - 1
        else:
            last_pos = current_tokens.shape[1] - 1
        next_logits = logits[0, last_pos, :] / temperature

        # Sample or take argmax
        if current_key is not None:
            if top_k is not None and top_k > 0:
                top_k = min(top_k, config.vocab_size)
                top_vals, top_idx = jnp.topk(next_logits, k=top_k)
                probs = jax.nn.softmax(top_vals)
                current_key, subkey = jax.random.split(current_key)
                choice = jax.random.choice(subkey, a=top_k, p=probs)
                next_token = top_idx[choice]
            else:
                probs = jax.nn.softmax(next_logits)
                current_key, subkey = jax.random.split(current_key)
                next_token = jax.random.choice(subkey, a=config.vocab_size, p=probs)
        else:
            next_token = jnp.argmax(next_logits)

        # Append new token
        current_tokens = jnp.concatenate([current_tokens, next_token[None, None]], axis=1)

    # Decode generated IDs back to text
    generated_ids = current_tokens[0].tolist()
    return tokenizer.decode(generated_ids)

