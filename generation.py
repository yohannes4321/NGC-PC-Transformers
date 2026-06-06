from model import NGCTransformer
import jax
import jax.numpy as jnp
import numpy as np
from config import Config as config
from data_preprocess.data_loader import DataLoader
from data_preprocess.tokenizer import get_tokenizer, BPETokenizer, TiktokenTokenizer
from pathlib import Path
import re
import textwrap


def _get_special_tokens(tokenizer):
    """
    Returns (pad_token_id, start_token_id) for whichever backend is active.

    BPETokenizer  → uses <pad> and <bos> from its trained vocab
    TiktokenTokenizer → uses eot_token for both (tiktoken has no <pad>/<bos>)
    """
    if isinstance(tokenizer, BPETokenizer):
        pad_id   = tokenizer.tokenizer.token_to_id("<pad>")
        start_id = tokenizer.tokenizer.token_to_id("<bos>")
        if pad_id   is None: pad_id   = 0
        if start_id is None: start_id = pad_id
        return pad_id, start_id

    if isinstance(tokenizer, TiktokenTokenizer):
        eot = tokenizer._enc.eot_token   # e.g. 199999 for o200k_base
        return eot, eot

    # Fallback for any future backend
    return 0, 0


def generate_text(
    model,
    tokenizer,
    max_new_tokens: int = 100,
    seq_len: int = config.seq_len,
    temperature: float = 1.0,
    top_k: int = 0,
    key=None,
):
    """
    Generate text using the model.
    Works with both BPETokenizer and TiktokenTokenizer backends.
    """
    pad_token_id, start_token_id = _get_special_tokens(tokenizer)

    # Start sequence with the start/bos token
    current_tokens = jnp.array([[start_token_id]], dtype=jnp.int32)
    current_key = key

    for _ in range(max_new_tokens):
        # Keep only the last seq_len tokens as context
        if current_tokens.shape[1] > config.seq_len:
            input_seq = current_tokens[:, -config.seq_len:]
        else:
            input_seq = current_tokens

        # Pad up to seq_len if shorter
        if input_seq.shape[1] < config.seq_len:
            pad_len = config.seq_len - input_seq.shape[1]
            input_seq = jnp.pad(
                input_seq, ((0, 0), (0, pad_len)),
                constant_values=pad_token_id
            )

        # Forward pass (no target clamping during inference)
        dummy_target = jnp.zeros(
            (config.batch_size * config.seq_len, config.vocab_size)
        )
        y_mu_inf, y_mu, _ = model.process(
            input_seq, dummy_target, adapt_synapses=False
        )
        logits = y_mu_inf.reshape(
            config.batch_size, config.seq_len, config.vocab_size
        )

        # Pick logits at the last *real* token position (not padding)
        real_len = min(current_tokens.shape[1], config.seq_len)
        last_pos = real_len - 1
        next_logits = logits[0, last_pos, :] / temperature

        # Sample with optional top-k, or greedy argmax
        if current_key is not None:
            if top_k is not None and top_k > 0:
                k = min(top_k, config.vocab_size)
                top_vals, top_idx = jax.lax.top_k(next_logits, k=k)
                probs = jax.nn.softmax(top_vals)
                current_key, subkey = jax.random.split(current_key)
                choice = jax.random.choice(subkey, a=k, p=probs)
                next_token = top_idx[choice]
            else:
                probs = jax.nn.softmax(next_logits)
                current_key, subkey = jax.random.split(current_key)
                next_token = jax.random.choice(
                    subkey, a=config.vocab_size, p=probs
                )
        else:
            next_token = jnp.argmax(next_logits)

        current_tokens = jnp.concatenate(
            [current_tokens, next_token[None, None]], axis=1
        )

    # Decode — tiktoken.decode expects List[int], BPE handles both
    generated_ids = current_tokens[0].tolist()

    # Strip the leading start/pad token before decoding
    if generated_ids and generated_ids[0] == start_token_id:
        generated_ids = generated_ids[1:]

    return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dkey = jax.random.PRNGKey(0)
    model = NGCTransformer(
        dkey,
        batch_size=config.batch_size,
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
        loadDir="exp",
        pos_learnable=config.pos_learnable,
        optim_type=config.optim_type,
        wub=config.wub,
        wlb=config.wlb,
        model_name="ngc_transformer"
    )

    tokenizer = get_tokenizer(config)

    # BPE: auto-load saved vocab if not already loaded
    if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is None:
        vocab_file = getattr(config, "tokenizer_vocab_file", None)
        if vocab_file is None:
            default_path = (
                Path(__file__).parent
                / "data_preprocess" / "outputs" / "tokenizer" / "bpe_tokenizer.json"
            )
            if default_path.exists():
                vocab_file = str(default_path)
                print(f"Auto-loading BPE tokenizer from: {vocab_file}")

        if vocab_file and Path(vocab_file).exists():
            tokenizer.load_tokenizer(vocab_file)
            print(f"Loaded BPE tokenizer (vocab size: {tokenizer.get_vocab_size()})")
        else:
            raise RuntimeError(
                "BPE tokenizer not trained or loaded!\n"
                "Run data_preprocess/tokenizer.py first, or set tokenizer_vocab_file in config."
            )

    # TiktokenTokenizer needs no loading — already ready after __init__
    if isinstance(tokenizer, TiktokenTokenizer):
        print(f"Using tiktoken (encoding='{tokenizer.encoding}', vocab_size={tokenizer.get_vocab_size()})")

    rng = jax.random.PRNGKey(0)
    rng, key_1 = jax.random.split(rng)
    rng, key_2 = jax.random.split(rng)

    print("\nFINAL GENERATED 1:\n")
    generated_1 = generate_text(
        model,
        tokenizer,
        max_new_tokens=100,
        temperature=0.9,
        top_k=50,
        key=key_1,
    )
    print(generated_1)
