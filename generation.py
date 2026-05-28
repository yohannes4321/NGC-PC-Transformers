                                                      generation.py                                                                   
from model import NGCTransformer
import jax
import jax.numpy as jnp
import numpy as np
from config import Config as config
from data_preprocess.data_loader import DataLoader
from data_preprocess.tokenizer import get_tokenizer, BPETokenizer
from pathlib import Path


# Initialize the model
dkey = jax.random.PRNGKey(0)
generation_batch_size = config.batch_size
model = NGCTransformer(
                                                                                                                                                                                                                                                                                        generation.py                                                                                                                                                                                                                                                                                               
    dkey, 
    batch_size=generation_batch_size,
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
    loadDir="exp",  # ← FIXED: Load trained model from exp directory 
    pos_learnable=config.pos_learnable, 
    optim_type=config.optim_type, 
    wub=config.wub, 
    wlb=config.wlb, 
    model_name="ngc_transformer"
)


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


def compute_embeddings_from_tokens(model, token_ids):
    """
    Convert token IDs to embeddings using EmbeddingSynapse.
    
    Args:
        model: NGCTransformer model
        token_ids: Token indices (batch_size, seq_len)
    
    Returns:
        embeddings: Word + position embeddings (batch_size*seq_len, embed_dim)
    """
    batch_size = token_ids.shape[0]
    seq_len = token_ids.shape[1]
    
    # Create a temporary EmbeddingSynapse for token→embedding conversion
    # Use the same parameters as model.embedding.W_embed
    from utils.embed_utils import EmbeddingSynapse
    temp_embed_synapse = EmbeddingSynapse(
        "temp_embed",
        vocab_size=model.vocab_size,
        seq_len=seq_len,
        embed_dim=model.n_embed,
        batch_size=batch_size,
        pos_learnable=config.pos_learnable,
        eta=config.eta,
        optim_type=config.optim_type,
        key=jax.random.PRNGKey(42)
    )
    
    # Convert token IDs to embeddings
    temp_embed_synapse.inputs.set(token_ids)
    temp_embed_synapse.advance_state()
    
    # Get embeddings: shape (batch_size, seq_len, embed_dim)
    embeddings_3d = temp_embed_synapse.outputs.get()
    
    # Reshape to (batch_size*seq_len, embed_dim) for z_embed
    embeddings_2d = embeddings_3d.reshape(batch_size * seq_len, model.n_embed)
    
    return embeddings_2d


def debug_model_internals(model, input_seq, embeddings=None, step=0):
    """
    Inspect internal model states to diagnose why outputs are identical.
    """
    print(f"\n[DEEP DEBUG] Step {step}:")
    print(f"  Input shape: {input_seq.shape}")
    print(f"  Input tokens (first 10): {input_seq[0, :10]}")
    
    if embeddings is not None:
        print(f"  Embeddings shape: {embeddings.shape}, mean: {jnp.mean(embeddings):.6f}, std: {jnp.std(embeddings):.6f}, max: {jnp.max(embeddings):.6f}")
    
    # Check embedding layer outputs
    try:
        emb_output = model.embedding.e_embed.mu.get()
        print(f"  Embedding mu shape: {emb_output.shape}, mean: {jnp.mean(emb_output):.6f}, std: {jnp.std(emb_output):.6f}, max: {jnp.max(emb_output):.6f}")
    except Exception as e:
        print(f"  Embedding mu: Error - {e}")
    
    # Check z_embed
    try:
        z_embed = model.embedding.z_embed.zF.get()
        print(f"  z_embed zF shape: {z_embed.shape}, mean: {jnp.mean(z_embed):.6f}, std: {jnp.std(z_embed):.6f}, max: {jnp.max(z_embed):.6f}")
    except Exception as e:
        print(f"  z_embed: Error - {e}")
    
    # Check block outputs
    for i, block in enumerate(model.blocks):
        try:
            attn_mu = block.attention.e_attn.mu.get()
            print(f"  Block {i} e_attn mu: mean={jnp.mean(attn_mu):.6f}, std={jnp.std(attn_mu):.6f}")
        except Exception as e:
            print(f"  Block {i} attention: Error - {e}")
    for i, block in enumerate(model.blocks):
        try:
            attn_mu = block.attention.e_mlp.mu.get()
            print(f"  Block {i} e_mlp mu: mean={jnp.mean(attn_mu):.6f}, std={jnp.std(attn_mu):.6f}")
        except Exception as e:
            print(f"  Block {i} attention: Error - {e}")
    for i, block in enumerate(model.blocks):
        try:
            attn_mu = block.attention.e_emlp1.mu.get()
            print(f"  Block {i} e_e mu: mean={jnp.mean(attn_mu):.6f}, std={jnp.std(attn_mu):.6f}")
        except Exception as e:
            print(f"  Block {i} attention: Error - {e}")
    
    # Check output layer
    try:
        out_mu = model.output.e_out.mu.get()
        print(f"  Output mu shape: {out_mu.shape}, mean: {jnp.mean(out_mu):.6f}, std: {jnp.std(out_mu):.6f}")
    except Exception as e:
        print(f"  Output mu: Error - {e}")
    
    # Check output z
    try:
        out_z = model.output.z_out.zF.get()
        print(f"  Output z_out zF: mean={jnp.mean(out_z):.6f}, std={jnp.std(out_z):.6f}, max={jnp.max(out_z):.6f}")
    except Exception as e:
        print(f"  Output z_out: Error - {e}")


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    seq_len: int = None,
    temperature: float = 1.0,
    key=None,
    debug=False
):
    """
    Generate text using the model and provided tokenizer.
    FIXED: Properly process tokens through embedding layer before running model.
    
    CRITICAL FIX: Token IDs must go through W_embed to get embeddings before
    being used as input to z_embed. Otherwise all activations are zero.
    """
    if seq_len is None:
        seq_len = config.seq_len
    
    # Encode prompt - returns jnp.ndarray for both backends
    prompt_ids = tokenizer.encode(prompt)
    print(f"[DEBUG] Prompt tokens shape: {prompt_ids.shape}, tokens: {prompt_ids[:20]}")
    
    # Match the checkpoint batch size so loaded component shapes stay consistent.
    if prompt_ids.ndim == 1:
        prompt_tensor = jnp.repeat(prompt_ids[None, :], generation_batch_size, axis=0)
    else:
        prompt_tensor = prompt_ids

    current_tokens = prompt_tensor
    current_key = key

    for step in range(max_new_tokens):
        # IMPORTANT: Use FULL accumulated context (sliding window if needed)
        if current_tokens.shape[1] > seq_len:
            input_seq = current_tokens[:, -seq_len:]
            print(f"[DEBUG] Step {step}: Context exceeds seq_len, using sliding window. Shape: {input_seq.shape}")
        else:
            input_seq = current_tokens
            print(f"[DEBUG] Step {step}: Using full context. Shape: {input_seq.shape}")

        # Pad to exactly seq_len if needed
        if input_seq.shape[1] < seq_len:
            pad_len = seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=0)

        # CORRECT FLOW: Convert token IDs to embeddings externally
        embeddings = compute_embeddings_from_tokens(model, input_seq)
        print(f"[DEBUG] Step {step}: Embeddings computed - shape: {embeddings.shape}, mean: {jnp.mean(embeddings):.6f}, std: {jnp.std(embeddings):.6f}")

        # Dummy target for inference
        dummy_target = jnp.zeros((generation_batch_size * seq_len, config.vocab_size))

        # Call debug on first few steps
        if debug and (step < 2 or step == max_new_tokens - 1):
            debug_model_internals(model, input_seq, embeddings=embeddings, step=step)

        # Forward pass with embeddings
        # model.process() will call clamp_input(embeddings) which sets z_embed.j
        y_mu_inf, y_mu, EFE = model.process(embeddings, dummy_target, adapt_synapses=False)
        
        if y_mu is None:
            print(f"[ERROR] Step {step}: model.process() returned None for y_mu!")
            logits = jnp.zeros((generation_batch_size, seq_len, config.vocab_size))
        else:
            logits = y_mu.reshape(generation_batch_size, seq_len, config.vocab_size)

        # Extract logits at the LAST real token position
        last_pos = min(current_tokens.shape[1], seq_len) - 1
        next_logits = logits[0, last_pos, :] / temperature
        
        print(f"[DEBUG] Step {step}: Last position: {last_pos}, Logit stats - mean: {jnp.mean(next_logits):.4f}, max: {jnp.max(next_logits):.4f}, min: {jnp.min(next_logits):.4f}")

        # Sample or take argmax
        if current_key is not None:
            probs = jax.nn.softmax(next_logits)
            current_key, subkey = jax.random.split(current_key)
            next_token = jax.random.choice(subkey, a=config.vocab_size, p=probs)
        else:
            next_token = jnp.argmax(next_logits)

        print(f"[DEBUG] Step {step}: Generated token ID: {next_token}")

        # Append new token to accumulated context
        current_tokens = jnp.concatenate(
            [current_tokens, jnp.full((generation_batch_size, 1), next_token, dtype=current_tokens.dtype)],
            axis=1,
        )

    # Decode generated IDs back to text
    generated_ids = current_tokens[0].tolist()
    return tokenizer.decode(generated_ids)


# Example usage
if __name__ == "__main__":
    # Test 1: Short prompt
    print("="*80)
    print("TEST 1: Short prompt")
    print("="*80)
    prompt1 = "The king said: "
    generated1 = generate_text(model, tokenizer, prompt=prompt1, max_new_tokens=5, temperature=0.8, key=jax.random.PRNGKey(42), debug=True)
    print("\n" + "="*80 + "\n")
    print("GENERATED TEXT (Prompt 1):")
    print(generated1)
    print("\n" + "="*80 + "\n")
    
    # Test 2: Long prompt  
    print("="*80)
    print("TEST 2: Long prompt")
    print("="*80)
    prompt2 = "You common cry of curs! whose breath I hate As reek o' the rotten fens..."
    generated2 = generate_text(model, tokenizer, prompt=prompt2, max_new_tokens=5, temperature=0.8, key=jax.random.PRNGKey(42), debug=True)
    print("\n" + "="*80 + "\n")
    print("GENERATED TEXT (Prompt 2):")
    print(generated2)