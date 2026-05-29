from model import NGCTransformer
import jax
import jax.numpy as jnp
import numpy as np
from config import Config as config
from data_preprocess.data_loader import DataLoader
from data_preprocess.tokenizer import get_tokenizer, BPETokenizer
from pathlib import Path

def stat(name, x):
    if x is None:
        print(f"{name:45s} is None")
        return
    x = jnp.asarray(x)
    print(
        f"{name:45s}"
        f" shape={str(x.shape):15s}"
        f" mean={jnp.mean(x):10.6f}"
        f" std={jnp.std(x):10.6f}"
        f" max={jnp.max(x):10.6f}"
        f" min={jnp.min(x):10.6f}"
    )

def trace_model(model):
    print("\n" + "="*100)
    print(f"EXHAUSTIVE MODEL TRACE (T={model.T})")
    print("="*100)

    # --- 1. EMBEDDING ---
    print("\n[EMBEDDING]")
    c = model.embedding
    stat("  z_embed.z", c.z_embed.z.get())
    stat("  z_embed.zF", c.z_embed.zF.get())
    stat("  W_embed.inputs", c.W_embed.inputs.get())
    stat("  W_embed.outputs", c.W_embed.outputs.get())
    stat("  e_embed.mu", c.e_embed.mu.get())
    stat("  e_embed.target", c.e_embed.target.get())

    # --- 2. BLOCKS ---
    for i, b in enumerate(model.blocks):
        print(f"\n[BLOCK {i} - ATTENTION]")
        stat("  z_qkv.z", b.attention.z_qkv.z.get())
        stat("  z_qkv.zF", b.attention.z_qkv.zF.get())
        stat("  W_q.inputs", b.attention.W_q.inputs.get())
        stat("  W_q.outputs", b.attention.W_q.outputs.get())
        stat("  W_k.inputs", b.attention.W_k.inputs.get())
        stat("  W_k.outputs", b.attention.W_k.outputs.get())
        stat("  W_v.inputs", b.attention.W_v.inputs.get())
        stat("  W_v.outputs", b.attention.W_v.outputs.get())
        stat("  attn_block.inputs_q", b.attention.attn_block.inputs_q.get())
        stat("  attn_block.outputs", b.attention.attn_block.outputs.get())
        stat("  z_attn.z", b.attention.z_attn.z.get())
        stat("  z_attn.zF", b.attention.z_attn.zF.get())
        stat("  W_attn_out.inputs", b.attention.W_attn_out.inputs.get())
        stat("  W_attn_out.outputs", b.attention.W_attn_out.outputs.get())
        stat("  e_qkv.mu", b.attention.e_qkv.mu.get())
        stat("  e_qkv.target", b.attention.e_qkv.target.get())
        stat("  e_attn.mu", b.attention.e_attn.mu.get())
        stat("  e_attn.target", b.attention.e_attn.target.get())

        print(f"\n[BLOCK {i} - MLP]")
        stat("  z_mlp.z", b.mlp.z_mlp.z.get())
        stat("  z_mlp.zF", b.mlp.z_mlp.zF.get())
        stat("  W_mlp1.inputs", b.mlp.W_mlp1.inputs.get())
        stat("  W_mlp1.outputs", b.mlp.W_mlp1.outputs.get())
        stat("  z_mlp2.z", b.mlp.z_mlp2.z.get())
        stat("  z_mlp2.zF", b.mlp.z_mlp2.zF.get())
        stat("  W_mlp2.inputs", b.mlp.W_mlp2.inputs.get())
        stat("  W_mlp2.outputs", b.mlp.W_mlp2.outputs.get())
        stat("  e_mlp1.mu", b.mlp.e_mlp1.mu.get())
        stat("  e_mlp1.target", b.mlp.e_mlp1.target.get())
        stat("  e_mlp.mu", b.mlp.e_mlp.mu.get())
        stat("  e_mlp.target", b.mlp.e_mlp.target.get())

    # --- 3. OUTPUT ---
    print("\n[OUTPUT]")
    stat("  z_out.z", model.output.z_out.z.get())
    stat("  z_out.zF", model.output.z_out.zF.get())
    stat("  W_out.inputs", model.output.W_out.inputs.get())
    stat("  W_out.outputs", model.output.W_out.outputs.get())
    stat("  e_out.mu", model.output.e_out.mu.get())
    stat("  e_out.target", model.output.e_out.target.get())
    stat("  z_target.z", model.z_target.z.get())
    stat("  z_actfx.z", model.z_actfx.z.get())
    stat("  z_actfx.zF", model.z_actfx.zF.get())

    # --- 4. PROJECTION ---
    

    print("\n" + "="*100 + "\n")

def weight_stats(model):

    print("\n=========== WEIGHTS ===========\n")

    stat(
        "W_embed.word_weights",
        model.embedding.W_embed.word_weights.get()
    )

    for i, block in enumerate(model.blocks):

        print(f"\n------ BLOCK {i} ------")

        stat("W_q", block.attention.W_q.weights.get())
        stat("W_k", block.attention.W_k.weights.get())
        stat("W_v", block.attention.W_v.weights.get())

        stat(
            "W_attn_out",
            block.attention.W_attn_out.weights.get()
        )

        stat("W_mlp1", block.mlp.W_mlp1.weights.get())
        stat("W_mlp2", block.mlp.W_mlp2.weights.get())

    stat("W_out", model.output.W_out.weights.get())

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
    seq_len: int = 8,
    temperature: float = 1.0,
    key=None
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

    for _ in range(max_new_tokens):
        # Truncate context to fit model's seq_len
        if current_tokens.shape[1] > seq_len:
            input_seq = current_tokens[:, -seq_len:]
        else:
            input_seq = current_tokens

        # Pad to exactly seq_len if needed (assumes token ID 0 = padding)
        if input_seq.shape[1] < seq_len:
            pad_len = seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=0)
        
        # Dummy target for inference (unused when adapt_synapses=False)
        dummy_target = jnp.zeros((model.batch_size * seq_len, config.vocab_size))  

        # Forward pass
        y_mu_inf, y_mu, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
        logits = y_mu.reshape(model.batch_size, seq_len, config.vocab_size)

        # Get logits for the last *real* token (excluding padding)
        actual_len = min(current_tokens.shape[1], seq_len)
        last_pos = actual_len - 1
        next_logits = logits[0, last_pos, :] / temperature

        # Sample or take argmax
        if current_key is not None:
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


# Example usage
if __name__ == "__main__":
    prompts = ["The king said: ", "hello bro i love you"]
    
    for i, prompt in enumerate(prompts):
        print(f"\n\n**************** PROMPT {i+1}: '{prompt}' ****************")
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=20, # Reduced for faster debugging
            seq_len=config.seq_len,        
            temperature=1.0,
            key=jax.random.PRNGKey(42)  
        )
        print(f"\nFINAL GENERATED {i+1}:\n{generated}")