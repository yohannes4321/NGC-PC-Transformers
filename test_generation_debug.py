"""
Debug script to diagnose why generation produces identical text
"""

from model import NGCTransformer
import jax
import jax.numpy as jnp
from config import Config as config
from data_preprocess.tokenizer import get_tokenizer, BPETokenizer
from pathlib import Path

print("=" * 80)
print("NGC-PC-Transformers - Generation Debug")
print("=" * 80)

# Step 1: Load model
print("\n[1/5] Loading model from exp/...")
dkey = jax.random.PRNGKey(0)
try:
    model = NGCTransformer(
        dkey, 
        batch_size=1,
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
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Load tokenizer
print("\n[2/5] Loading tokenizer...")
tokenizer = get_tokenizer(config)
if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is None:
    vocab_file = Path("data_preprocess/outputs/tokenizer/bpe_tokenizer.json")
    if vocab_file.exists():
        tokenizer.load_tokenizer(str(vocab_file))
        print(f"✓ Tokenizer loaded (vocab size: {tokenizer.get_vocab_size()})")
    else:
        print(f"✗ Tokenizer file not found at {vocab_file}")
        exit(1)

# Step 3: Test encoding/decoding
print("\n[3/5] Testing tokenizer...")
test_prompt = "The king said:"
encoded = tokenizer.encode(test_prompt)
print(f"  Input: '{test_prompt}'")
print(f"  Encoded: {encoded}")
print(f"  Shape: {encoded.shape}")
decoded = tokenizer.decode(encoded.tolist())
print(f"  Decoded: '{decoded}'")

# Step 4: Test model forward pass
print("\n[4/5] Testing model forward pass...")
try:
    # Create dummy input and target
    input_ids = encoded[None, :]  # Add batch dimension
    if input_ids.shape[1] < config.seq_len:
        pad_len = config.seq_len - input_ids.shape[1]
        input_ids = jnp.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=0)
    
    dummy_target = jnp.zeros((1 * config.seq_len, config.vocab_size))
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Target shape: {dummy_target.shape}")
    
    y_mu_inf, y_mu, EFE = model.process(input_ids, dummy_target, adapt_synapses=False)
    
    print(f"  y_mu_inf shape: {y_mu_inf.shape if hasattr(y_mu_inf, 'shape') else type(y_mu_inf)}")
    print(f"  y_mu shape: {y_mu.shape}")
    print(f"  EFE value: {EFE}")
    
    # Reshape and get logits for next token
    logits = y_mu.reshape(1, config.seq_len, config.vocab_size)
    print(f"  Logits shape: {logits.shape}")
    
    # Get top-5 predictions for next token
    next_logits = logits[0, -1, :]  # Last position
    top_k_indices = jnp.argsort(next_logits)[-5:][::-1]
    top_k_probs = jax.nn.softmax(next_logits)[top_k_indices]
    
    print(f"  Top 5 next token predictions:")
    for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
        token_id = int(idx)
        predicted_token = tokenizer.decode([token_id])
        print(f"    {i+1}. Token {token_id}: '{predicted_token}' (prob: {float(prob):.4f})")
    
    print("✓ Model forward pass successful")
except Exception as e:
    print(f"✗ Model forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Test multiple generations
print("\n[5/5] Testing generation with different temperatures...")
def quick_generate(prompt, max_tokens=10, temperature=1.0):
    prompt_ids = tokenizer.encode(prompt)
    if prompt_ids.ndim == 1:
        prompt_tensor = prompt_ids[None, :]
    else:
        prompt_tensor = prompt_ids
    
    current_tokens = prompt_tensor
    key = jax.random.PRNGKey(42)
    
    for _ in range(max_tokens):
        if current_tokens.shape[1] > config.seq_len:
            input_seq = current_tokens[:, -config.seq_len:]
        else:
            input_seq = current_tokens
        
        if input_seq.shape[1] < config.seq_len:
            pad_len = config.seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=0)
        
        dummy_target = jnp.zeros((1 * config.seq_len, config.vocab_size))
        y_mu_inf, y_mu, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
        logits = y_mu.reshape(1, config.seq_len, config.vocab_size)
        
        next_logits = logits[0, -1, :] / temperature
        probs = jax.nn.softmax(next_logits)
        
        key, subkey = jax.random.split(key)
        next_token = jax.random.choice(subkey, a=config.vocab_size, p=probs)
        
        current_tokens = jnp.concatenate([current_tokens, jnp.array([[next_token]], dtype=jnp.int32)], axis=1)
    
    return tokenizer.decode(current_tokens[0].tolist())

temperatures = [0.5, 1.0, 1.5]
prompt = "The king said:"

for temp in temperatures:
    result = quick_generate(prompt, max_tokens=15, temperature=temp)
    print(f"  Temperature {temp}: {result[:100]}...")

print("\n" + "=" * 80)
print("If all tests passed but generation is identical:")
print("  → The model may not have trained properly")
print("  → Check that training loss decreased over epochs")
print("  → Try: python train.py (again)")
print("=" * 80)
