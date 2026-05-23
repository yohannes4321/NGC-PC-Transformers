# Residual Connections (Skip Connections) Implementation Guide

## Overview
Residual connections (skip connections) have been implemented in the NGC-PC Transformer model to enable:
1. Better gradient flow through deep networks
2. Improved training stability
3. More direct information pathways between layers

## Architecture Changes

### 1. Block-level Residual Components (`layers/blocks.py`)

Added three new components to each transformer block:

#### a) Attention Input Residual Cell
```python
self.z_residual_attn = RateCell(
    f"{prefix}z_residual_attn", 
    n_units=n_embed, 
    tau_m=0.,  # Stateless (identity pass-through)
    act_fx="identity", 
    batch_size=batch_size * seq_len
)
```
- **Purpose**: Stores the input to the attention block
- **tau_m=0**: Makes it stateless—acts as a buffer/pass-through
- **Dimensions**: n_embed (matches attention input dimension)

#### b) MLP Input Residual Cell
```python
self.z_residual_mlp = RateCell(
    f"{prefix}z_residual_mlp", 
    n_units=n_embed, 
    tau_m=0., 
    act_fx="identity", 
    batch_size=batch_size * seq_len
)
```
- **Purpose**: Stores the input to the MLP block
- **tau_m=0**: Stateless buffer
- **Dimensions**: n_embed (matches MLP input dimension)

#### c) MLP Residual Projection Layer
```python
self.W_residual_mlp_proj = StaticSynapse(
    f"{prefix}W_residual_mlp_proj", 
    shape=(n_embed, 4 * n_embed),
    weight_init=dist.gaussian(mean=0.0, std=0.02),
    key=random.PRNGKey(42)
)
```
- **Purpose**: Projects residual from n_embed to 4*n_embed dimensions
- **Why needed**: z_mlp2 has hidden dimension 4*n_embed, but residual input is n_embed
- **Implementation**: Static synapse (non-learnable projection)

## Circuit Connections

### Forward Pass: Storing Residuals

```
# Attention block
z_qkv.z --> z_residual_attn.j --> z_residual_attn.z (stores input)
                                        |
                                    (processed by attention)
                                        |
                                    z_attn.j_td (added as residual)

# MLP block
z_mlp1.z --> z_residual_mlp.j --> z_residual_mlp.z (stores input)
                                        |
                                  (projection layer)
                                        |
                               W_residual_mlp_proj.outputs
                                        |
                                    z_mlp2.j_td (added as residual)
```

### Detailed Circuit Wiring

#### In model.py forward pass:

```python
# Store attention input
block.attention.z_qkv.z >> block.z_residual_attn.j
block.z_residual_attn.advance_state(1.)

# ... attention computation ...

# Add attention residual to output
block.z_residual_attn.z >> block.attention.z_attn.j_td

# Store MLP input  
block.mlp.z_mlp1.z >> block.z_residual_mlp.j
block.z_residual_mlp.advance_state(1.)

# ... MLP computation ...

# Project and add MLP residual to output
block.z_residual_mlp.z >> block.W_residual_mlp_proj.inputs
block.W_residual_mlp_proj.outputs >> block.mlp.z_mlp2.j_td
```

## How Residuals Work with RateCell Dynamics

In ngclearn's RateCell, the state dynamics are:

```
dz/dt = (-leak + (j + j_td)) / tau_m
```

Where:
- **j**: Bottom-up input (normal signal)
- **j_td**: Top-down input (residual addition)
- Both get summed and drive the neural dynamics

This means the residual signal is naturally integrated through the modulation system!

## Data Flow Diagram

```
═══════════════════════════════════════════════════════════════════════
                    TRANSFORMER BLOCK WITH RESIDUALS
═══════════════════════════════════════════════════════════════════════

INPUT
  │
  ├─→ z_qkv (prediction cell)
  │     │
  │     ├─→ [store in z_residual_attn.j]
  │     │
  │     └─→ W_q, W_k, W_v (project to query, key, value)
  │           │
  │           └─→ Attention Block (multi-head attention)
  │                 │
  │                 └─→ W_attn_out (output projection)
  │                       │
  │                       └─→ z_attn (output rate cell)
  │                           │
  │                           ├─ j: from error signal E_attn
  │                           └─ j_td: + z_residual_attn.z  ◄── RESIDUAL ADDED HERE
  │                                 │
  ├───────────────────────────────→ z_mlp1 (next prediction cell)
  │                                   │
  │                                   ├─→ [store in z_residual_mlp.j]
  │                                   │
  │                                   └─→ W_mlp1 (expand to 4*n_embed)
  │                                         │
  │                                         └─→ z_mlp2 (hidden state)
  │                                               │
  │                                               └─→ W_mlp2 (project back to n_embed)
  │                                                     │
  │                                                     └─→ z_mlp2 (output rate cell)
  │                                                         │
  │                                                         ├─ j: from error signal E_mlp2
  │                                                         └─ j_td: + W_residual_mlp_proj(z_residual_mlp.z)
  │                                                               │
  │                                                             ◄── RESIDUAL ADDED HERE
  │
  └────────────────────────────────────────────→ NEXT BLOCK or OUTPUT

═══════════════════════════════════════════════════════════════════════
```

## Mathematical Formulation

### Attention Residual
```
z_attn(t+1) = z_attn(t) + dt * [
    (-leak(z_attn) + j_from_error + z_residual_attn) / tau_m
]
```

Where:
- `j_from_error`: Bottom-up error signal from E_attn
- `z_residual_attn`: Skip connection from input (z_qkv)

### MLP Residual
```
z_mlp2(t+1) = z_mlp2(t) + dt * [
    (-leak(z_mlp2) + j_from_error + proj(z_residual_mlp)) / tau_m
]
```

Where:
- `j_from_error`: Bottom-up error signal from E_mlp2
- `proj(z_residual_mlp)`: Projected skip connection from input (z_mlp1 → 4*n_embed)

## Benefits

1. **Improved Gradient Flow**: Residuals provide direct paths for gradient propagation
2. **Training Stability**: Helps prevent vanishing/exploding gradients in deep networks
3. **Faster Convergence**: Information can bypass layers when needed
4. **Better Error Propagation**: Error signals can travel more directly through layers

## Implementation Notes

- **Stateless Cells (tau_m=0)**: The z_residual_* cells use tau_m=0, making them stateless pass-through buffers
- **Static Projection**: W_residual_mlp_proj is a StaticSynapse (non-learnable) to keep projections simple
- **Additive Integration**: Residuals are added via the j_td channel of RateCell dynamics
- **Dimension Handling**: The MLP residual requires projection from n_embed to 4*n_embed to match z_mlp2

## Testing & Verification

To verify residuals are working:

```python
# Check residual values during training
for block in model.blocks:
    # Attention residual
    res_attn = block.z_residual_attn.z.get()
    print(f"Block {i}: Attention residual shape {res_attn.shape}, mean {jnp.mean(res_attn):.4f}")
    
    # MLP residual  
    res_mlp = block.z_residual_mlp.z.get()
    print(f"Block {i}: MLP residual shape {res_mlp.shape}, mean {jnp.mean(res_mlp):.4f}")
    
    # MLP projected residual
    proj_mlp = block.W_residual_mlp_proj.outputs.get()
    print(f"Block {i}: Projected MLP residual shape {proj_mlp.shape}, mean {jnp.mean(proj_mlp):.4f}")
```

## Comparison: With vs Without Residuals

### Without Residuals (Before)
- Information must flow through all weight matrices
- Gradients multiply through many layers → vanishing/exploding
- Error signals take longer to propagate backward

### With Residuals (After)
- Direct pathways for information bypass
- Gradients have alternative routes through j_td
- Faster error signal propagation
- More stable training dynamics

## Future Enhancements

1. **Learnable residual scaling**: Add per-layer scaling factors (alpha parameters)
2. **Gated residuals**: Use gating mechanisms to learn when to use residuals
3. **Dense connections**: Add skip connections spanning multiple layers
4. **LayerNorm on residuals**: Normalize residuals before addition
