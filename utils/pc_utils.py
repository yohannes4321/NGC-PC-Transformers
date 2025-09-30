import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import numpy as np
from torch.amp import autocast
from contextlib import nullcontext
from predictive_coding.config import GPTConfig
from utils.attention_utils import apply_flash_attention, apply_standard_attention
from jax import numpy as jnp
from ngclearn.components import GaussianErrorCell as ErrorCell,HebbianSynapse
def compute_DVL(attn_v, requires_update):
    B, H, T, D = attn_v.shape
    device = attn_v.device
    x = attn_v.transpose(0, 1).flatten(2, 3)  # (H, B, T*D)
    x = x.transpose(0, 1)  
    x = F.normalize(x, p=2, dim=-1)
    s_m = torch.bmm(x, x.transpose(1, 2))  
    s_m = s_m.mean(dim=0)  
    identity = torch.eye(H, device=attn_v.device)  
    corr = s_m - identity  
    dvl = (corr ** 2).mean()  
    dvl_grad = torch.zeros_like(attn_v, device=device)
    try:
        if requires_update:
            dvl_grad = torch.autograd.grad(dvl, attn_v, retain_graph=True)[0]
    except Exception as e:
        print(f"Error computing diversity gradient: {e}")
    return dvl_grad

def get_head_similarity(mu_heads):
    B, H, T, D = mu_heads.shape
    x = mu_heads.transpose(0, 1).flatten(2, 3)  # (H, B, T*D) 
    x = F.normalize(x, p=2, dim=-1)
    corr = torch.bmm(x, x.transpose(1, 2))  
    mask = ~torch.eye(corr.size(1), device=corr.device).bool()
    s_v = corr[:, mask].mean(dim= -1)
    corr = s_v.abs().mean(dim=-1)  

    return corr.detach().cpu()
    
def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, embedding_size, device = device)


def step_embed(t, T, target, layer, layer_type, input_ids, position_ids, local_lr, clamp_value, energy_fn_name, is_holding_error, requires_update, layer_norm, mu_word_cache=None, mu_pos_cache=None):
    """
    Perform a predictive coding update step for the embedding layer using GaussianErrorCell and HebbianSynapse.
    Uses HebbianSynapse for weight updates and supports vectorized updates and caching for inference.

    Args:
        t (int): Current inference step.
        T (int): Total number of inference steps.
        target (torch.Tensor): Target activity tensor, shape [batch_size, seq_len, n_embed].
        layer (dict): Dictionary with 'word' and 'pos' embedding layers.
        layer_type (str): Layer type string.
        input_ids (torch.Tensor): Input token IDs, shape [batch_size, seq_len].
        position_ids (torch.Tensor): Position IDs, shape [batch_size, seq_len].
        local_lr (float): Local learning rate.
        clamp_value (float): Value to clamp updates.
        energy_fn_name (str): Name of energy function.
        is_holding_error (bool): Whether to accumulate errors.
        requires_update (bool): Whether to update weights.
        layer_norm: Layer normalization module (nn.Module instance).
        mu_word_cache, mu_pos_cache: Optional cached values for inference.

    Returns:
        tuple: (mu, mu_word, mu_pos, error)
    """
    word_layer = layer["word"]
    pos_layer = layer["pos"]
    device = target.device
    
    use_amp = target.is_cuda
    autocast_ctx = autocast('cuda') if use_amp else nullcontext()

    # Clip input_ids and position_ids to valid ranges
    vocab_size = word_layer.weight.size(0)  # 50259
    max_pos = pos_layer.weight.size(0)  # Should be 64
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
    if position_ids.max() >= max_pos:
        position_ids = torch.clamp(position_ids, max=max_pos-1)
    
    batch_size, seq_len = input_ids.shape  # e.g., [8, 36]
    embed_dim = target.size(-1)  # 12
    n_units = seq_len * embed_dim  # e.g., 36 * 12 = 432
    
    with autocast_ctx:
        if requires_update or mu_word_cache is None or mu_pos_cache is None:
            mu_word = word_layer(input_ids)  # [8, 36, 12]
            mu_pos = pos_layer(position_ids)  # [8, 36, 12]
        else:
            mu_word = mu_word_cache
            mu_pos = mu_pos_cache
        mu = mu_word + mu_pos  # [8, 36, 12]
        mu_norm = layer_norm(mu)  # [8, 36, 12]
    
        # Convert to JAX for ErrorCell
        target_jax = jnp.array(target.detach().cpu().numpy())  # [8, 36, 12]
        mu_norm_jax = jnp.array(mu_norm.detach().cpu().numpy())  # [8, 36, 12]
        
        # Flatten for ErrorCell
        target_jax_flat = target_jax.reshape(batch_size, -1)  # [8, 432]
        mu_norm_jax_flat = mu_norm_jax.reshape(batch_size, -1)  # [8, 432]
        
        # Initialize GaussianErrorCell
        error_cell = ErrorCell(
            "embed_error",
            n_units=n_units,  # 36 * 12 = 432
            batch_size=batch_size
        )
        
        # Set inputs for ErrorCell
        error_cell.mu.set(mu_norm_jax_flat)
        error_cell.target.set(target_jax_flat)
        error_cell.Sigma.set(jnp.ones((1, 1)))
        error_cell.modulator.set(jnp.ones_like(mu_norm_jax_flat))
        error_cell.mask.set(jnp.ones_like(mu_norm_jax_flat))
        
        # Compute error
        error_cell.advance_state(
            dt=1.0,
            mu=error_cell.mu.value,
            target=error_cell.target.value,
            Sigma=error_cell.Sigma.value,
            modulator=error_cell.modulator.value,
            mask=error_cell.mask.value
        )
        
        # Convert error back to PyTorch
        error = torch.from_numpy(np.array(error_cell.dmu.value)).to(device)
        error = error.view(batch_size, seq_len, embed_dim)  # [8, 36, 12]
        
        # Debug shapes
        # print("input_ids shape:", input_ids.shape)
        # print("mu_word shape:", mu_word.shape)
        # print("mu_pos shape:", mu_pos.shape)
        # print("mu shape:", mu.shape)
        # print("mu_norm shape:", mu_norm.shape)
        # print("target shape:", target.shape)
        # print("target_jax_flat shape:", target_jax_flat.shape)
        # print("mu_norm_jax_flat shape:", mu_norm_jax_flat.shape)
        # print("error shape:", error.shape)
        
        if not requires_update:
            if t == T - 1:
                finalize_step(mu, target, error, t, layer_type, energy_fn_name, is_holding_error)
            return mu, mu_word, mu_pos, error
        
        if requires_update:
            # Convert input_ids and position_ids to one-hot
            input_one_hot = F.one_hot(input_ids, num_classes=vocab_size).float()  # [8, 36, 50259]
            position_one_hot = F.one_hot(position_ids, num_classes=max_pos).float()  # [8, 36, 64]
            
            # Debug one-hot sizes
            # print("input_one_hot shape:", input_one_hot.shape)
            # print("input_one_hot size:", input_one_hot.numel())
            # print("position_one_hot shape:", position_one_hot.shape)
            # print("position_one_hot size:", position_one_hot.numel())
            
            # Flatten for HebbianSynapse
            input_one_hot_flat = input_one_hot.reshape(batch_size * seq_len, -1)  # [288, 50259]
            position_one_hot_flat = position_one_hot.reshape(batch_size * seq_len, -1)  # [288, 64]
            error_flat = error.reshape(batch_size * seq_len, embed_dim)  # [288, 12]
            
            # Initialize HebbianSynapse
            word_synapse = HebbianSynapse(
                name="word_hebb",
                shape=(vocab_size, embed_dim),  # [50259, 12]
                eta=local_lr,
                preact="identity",
                postact="identity",
                bias_init=None  # Disable biases
            )
            pos_synapse = HebbianSynapse(
                name="pos_hebb",
                shape=(max_pos, embed_dim),  # [64, 12]
                eta=local_lr,
                preact="identity",
                postact="identity",
                bias_init=None  # Disable biases
            )
            
            # Initialize weights directly
            word_weights_init = jnp.array(word_layer.weight.data.detach().cpu().numpy())
            pos_weights_init = jnp.array(pos_layer.weight.data.detach().cpu().numpy())
            
            # Set pre and post inputs
            input_one_hot_flat_jax = jnp.array(input_one_hot_flat.cpu().numpy())
            position_one_hot_flat_jax = jnp.array(position_one_hot_flat.cpu().numpy())
            error_flat_jax = jnp.array(error_flat.cpu().numpy())
            
            # Compute Hebbian updates
            word_dW, _ = word_synapse._compute_update(
                w_bound=word_synapse.w_bound,
                is_nonnegative=word_synapse.is_nonnegative,
                sign_value=word_synapse.sign_value,
                prior_type=word_synapse.prior_type,
                prior_lmbda=word_synapse.prior_lmbda,
                pre_wght=word_synapse.pre_wght,
                post_wght=word_synapse.post_wght,
                pre=input_one_hot_flat_jax,
                post=error_flat_jax,
                weights=word_weights_init
            )
            pos_dW, _ = pos_synapse._compute_update(
                w_bound=pos_synapse.w_bound,
                is_nonnegative=pos_synapse.is_nonnegative,
                sign_value=pos_synapse.sign_value,
                prior_type=pos_synapse.prior_type,
                prior_lmbda=pos_synapse.prior_lmbda,
                pre_wght=pos_synapse.pre_wght,
                post_wght=pos_synapse.post_wght,
                pre=position_one_hot_flat_jax,
                post=error_flat_jax,
                weights=pos_weights_init
            )
            
            # # # Apply optimization step (using SGD or Adam)
            # word_opt_params, [word_weights_new] = word_synapse.opt(
            #     word_synapse.opt_params.value, [word_weights_init], [word_dW]
            # )
            # pos_opt_params, [pos_weights_new] = pos_synapse.opt(
            #     pos_synapse.opt_params.value, [pos_weights_init], [pos_dW]
            # )
            
            # Enforce constraints
           # Enforce constraints using static method
            # word_weights_new = HebbianSynapse._enforce_constraints(
            #     word_weights_new, word_synapse.w_bound, word_synapse.is_nonnegative
            # )
            # pos_weights_new = HebbianSynapse._enforce_constraints(
            #     pos_weights_new, pos_synapse.w_bound, pos_synapse.is_nonnegative
            # )
            word_weights_new = word_weights_init + word_synapse.eta * word_dW
            pos_weights_new = pos_weights_init + pos_synapse.eta * pos_dW
            # Calculate weight deltas
            word_delta = torch.from_numpy(np.array(word_weights_new - word_weights_init)).to(device)
            pos_delta = torch.from_numpy(np.array(pos_weights_new - pos_weights_init)).to(device)
            word_delta = torch.clamp(word_delta, -clamp_value, clamp_value)
            pos_delta = torch.clamp(pos_delta, -clamp_value, clamp_value)
            
            # Apply updates to PyTorch layers
            with torch.no_grad():
                word_layer.weight.data.add_(word_delta)
                pos_layer.weight.data.add_(pos_delta)
            
            # Debug weight shapes
            # print("input_one_hot_flat shape:", input_one_hot_flat.shape)
            # print("position_one_hot_flat shape:", position_one_hot_flat.shape)
            # print("error_flat shape:", error_flat.shape)
            # print("word_layer.weight shape:", word_layer.weight.shape)
            # print("pos_layer.weight shape:", pos_layer.weight.shape)
        
        if t == T - 1:
            finalize_step(mu, target, error, t, layer_type, energy_fn_name, is_holding_error)
  
        return mu, mu_word, mu_pos, error


def step_linear(t, T, target, x, layer, W_latents, layer_type, local_lr, clamp_value, use_lateral, is_holding_error, energy_fn_name, update_bias, requires_update, td_err, layer_norm):
    """
    Perform a predictive coding update step for a linear (fully connected) layer.

    Args:
        t (int): Current inference step.
        T (int): Total number of inference steps.
        target (torch.Tensor): Target activity tensor.
        x (torch.Tensor): Current activity tensor.
        layer (nn.Module): Linear layer.
        W_latents (dict): Lateral weights.
        layer_type (str): Layer type string.
        local_lr (float): Local learning rate.
        clamp_value (float): Value to clamp updates.
        use_lateral (bool): Whether to use lateral connections.
        is_holding_error (bool): Whether to accumulate errors.
        energy_fn_name (str): Name of energy function.
        update_bias (bool): Whether to update bias.
        requires_update (bool): Whether to update weights.
        td_err: this is the error that comes from the above layer prediction.
        bu_err: an error that comes from the bottom layer.
    Returns:
        tuple: (updated activity tensor, predicted output tensor)
    """
    device = x.device
    use_amp = target.is_cuda
    autocast_ctx = autocast('cuda') if use_amp else nullcontext()
    
    with autocast_ctx:
        if layer_norm is not None and layer_type == "fc1":
           x = layer_norm(x)
        elif layer_type == "fc2":
           x = F.gelu(x)
           
        mu = layer(x)
        if layer_type == "fc1":
            mu = F.gelu(mu)
        elif layer_norm is not None and layer_type in ["linear_attn", "fc2"]:
              mu = layer_norm(mu)
        if layer_type=="linear_output":
          bu_err= target - F.softmax(mu, dim=-1) 
        else:    
          bu_err = target - mu 
          
        error_proj= bu_err @layer.weight # project the error 
       
        if td_err is not None:
           error= error_proj- td_err
        else:
           error= error_proj     
          

        if use_lateral and layer_type in W_latents:
           W_latent = W_latents[layer_type].to(device) 
           x_latent = torch.einsum("bsh,hv->bsv", x, W_latent)
           delta_x = error + x_latent
           x = x + local_lr * delta_x

           if requires_update:
              anti_hebbian_latent = -torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
              W_latents[layer_type] = W_latents[layer_type] + local_lr * anti_hebbian_latent
              W_latents[layer_type].data = F.normalize(W_latents[layer_type].data, p=2, dim=1)
    
        else:
          x= x + local_lr * error 
    
        x = torch.clamp(x, clamp_value, clamp_value)
    
    # PC Update W_layer
    if requires_update:
        delta_W = local_lr * torch.einsum("bsv, bsh -> vh", bu_err, x.detach())
        delta_W = torch.clamp(delta_W, -0.01, 0.01)
        layer.weight.data.add_(delta_W)
        if layer.bias is not None and update_bias:
            delta_b = local_lr * bu_err.mean(dim=(0, 1))
            delta_b = torch.clamp(delta_b, -0.01, 0.01)
            layer.bias.data.add_(delta_b)

    x = torch.clamp(x, -clamp_value, clamp_value)
    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error)

    return x, mu, bu_err

def step_attn(t, T, target, x, W_latents, proj_layers, layer_type, local_lr, clamp_value, use_lateral, is_holding_error, energy_fn_name, update_bias, requires_update, layer_instance, num_heads, n_embed, la, td_err,layer_norm, flash=False):
        assert proj_layers is not None, "proj_layers dict is required for attention"
        device = x.device
        x=layer_norm(x)
        q_proj = proj_layers.get("q_proj", None)
        k_proj = proj_layers.get("k_proj", None)
        v_proj = proj_layers.get("v_proj", None)
        assert all(p is not None for p in (q_proj, k_proj, v_proj)), "Missing Q/K/V projections in dict"    
        
        use_amp = target.is_cuda
        autocast_ctx = autocast('cuda') if use_amp else nullcontext()
        
        batch_size, seq_len, embed_dim = target.shape
        head_dim = n_embed // num_heads
        la = la * math.sqrt(1.0 / head_dim)

        with autocast_ctx:      
            Q= q_proj(x)
            K= k_proj(x)
            V= v_proj(x)
            
            Q = Q.view(batch_size, num_heads, seq_len, head_dim)
            K = K.view(batch_size, num_heads, seq_len, head_dim)
            V = V.view(batch_size, num_heads, seq_len, head_dim)
            
            #create causal mask (1=keep, 0=mask)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

            # !! Causal Mask
            if flash:
                # TODO: add support for causal masking in flash attention
                mu_heads = apply_flash_attention(Q, K, V)
            else:
                mu_heads = apply_standard_attention(Q, K, V, mask=causal_mask)

            dvl_grad = compute_DVL(mu_heads, requires_update)
            if dvl_grad is not None:
               dvl_grad = dvl_grad.to(device)
            dvl_norm = dvl_grad.norm().item() if dvl_grad is not None else 0.0
            similarity = get_head_similarity(mu_heads)
            mu = mu_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
     
            bu_err = target - mu  # B, T, D
            if td_err is not None:
               error= bu_err - td_err
            else:
                error = bu_err  
          
            if dvl_grad is not None:
                B, H, T, D = dvl_grad.shape               # matches compute_DVL output
                dvl_projected = dvl_grad.permute(0, 2, 1, 3).contiguous().view(B, T, H*D)  # [B, T, embed_dim]
                dvl_projected=dvl_projected.clamp(-1e-3, 1e-3)
                error = error + la * dvl_projected
                
        if layer_instance is not None:
            setattr(layer_instance, '_head_similarity', similarity)
            setattr(layer_instance, '_head_similarity_avg', similarity.mean().item())
            setattr(layer_instance, '_head_similarity_max', similarity.max().item())
        
        if use_lateral and layer_type in W_latents:
            W_latent = W_latents[layer_type].to(device) 
            x_latent = x @ W_latent
            delta_x = error + x_latent
            x = x + local_lr * delta_x 

            if requires_update:
               anti_hebbian_latent = - torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
               W_latents[layer_type] =W_latent + local_lr * anti_hebbian_latent
               W_latents[layer_type].data = F.normalize(W_latents[layer_type].data, p=2, dim=1)
        else:
            x= x+ local_lr * error

        x = torch.clamp(x, -clamp_value, clamp_value)

        # PC update W_latent
        if requires_update:
            for proj in (q_proj, k_proj, v_proj):
                delta_W = local_lr * torch.einsum("bsv, bsh -> vh", bu_err, x.detach())
                delta_W = torch.clamp(delta_W, -0.01, 0.01)
                proj.weight.data.add_(delta_W)
                if proj.bias is not None and update_bias:
                    delta_b = local_lr * bu_err.mean(dim=(0, 1))
                    delta_b = torch.clamp(delta_b, -0.01, 0.01)
                    delta_b = delta_b.view(-1)
                    proj.bias.data.add_(delta_b)
 
        if t == T - 1:
            finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error)
     
        return x, mu, bu_err
    
ENERGY_FUNCTIONS = {
    "scaled_mse": lambda mu, x: ((mu - x) ** 2).mean(dim=-1) * 0.05,
    "mse": lambda mu, x: ((mu - x) ** 2).mean(dim=-1),
    "pc_e": lambda mu, x: ((mu - x) ** 2) * 0.5,    
    "l1": lambda mu, x: (mu - x).abs().mean(dim=-1),
    "cosine": lambda mu, x: 1 - F.cosine_similarity(mu, x, dim=-1),
    "kld": lambda mu, x: torch.clamp(F.kl_div(
        mu.log_softmax(dim=-1),
        x,
        reduction='batchmean'
    ), min=0.0, max=100.0)
}

def energy_fn(mu: torch.Tensor, x: torch.Tensor,energy_fn_name: str) -> torch.Tensor:
    """
    Compute the energy (error) between predicted and target activity using the specified function.

    Args:
        mu (torch.Tensor): Predicted activity.
        x (torch.Tensor): Target activity.
        energy_fn_name (str): Name of energy function ('scaled_mse', 'mse', 'l1', 'cosine', 'kld').
    Returns:
        torch.Tensor: Computed energy value.
    """
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(f"Unknown energy function: {energy_fn_name}. Choose from {list(ENERGY_FUNCTIONS.keys())}")
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)

def finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error = False):
    """
    Finalize a predictive coding inference step by computing energy and error statistics.

    Args:
        mu (torch.Tensor): Predicted activity.
        target (torch.Tensor): Target activity.
        error (torch.Tensor): Error tensor.
        t (int): Current inference step.
        layer_type (str): Layer type string.
        energy_fn_name (str): Name of energy function.
        is_holding_error (bool): Whether to accumulate errors.
    Returns:
        tuple: (energy value, list of error statistics)
    """
    device = mu.device
    target = target.to(device)
    error = error.to(device)
    energy = energy_fn(mu, target,energy_fn_name).mean().item() if is_holding_error else None
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors
    
def ids_to_one_hot(input_ids, vocab_size):
    """
    Convert input token IDs to one-hot encoded tensor.

    Args:
        input_ids (torch.Tensor): Tensor of shape (B, S) with token IDs.
        vocab_size (int): Size of the vocabulary.
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (B, S, vocab_size).
    """
    device = input_ids.device

    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
    
    return F.one_hot(input_ids, num_classes=vocab_size).float().to(device)

def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()