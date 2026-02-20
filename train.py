import os
import jax
from jax import numpy as jnp, random, device_put, jit
from jax.nn import one_hot
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model
import time

# Prefer fast 32-bit math on GPU/TPU
jax.config.update("jax_enable_x64", False)

# Optional XLA runtime tuning (can help memory / autotuning)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

# JIT-compiled helper functions for speed
@jit
def compute_one_hot(flat_targets):
    """
    JIT-compiled one-hot encoding.
    vocab_size is captured from config so it is a static constant,
    avoiding the ConcretizationTypeError.
    """
    return one_hot(flat_targets, config.vocab_size)

@jit
def compute_metrics(y_pred, y_true):
    """JIT-compiled metric computation (CE + PPL)."""
    batch_nll = measure_CatNLL(y_pred, y_true)
    batch_ce_loss = batch_nll.mean()
    batch_ppl = jnp.exp(batch_ce_loss)
    return batch_ce_loss, batch_ppl

def main():
    seq_len, batch_size, n_embed, vocab_size, n_layers, n_heads, n_iter, optim_type = config.seq_len, config.batch_size, config.n_embed, config.vocab_size, config.n_layers, config.n_heads, config.n_iter, config.optim_type
    pos_learnable= config.pos_learnable
    epoch= config.epoch
    wub= config.wub 
    wlb= config.wlb
    eta = config.eta
    T = config.n_iter
    tau_m= config.tau_m
    act_fx= config.act_fx
    dropout_rate= config.dropout_rate
    dkey = random.PRNGKey(1234)
    
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, test_loader = data_loader.load_and_prepare_data()
    
    model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=n_embed, vocab_size=vocab_size, n_layers=n_layers, n_heads=config.n_heads,
                          T=T, dt=1., tau_m=tau_m , act_fx=act_fx, eta=eta, dropout_rate= dropout_rate, exp_dir="exp",
                  loadDir= None, pos_learnable= pos_learnable, optim_type=optim_type, wub = wub, wlb= wlb, model_name="ngc_transformer" )

    def train_model(data_loader):
        total_nll, total_tokens = 0., 0
        for batch in data_loader:
            inputs = device_put(batch[0][1])
            targets = device_put(batch[1][1])
            targets_flat = compute_one_hot(targets.reshape(-1))
            yMu_inf, y_mu, _EFE = model.process_jit(model, obs=inputs, lab=targets_flat, adapt_synapses=False)
            yMu_inf.block_until_ready()
            y_pred = yMu_inf.reshape(-1, vocab_size)
            y_true = targets_flat
            batch_nll = measure_CatNLL(y_pred, y_true)
            total_nll += float(batch_nll.sum()) * y_true.shape[0]
            total_tokens += y_true.shape[0]
        ce_loss = total_nll / total_tokens
        return ce_loss, jnp.exp(ce_loss)
    
   

    for i in range(epoch):
        start_time = time.time()
        train_EFE = 0.
        total_batches = 0
        
        print(f"\n iter {i}:")
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = device_put(batch[0][1])
            targets = device_put(batch[1][1])
            targets_flat = compute_one_hot(targets.reshape(-1))
            yMu_inf, _, _EFE = model.process_jit(model, obs=inputs, lab=targets_flat, adapt_synapses=True)
            train_EFE += _EFE
            total_batches += 1
            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)
                y_true = targets_flat
                batch_ce_loss, batch_ppl = compute_metrics(y_pred, y_true)
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx}: EFE = {_EFE:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}, Time = {elapsed:.2f}s")
        
        avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0

        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
        print(f"Iter {i} Summary: CE = {dev_ce:.4f}, PPL = {dev_ppl:.4f}, Avg EFE = {avg_train_EFE:.4f}")
        if  i == (epoch-1):
            model.save_to_disk(params_only=False) # save final state of model to disk
    total_time = time.time() - start_time
    print(f"\nTraining finished.")
    print(f"Total training time: {total_time:.0f} seconds")
   
if __name__ == "__main__":
    main()
