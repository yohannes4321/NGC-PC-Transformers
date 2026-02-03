import time
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

# ------------------------------------------------------------------
# OPTIMIZATION: JIT-compiled training step
# ------------------------------------------------------------------
@partial(jax.jit, static_argnames=['model', 'adapt_synapses'])
def train_step(model, inputs, targets_flat, adapt_synapses=True):
    """
    JIT-compiled step using one_hot. 
    Returns the inference mean (yMu_inf) and the Expected Free Energy (_EFE).
    """
    yMu_inf, _, _EFE = model.process(
        obs=inputs,
        lab=targets_flat,
        adapt_synapses=adapt_synapses
    )
    # We return EFE so we can print it during training
    return yMu_inf, _EFE

def main():
    total_start_time = time.time()

    # ----------------------------
    # Config & Initialization
    # ----------------------------
    seq_len = config.seq_len
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    num_iter = config.num_iter
    dkey = random.PRNGKey(1234)

    print("\n✅ RUNNING OPTIMIZED JIT + ONE-HOT VERSION")
    print(f"Vocab size: {vocab_size} | Epochs: {num_iter}")
    print("-" * 50)

    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()

    model = NGCTransformer(
        dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1.0, tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.eta, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer",
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    for i in range(num_iter):
        print(f"\nIteration (Epoch) {i}")
        
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            # Data parsing
            inputs = batch[0][1]
            targets = batch[1][1]

            # ✅ EFFICIENT: No jnp.eye() matrix created. 
            # jax.nn.one_hot creates the representation lazily/efficiently.
            targets_onehot = jax.nn.one_hot(targets, vocab_size)
            targets_flat = targets_onehot.reshape(-1, vocab_size)
            
            # Tracking memory footprint of the actual batch
            one_hot_mb = targets_onehot.nbytes / (1024**2)

            # --- JIT EXECUTION ---
            yMu_inf, _EFE = train_step(model, inputs, targets_flat)

            # Block to ensure accurate timing
            yMu_inf.block_until_ready()
            step_duration = time.time() - step_start

            # Logging every 10 batches
            if batch_idx % 10 == 0:
                y_pred = yMu_inf.reshape(-1, vocab_size)
                
                # Calculate Cross Entropy (CE)
                batch_nll = measure_CatNLL(y_pred, targets_flat)
                batch_ce = batch_nll.mean()
                
                # Calculate Perplexity (PPL)
                batch_ppl = jnp.exp(batch_ce)

                print(
                    f"Batch {batch_idx:03d} | "
                    f"Time: {step_duration:.4f}s | "
                    f"EFE: {_EFE:.4f} | "
                    f"CE: {batch_ce:.4f} | "
                    f"PPL: {batch_ppl:.2f}"
                )

        # Eval after each epoch
        dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
        print(f"--- Iter {i} Summary: Dev CE={dev_ce:.4f}, Dev PPL={dev_ppl:.4f} ---")

    # FINAL REPORTING
    total_duration = time.time() - total_start_time
    print("\n" + "="*50)
    print("✅ OPTIMIZED TRAINING FINISHED")
    print(f"Total Program Runtime: {total_duration:.2f} seconds")
    print(f"Efficiency: Used jax.nn.one_hot ({one_hot_mb:.2f} MB) instead of jnp.eye.")
    print("="*50)

if __name__ == "__main__":
    main()
# import jax
# from jax import numpy as jnp, random
# from model import NGCTransformer
# from ngclearn.utils.metric_utils import measure_CatNLL
# from data_preprocess.data_loader import DataLoader
# from config import Config as config
# from eval import eval_model
# import time

# def main():
#     seq_len, batch_size, n_embed, vocab_size, n_layers, n_heads, n_iter, optim_type = config.seq_len, config.batch_size, config.n_embed, config.vocab_size, config.n_layers, config.n_heads, config.n_iter, config.optim_type
#     pos_learnable= config.pos_learnable
#     epoch= config.num_iter
#     wub= config.wub 
#     wlb= config.wlb
#     eta = config.eta
#     T = config.n_iter
#     tau_m= config.tau_m
#     act_fx= config.act_fx
#     dropout_rate= config.dropout_rate
#     dkey = random.PRNGKey(1234)
    
#     data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
#     train_loader, valid_loader, test_loader = data_loader.load_and_prepare_data()
    
#     model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=n_embed, vocab_size=vocab_size, n_layers=n_layers, n_heads=config.n_heads,
#                           T=T, dt=1., tau_m=tau_m , act_fx=act_fx, eta=eta, dropout_rate= dropout_rate, exp_dir="exp",
#                   loadDir= None, pos_learnable= pos_learnable, optim_type=optim_type, wub = wub, wlb= wlb, model_name="ngc_transformer" )

#     def train_model(data_loader):
#         total_nll, total_tokens = 0., 0
        
#         for batch in data_loader:
#             inputs = batch[0][1]
#             targets = batch[1][1]
            
#             targets_onehot = jax.nn.one_hot(targets, vocab_size)# (B, S, V)
#             targets_flat = targets_onehot.reshape(-1, vocab_size)  # (B*S, V)

#             yMu_inf, y_mu, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=False)
            
#             y_pred = yMu_inf.reshape(-1, vocab_size)
#             y_true = targets_flat
            
#             total_nll += measure_CatNLL(y_pred, y_true) * y_true.shape[0]
#             total_tokens += y_true.shape[0]
        
#         ce_loss = total_nll / total_tokens
#         return ce_loss, jnp.exp(ce_loss)

#     for i in range(epoch):
#         train_EFE = 0.
#         total_batches = 0
        
#         print(f"\n iter {i}:")
        
#         for batch_idx, batch in enumerate(train_loader):
#             inputs = batch[0][1]
#             targets = batch[1][1]
            
#             #Convert targets to one-hot and flatten
#             targets_onehot = jax.nn.one_hot(targets, vocab_size) # (B, S, V)
#             targets_flat = targets_onehot.reshape(-1, vocab_size)  # (B*S, V)

            
#             yMu_inf, _, _EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
#             train_EFE += _EFE
#             total_batches += 1

#             if batch_idx % 10 == 0:
#                 y_pred = yMu_inf.reshape(-1, vocab_size)
#                 y_true= jax.nn.one_hot(targets, vocab_size)
                
#                 batch_nll = measure_CatNLL(y_pred, y_true)
#                 batch_ce_loss = batch_nll.mean()  
#                 batch_ppl = jnp.exp(batch_ce_loss)
                
#                 print(f"  Batch {batch_idx}: EFE = {_EFE:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f}")
        
#         avg_train_EFE = train_EFE / total_batches if total_batches > 0 else 0
        
#         dev_ce, dev_ppl = eval_model(model, valid_loader, vocab_size)
#         print(f"Iter {i} Summary: CE = {dev_ce:.4f}, PPL = {dev_ppl:.4f}, Avg EFE = {avg_train_EFE:.4f}")
#         if  i == (epoch-1):
#           model.save_to_disk(params_only=False) # save final state of model to disk
#     total_time = time.time() - total_start_time
#     print(f"\nTraining finished.")
#     print(f"Total training time: {total_time:.0f} seconds")
   
# if __name__ == "__main__":
#     main()