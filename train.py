import time
import psutil
import os
from jax import numpy as jnp, random
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model

def log_mem(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2) 
    print(f"--- [STANDARD LOG] {label} | Resident Memory: {mem:.2f} MB ---")

def main():
    log_mem("INITIAL STARTUP")
    
    seq_len, batch_size, vocab_size = config.seq_len, config.batch_size, config.vocab_size
    dkey = random.PRNGKey(1234)
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()
    
    model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=config.n_embed,
        vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
        T=config.n_iter, dt=1., tau_m=config.tau_m, act_fx=config.act_fx,
        eta=config.eta, dropout_rate=config.dropout_rate, exp_dir="exp",
        loadDir=None, pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, wub=config.wub, wlb=config.wlb,
        model_name="ngc_transformer")

    start_time = time.time()

    for i in range(config.epoch):
        print(f"\n>> Starting Epoch {i}")
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            inputs = batch[0][1] # Standard float32
            targets = batch[1][1]
            targets_onehot = jnp.eye(vocab_size)[targets] 
            targets_flat = targets_onehot.reshape(-1, vocab_size)

            yMu_inf, _, _ = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            yMu_inf.block_until_ready() 
            
            step_duration = time.time() - step_start

            if batch_idx % 20 == 0:
                # Logging memory normally (no cleanup performed)
                log_mem(f"Epoch {i} Batch {batch_idx}")
                print(f"Step Time: {step_duration:.4f}s")

    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()