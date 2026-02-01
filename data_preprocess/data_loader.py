# Force JAX to only use one GPU for now to avoid initialization conflicts
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# Stop JAX from taking 90% of VRAM immediately
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"