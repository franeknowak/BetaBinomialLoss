import random
import numpy as np
import torch
import os
def set_deterministic_behaviour(seed):
    # Environment Standardisation
    random.seed(seed)                      # Set random seed
    np.random.seed(seed)                   # Set NumPy seed
    torch.manual_seed(seed)                # Set PyTorch seed
    torch.cuda.manual_seed(seed)           # Set CUDA seed
    torch.backends.cudnn.benchmark = False # Disable dynamic tuning
    torch.use_deterministic_algorithms(True) # Force deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA workspace config