import random
import numpy as np
import torch
import os
import yaml
from yacs.config import CfgNode as CN

def set_deterministic_behaviour(seed):
    # Environment Standardisation
    random.seed(seed)                      # Set random seed
    np.random.seed(seed)                   # Set NumPy seed
    torch.manual_seed(seed)                # Set PyTorch seed
    torch.cuda.manual_seed(seed)           # Set CUDA seed
    torch.backends.cudnn.benchmark = False # Disable dynamic tuning
    torch.use_deterministic_algorithms(True) # Force deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA workspace config

def get_config(config_path):
    """
    Runs functions related to reading, processing, and informing user about the config setup.
    """
    config_dict = read_config(config_path)
    #config = config_to_yacs(config_dict)
    return config_dict

def read_config(config_file):
    """
    Read Yaml file into dict
    """
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

#def config_to_yacs(dict_config):
#    """
#    Convert the dict to a yacs.
#    """
#    if not isinstance(dict_config, dict):
#        return dict_config  # Return non-dict values as is
#    cfg = CN()
#    for key, value in dict_config.items():
#        cfg[key] = config_to_yacs(value)  # Recursively convert nested dictionaries
#    return cfg
#