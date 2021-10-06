"""Set up Environment."""
import sys
import os
import random
import torch
import numpy as np
import src.utils.pathutils as pathutils

_ENVIRONMENT_SETUP_DONE = False

def setup_environment():
    """
    Called at initial application booting, is used for environment setup and preparation
    """
    global _ENVIRONMENT_SETUP_DONE
    if _ENVIRONMENT_SETUP_DONE:
        return

    sys.path.insert(0, str(pathutils.get_models_path()))

    _ENVIRONMENT_SETUP_DONE = True


def setup_random_environment(cfg):
    """
    Utility for random environment initialization
    Args
        cfg (cfgNode): Model configurations
    """
    if cfg.SET_RNG_SEED:
        os.environ['PYTHONHASHSEED']=str(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
