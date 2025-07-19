import os, random, numpy as np

SEED = 2  # default global seed (will be updated by set_global_seed)

def set_global_seed(seed: int | None = None):
    """
    Set all RNG seeds and synchronously update the module-level SEED.

    Parameters
    ----------
    seed : int or None
        New seed value; if None, reuse the current env_utils.SEED.
    """
    global SEED
    if seed is None:
        seed = SEED          # keep existing default
    else:
        SEED = seed          # update the global constant

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Uncomment if using PyTorch
    # try:
    #     import torch
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # except ImportError:
    #     pass

def get_device():
    # Uncomment if using PyTorch
    # try:
    #     import torch
    #     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # except ImportError:
    #     return "cpu"
    
    return "cpu"

