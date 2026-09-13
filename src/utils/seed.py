import os
import random

def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Seed random, numpy, and torch (CPU + CUDA) for reproducibility.
 
    Args:
        seed: Value applied to every RNG.
        deterministic: If True, force reproducible cuDNN algorithms and
            disable the autotuner. Bitwise-repeatable but slower (~5-15%,
            workload dependent); use False for max throughput.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
 
    # numpy and torch imported lazily so this module loads on a CPU box
    # before torch is installed
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
 
    try:
        import torch
    except ImportError:
        import warnings
        warnings.warn("torch not installed; skipping torch seeding.", stacklevel=2)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False