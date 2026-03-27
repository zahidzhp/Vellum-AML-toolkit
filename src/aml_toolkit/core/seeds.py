"""Centralized seed management for reproducibility."""

import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seeds for all random number generators used by the toolkit.

    Args:
        seed: Integer seed value. Defaults to DEFAULT_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
