import numpy as np
from .includemix import get_params


def simplemix(x1, f1, mix=None):
    """
    Simple mixing: x_new = x1 + mix * f1
    Returns (x_new, m) where m=1 (one iterate stored), mirroring simplemix.m.
    """
    if mix is None:
        mix = get_params()["mix"]
    x_new = x1 + mix * f1
    return x_new, 1
