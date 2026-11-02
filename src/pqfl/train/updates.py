"""Pure-numpy pieces of a federated round: clamping, averaging, seeds (testable without torch)."""
import numpy as np


def clamp_update(update: np.ndarray, bound: float | None):
    """Clamp each element to +-bound (post-processing: safe for DP); returns (update, fraction clamped)."""
    if bound is None:
        return update, 0.0
    if not bound > 0:
        raise ValueError("bound must be positive")
    return np.clip(update, -bound, bound), float(np.mean(np.abs(update) > bound))


def apply_average(global_vector: np.ndarray, summed_update: np.ndarray, sites: int) -> np.ndarray:
    """FedAvg with equal weights (the sites hold similar amounts of data): global + sum / n."""
    if sites < 1 or global_vector.shape != summed_update.shape:
        raise ValueError("Invalid sites or mismatched shapes")
    return global_vector + summed_update / sites


def local_seed(seed: int, round_idx: int, site_index: int) -> int:
    """Seed of one site in one round: distinct per round and site, reproducible across resumes.
    The formula is kept as is so that earlier runs stay reproducible; it is injective only for
    site_index < 1009, which is enforced."""
    if not 0 <= site_index < 1009 or round_idx < 0:
        raise ValueError("local_seed supports 0 <= site_index < 1009 and nonnegative rounds")
    return (seed * 1_000_003 + round_idx * 1009 + site_index) % (2 ** 31)
