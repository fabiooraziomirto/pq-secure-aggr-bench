"""Percentile bootstrap over seeds, as the protocol prescribes (no significance tests)."""
import numpy as np


def bootstrap_ci(values, *, level: float = 0.95, resamples: int = 10000, seed: int = 0):
    """(mean, low, high): percentile bootstrap of the mean over seeds. With few seeds the
    interval is coarse: it can only take values from resampled means of those seeds."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not 0 < level < 1:
        raise ValueError("Need at least two values and a level in (0, 1)")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(values.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def paired_difference_ci(a, b, **kwargs):
    """CI of the mean per-seed difference a - b; the seeds must be paired."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired samples must have the same length")
    return bootstrap_ci(a - b, **kwargs)


def intervals_overlap(first, second) -> bool:
    """True if two (mean, low, high) intervals overlap; the protocol's H1/H2 criterion is non-overlap."""
    return not (first[2] < second[1] or second[2] < first[1])
