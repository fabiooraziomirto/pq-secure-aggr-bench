"""RDP accountant for DP-SGD (Poisson-subsampled Gaussian) and noise calibration.

Example-level DP in each site's local training; sites hold disjoint data, so a record's
guarantee is that of its own site's mechanism. Integer Renyi orders only; the conversion to
(epsilon, delta) is the improved one of Balle et al. (2020). Independent of any DP library.
TODO: cross-check against Opacus' accountant once it is installed in the training environment.
"""
import math

ORDERS = tuple(range(2, 257))


def _logsumexp(values):
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """Renyi DP of one step at integer order alpha for sampling rate q and noise multiplier sigma."""
    if not 0 <= q <= 1 or sigma <= 0 or type(alpha) is not int or alpha < 2:
        raise ValueError("Invalid accountant parameters")
    if q == 0:
        return 0.0
    if q == 1:
        return alpha / (2 * sigma ** 2)
    terms = []
    for k in range(alpha + 1):
        log_binomial = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        terms.append(log_binomial + (alpha - k) * math.log1p(-q) + k * math.log(q)
                     + (k * k - k) / (2 * sigma ** 2))
    return _logsumexp(terms) / (alpha - 1)


def epsilon(q: float, sigma: float, steps: int, delta: float, orders=ORDERS) -> float:
    """(epsilon, delta) after `steps` identical steps: compose in RDP, then convert."""
    if steps < 0 or not 0 < delta < 1:
        raise ValueError("Invalid steps or delta")
    best = math.inf
    for alpha in orders:
        rdp = steps * rdp_sampled_gaussian(q, sigma, alpha)
        value = rdp + math.log1p(-1 / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        best = min(best, value)
    return max(best, 0.0)


def calibrate_sigma(target_epsilon: float, delta: float, q: float, steps: int,
                    tolerance: float = 1e-3) -> float:
    """Smallest noise multiplier (to `tolerance`) whose epsilon is at most the target."""
    if target_epsilon <= 0:
        raise ValueError("Target epsilon must be positive")
    low, high = 0.2, 200.0
    if epsilon(q, high, steps, delta) > target_epsilon:
        raise ValueError("Target epsilon unreachable with sigma <= 200")
    while high - low > tolerance:
        middle = (low + high) / 2
        if epsilon(q, middle, steps, delta) > target_epsilon:
            low = middle
        else:
            high = middle
    return high


def group_epsilon_bound(epsilon_single: float, delta_single: float, copies: int) -> tuple[float, float]:
    """Group privacy for `copies` identical records of an (epsilon, delta)-DP mechanism:
    (k * epsilon, k * exp((k - 1) * epsilon) * delta), the standard (loose) bound. The delta term
    grows quickly, so for large k * epsilon the bound can be vacuous (delta >= 1 is returned as is)."""
    if type(copies) is not int or copies < 1:
        raise ValueError("copies must be a positive integer")
    if epsilon_single < 0 or not 0 <= delta_single < 1:
        raise ValueError("Invalid epsilon or delta")
    return epsilon_single * copies, copies * math.exp((copies - 1) * epsilon_single) * delta_single
