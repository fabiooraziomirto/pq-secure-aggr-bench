"""RDP accountant for DP-SGD (Poisson-subsampled Gaussian) and noise calibration.

Example-level DP in each site's local training; sites hold disjoint data, so a record's
guarantee is that of its own site's mechanism. Integer Renyi orders only; the conversion to
(epsilon, delta) is the improved one of Balle et al. (2020). Independent of any DP library.
Cross-checked on the classic MNIST setting (q = 256/60000, sigma = 1.1, 60 epochs): the standard
conversion gives 3.009, the published TF Privacy value (tests/test_dp_distributed.py).

Distributed DP (DP-FedSGD with secure aggregation): every site adds an integer Skellam share to its
rounded, clipped per-example gradients. Skellam RDP from Agarwal, Kairouz and Liu (NeurIPS 2021,
Cor. 3.6), Poisson subsampling with the general upper bound of Zhu and Wang (ICML 2019, Thm. 5),
which holds for any mechanism and is looser than the exact subsampled Gaussian formula above.
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
    return _to_epsilon(lambda alpha: rdp_sampled_gaussian(q, sigma, alpha), steps, delta, orders)


def _to_epsilon(step_rdp, steps: int, delta: float, orders=ORDERS) -> float:
    if steps < 0 or not 0 < delta < 1:
        raise ValueError("Invalid steps or delta")
    best = math.inf
    for alpha in orders:
        rdp = steps * step_rdp(alpha)
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


def _log_binomial(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def rdp_subsampled_generic(q: float, rdp, alpha: int) -> float:
    """Zhu and Wang (2019), Thm. 5: RDP at integer order alpha of any mechanism with RDP curve `rdp`
    (a function of the integer order) run on a Poisson subsample with rate q."""
    if not 0 <= q <= 1 or type(alpha) is not int or alpha < 2:
        raise ValueError("Invalid accountant parameters")
    if q == 0:
        return 0.0
    if q == 1:
        return rdp(alpha)
    log_rest = math.log1p(-q)
    terms = [(alpha - 1) * log_rest + math.log(alpha * q - q + 1),
             _log_binomial(alpha, 2) + 2 * math.log(q) + (alpha - 2) * log_rest + rdp(2)]
    terms += [math.log(3) + _log_binomial(alpha, l) + (alpha - l) * log_rest + l * math.log(q) + (l - 1) * rdp(l)
              for l in range(3, alpha + 1)]
    return _logsumexp(terms) / (alpha - 1)


def rdp_skellam(alpha: int, l2: float, l1: float, mu: float) -> float:
    """Multidimensional Skellam mechanism with per-coordinate variance mu (Agarwal et al., Cor. 3.6)."""
    if type(alpha) is not int or alpha < 2 or mu <= 0 or l2 < 0 or l1 < 0:
        raise ValueError("Invalid Skellam parameters")
    return alpha * l2 ** 2 / (2 * mu) + min(((2 * alpha - 1) * l2 ** 2 + 6 * l1) / (4 * mu ** 2), 3 * l1 / (2 * mu))


def skellam_sensitivity(clip: float, scale: float, dim: int) -> tuple[float, float]:
    """(L2, L1) sensitivity of one example's gradient, clipped to `clip`, scaled by `scale` and rounded to the
    nearest integer coordinate-wise: rounding moves each coordinate by at most 1/2, so the L2 norm is at most
    scale * clip + sqrt(dim) / 2; for integer vectors L1 <= L2 * min(sqrt(dim), L2)."""
    if not clip > 0 or not scale > 0 or type(dim) is not int or dim < 1:
        raise ValueError("Invalid sensitivity parameters")
    l2 = scale * clip + math.sqrt(dim) / 2
    return l2, l2 * min(math.sqrt(dim), l2)


def skellam_epsilon(q: float, sigma: float, steps: int, delta: float, *, clip: float, scale: float, dim: int,
                    orders=ORDERS) -> float:
    """(epsilon, delta) of `steps` Poisson-subsampled Skellam steps whose total noise standard deviation is
    sigma * clip in gradient units (variance (scale * clip * sigma)^2 on the integer grid)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    l2, l1 = skellam_sensitivity(clip, scale, dim)
    mu = (scale * clip * sigma) ** 2
    curve = lambda order: rdp_skellam(order, l2, l1, mu)
    return _to_epsilon(lambda alpha: rdp_subsampled_generic(q, curve, alpha), steps, delta, orders)


def calibrate_skellam_sigma(target_epsilon: float, delta: float, q: float, steps: int, *, clip: float,
                            scale: float, dim: int, tolerance: float = 1e-3) -> float:
    """Smallest noise multiplier (to `tolerance`) whose Skellam epsilon is at most the target."""
    if target_epsilon <= 0:
        raise ValueError("Target epsilon must be positive")
    eps = lambda sigma: skellam_epsilon(q, sigma, steps, delta, clip=clip, scale=scale, dim=dim)
    low, high = 0.2, 200.0
    if eps(high) > target_epsilon:
        raise ValueError("Target epsilon unreachable with sigma <= 200")
    while high - low > tolerance:
        middle = (low + high) / 2
        if eps(middle) > target_epsilon:
            low = middle
        else:
            high = middle
    return high


def noise_shares(sigma: float, *, sites: int, tolerated: int) -> tuple[float, float]:
    """Noise multipliers (per-site share, whole aggregate) when every site adds variance sigma^2 / (n - t):
    any n - t sites together carry sigma, so the guarantee calibrated on sigma holds against up to t colluding
    sites (or a reducer colluding with them). Once the masks are broken an adversary sees every site's share
    alone: that is n - 1 colluders, and each site keeps only sigma / sqrt(n - t). t = n - 1 is local DP."""
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    if type(sites) is not int or sites < 2 or type(tolerated) is not int or not 0 <= tolerated < sites:
        raise ValueError("Require at least two sites and 0 <= tolerated < sites")
    share = sigma / math.sqrt(sites - tolerated)
    return share, share * math.sqrt(sites)
