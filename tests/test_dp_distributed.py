"""Distributed Skellam accounting: generic Poisson subsampling, Skellam RDP, shares tolerating t colluders."""
import math

import numpy as np
import pytest

from pqfl.dp import (calibrate_sigma, calibrate_skellam_sigma, noise_shares, rdp_sampled_gaussian,
                     rdp_skellam, rdp_subsampled_generic, skellam_epsilon, skellam_sensitivity)

PILOT = dict(q=32 / 2210, steps=276, delta=1e-5)
GEOMETRY = dict(clip=4.0, scale=2.0 ** 16, dim=8_409_600)


def test_gaussian_accountant_matches_the_published_mnist_reference():
    """TF Privacy / Abadi et al. MNIST example: 3.01 with the standard RDP-to-DP conversion."""
    q, steps = 256 / 60000, int(60 * 60000 / 256)
    standard = min(steps * rdp_sampled_gaussian(q, 1.1, a) - math.log(1e-5) / (a - 1) for a in range(2, 257))
    assert standard == pytest.approx(3.01, abs=0.01)


def test_generic_subsampling_bound_is_never_below_the_exact_gaussian_one():
    for q in (0.001, 0.0145, 0.2):
        for sigma in (0.6, 1.0, 3.0):
            for alpha in (2, 5, 32, 128):
                exact = rdp_sampled_gaussian(q, sigma, alpha)
                generic = rdp_subsampled_generic(q, lambda l: l / (2 * sigma ** 2), alpha)
                assert generic >= exact - 1e-12
    assert rdp_subsampled_generic(0.0, lambda l: l / 2, 8) == pytest.approx(0.0, abs=1e-12)


def test_skellam_rdp_is_gaussian_plus_a_vanishing_term():
    l2, l1 = skellam_sensitivity(**GEOMETRY)
    for alpha in (2, 16, 256):
        for sigma in (0.5, 1.0):
            mu = (GEOMETRY["scale"] * GEOMETRY["clip"] * sigma) ** 2
            gaussian = alpha * l2 ** 2 / (2 * mu)
            value = rdp_skellam(alpha, l2, l1, mu)
            assert gaussian <= value <= gaussian * (1 + 1e-3)
    assert rdp_skellam(2, 1, 1, 0.5) > 2 / (2 * 0.5)           # small mu: the extra term is visible


def test_sensitivity_bounds_every_rounded_clipped_example():
    rng = np.random.default_rng(0)
    clip, scale, dim = 4.0, 2.0 ** 6, 4096
    l2, l1 = skellam_sensitivity(clip, scale, dim)
    for _ in range(50):
        g = rng.normal(size=dim)
        g *= clip / np.linalg.norm(g) * rng.uniform(0.1, 1.0)
        rounded = np.rint(scale * g)
        assert np.linalg.norm(rounded) <= l2 and np.abs(rounded).sum() <= l1


def test_calibration_hits_the_target_and_costs_more_noise_than_the_exact_gaussian():
    for target in (2.0, 8.0):
        sigma = calibrate_skellam_sigma(target, PILOT["delta"], PILOT["q"], PILOT["steps"], **GEOMETRY)
        achieved = skellam_epsilon(PILOT["q"], sigma, PILOT["steps"], PILOT["delta"], **GEOMETRY)
        assert target - 0.05 < achieved <= target
        assert sigma >= calibrate_sigma(target, PILOT["delta"], PILOT["q"], PILOT["steps"])


def test_pilot_regression_values():
    """Values reported to the authors on 2 Nov 2026 (sigma of the eps = 2 Gaussian pilot)."""
    eps = lambda s: skellam_epsilon(PILOT["q"], s, PILOT["steps"], PILOT["delta"], **GEOMETRY)
    assert eps(0.974) == pytest.approx(2.41, abs=0.02)
    assert eps(0.974 / math.sqrt(3)) == pytest.approx(11.48, abs=0.05)


def test_noise_shares_tolerate_t_colluders():
    share, aggregate = noise_shares(1.0, sites=3, tolerated=0)
    assert share == pytest.approx(1 / math.sqrt(3)) and aggregate == pytest.approx(1.0)
    share, aggregate = noise_shares(1.0, sites=3, tolerated=2)          # local DP: each site alone
    assert share == pytest.approx(1.0) and aggregate == pytest.approx(math.sqrt(3))
    share, aggregate = noise_shares(2.0, sites=10, tolerated=5)
    assert share * math.sqrt(10 - 5) == pytest.approx(2.0)              # any n - t honest sites give sigma
    for bad in (dict(sites=3, tolerated=3), dict(sites=3, tolerated=-1), dict(sites=1, tolerated=0),
                dict(sites=3, tolerated=1.5)):
        with pytest.raises(ValueError):
            noise_shares(1.0, **bad)
    with pytest.raises(ValueError):
        noise_shares(0.0, sites=3, tolerated=0)


def test_validation():
    for call in (lambda: rdp_subsampled_generic(1.5, lambda l: 1.0, 2), lambda: rdp_subsampled_generic(0.1, lambda l: 1.0, 1),
                 lambda: rdp_skellam(2, 1, 1, 0), lambda: rdp_skellam(1, 1, 1, 1),
                 lambda: skellam_sensitivity(0, 1, 1), lambda: skellam_sensitivity(1, 1, 0),
                 lambda: skellam_epsilon(0.1, 0, 10, 1e-5, **GEOMETRY)):
        with pytest.raises(ValueError):
            call()
