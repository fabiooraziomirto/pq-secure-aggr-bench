import math

import pytest

from pqfl.dp import calibrate_sigma, epsilon, group_epsilon_bound, rdp_sampled_gaussian


def test_full_batch_equals_the_gaussian_mechanism_exactly():
    for alpha in (2, 8, 64):
        assert rdp_sampled_gaussian(1.0, 3.0, alpha) == pytest.approx(alpha / 18.0)
    assert rdp_sampled_gaussian(0.0, 1.0, 5) == 0.0


def test_small_rate_matches_the_second_order_expansion():
    q, sigma, alpha = 0.01, 10.0, 8
    approx = alpha * q * q * (math.exp(1 / sigma ** 2) - 1) / 2
    assert rdp_sampled_gaussian(q, sigma, alpha) == pytest.approx(approx, rel=0.05)


def test_epsilon_is_monotone_in_noise_rate_and_steps():
    base = epsilon(0.02, 1.0, 500, 1e-5)
    assert epsilon(0.02, 2.0, 500, 1e-5) < base            # more noise
    assert epsilon(0.01, 1.0, 500, 1e-5) < base            # less sampling
    assert epsilon(0.02, 1.0, 1000, 1e-5) > base           # more steps
    assert epsilon(0.02, 1.0, 500, 1e-3) < base            # looser delta
    # With no steps the true epsilon is 0; the conversion at orders <= 256 stays slightly above
    # (conservative), which is why the bound is loose here.
    assert epsilon(0.02, 1.0, 0, 1e-5) < 0.05


def test_calibration_hits_the_target_from_below():
    for target in (2.0, 8.0):
        sigma = calibrate_sigma(target, 1e-5, 0.0145, 700)
        achieved = epsilon(0.0145, sigma, 700, 1e-5)
        assert target - 0.05 < achieved <= target
    assert calibrate_sigma(2.0, 1e-5, 0.0145, 700) > calibrate_sigma(8.0, 1e-5, 0.0145, 700)
    with pytest.raises(ValueError):
        calibrate_sigma(1e-6, 1e-5, 0.5, 100000)


def test_validation_and_group_privacy():
    for call in (lambda: rdp_sampled_gaussian(1.5, 1, 2), lambda: rdp_sampled_gaussian(0.1, 0, 2),
                 lambda: rdp_sampled_gaussian(0.1, 1, 1), lambda: epsilon(0.1, 1, 10, 2.0),
                 lambda: calibrate_sigma(0, 1e-5, 0.1, 10)):
        with pytest.raises(ValueError):
            call()
    assert group_epsilon_bound(8.0, 1e-5, 1) == (8.0, 1e-5)
    eps, delta = group_epsilon_bound(0.5, 1e-6, 4)
    assert eps == 2.0 and delta == pytest.approx(4 * math.exp(1.5) * 1e-6)
    for bad in ((1.0, 1e-5, 0), (1.0, 1e-5, 2.5), (-1.0, 1e-5, 2), (1.0, 1.0, 2)):
        with pytest.raises(ValueError):
            group_epsilon_bound(*bad)


def test_accountant_matches_numerical_integration():
    """Independent check of the binomial expansion: E_z[(1-q+q*exp((2z-1)/(2s^2)))^a], z ~ N(0, s^2)."""
    np = pytest.importorskip("numpy")
    q = 32 / 2210
    for sigma, alpha in ((0.6, 3), (1.0, 8), (2.5, 32)):
        z = np.linspace(-20 * sigma - 5, alpha + 20 * sigma + 5, 400_001)
        log_pdf = -z * z / (2 * sigma ** 2) - 0.5 * math.log(2 * math.pi * sigma ** 2)
        inner = np.logaddexp(math.log1p(-q), math.log(q) + (2 * z - 1) / (2 * sigma ** 2))
        values = log_pdf + alpha * inner
        top = values.max()
        integral = top + math.log(np.trapezoid(np.exp(values - top), z))
        assert rdp_sampled_gaussian(q, sigma, alpha) == pytest.approx(integral / (alpha - 1), rel=1e-6)
