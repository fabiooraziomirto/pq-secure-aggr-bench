import numpy as np
import pytest

from pqfl.train.updates import apply_average, clamp_update, local_seed


def test_clamp_reports_the_fraction_and_bounds_every_element():
    update = np.array([0.5, -2.0, 0.1, 3.0])
    clamped, fraction = clamp_update(update, 1.0)
    assert np.array_equal(clamped, [0.5, -1.0, 0.1, 1.0]) and fraction == 0.5
    assert np.array_equal(clamp_update(update, None)[0], update) and clamp_update(update, None)[1] == 0.0
    with pytest.raises(ValueError):
        clamp_update(update, 0.0)


def test_average_is_fedavg_with_equal_weights():
    g, total = np.array([1.0, 2.0]), np.array([0.3, -0.6])
    assert np.allclose(apply_average(g, total, 3), [1.1, 1.8])
    for bad in ((g, total[:1], 3), (g, total, 0)):
        with pytest.raises(ValueError):
            apply_average(*bad)


def test_local_seeds_differ_per_round_and_site_and_repeat():
    seeds = {local_seed(7, r, s) for r in range(10) for s in range(3)}
    assert len(seeds) == 30 and local_seed(7, 2, 1) == local_seed(7, 2, 1) and local_seed(8, 2, 1) != local_seed(7, 2, 1)
    assert all(0 <= x < 2 ** 31 for x in seeds)



def test_local_seed_rejects_site_indices_where_it_would_collide():
    assert local_seed(0, 1, 0) != local_seed(0, 0, 1008)
    with pytest.raises(ValueError):
        local_seed(0, 0, 1009)
    with pytest.raises(ValueError):
        local_seed(0, -1, 0)
