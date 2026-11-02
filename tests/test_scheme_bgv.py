"""Scheme B-BGV: exact integer sums under threshold BGV with multiparty noise flooding."""
import numpy as np
import pytest

from conftest import InsecureTestAuth
from pqfl.aggregators.scheme_b_bgv import SchemeBGV
from pqfl.he.bgv import ThresholdBGV, batching_prime
from pqfl.net.replay import ledger_stages

pytestmark = pytest.mark.requires_openfhe
SITES = ["A", "B", "C"]
UPDATES = {"A": np.array([0.1, 0.25, -0.3]), "B": np.array([0.4, -0.5, 0.6]),
           "C": np.array([-0.7, 0.8, 0.9])}
SCALE = 20


def make(sites=SITES, **kwargs):
    agg = SchemeBGV(clip_bound=1., authenticator=InsecureTestAuth(), **kwargs)
    agg.setup(sites)
    return agg


def test_sum_matches_quantized_sum_exactly():
    out = make().aggregate_round(0, UPDATES)
    quantized = sum(np.rint(u * 2 ** SCALE) for u in UPDATES.values()) / 2 ** SCALE
    np.testing.assert_array_equal(out, quantized)  # exact: only quantization error remains
    np.testing.assert_allclose(out, sum(UPDATES.values()), rtol=0, atol=3 * 2 ** -(SCALE + 1))


def test_vector_larger_than_slots_and_ten_parties():
    rng = np.random.default_rng(2)
    sites = [f"S{i:02d}" for i in range(10)]
    updates = {s: rng.uniform(-1, 1, 40000) for s in sites}  # 3 chunks of 16384 slots
    out = make(sites).aggregate_round(0, updates)
    expected = sum(np.rint(u * 2 ** SCALE) for u in updates.values()) / 2 ** SCALE
    np.testing.assert_array_equal(out, expected)


def test_noise_flooding_is_active_in_the_context():
    # Flooding widens the ciphertext modulus; in CKKS the size did not change.
    sizes = {}
    for flooding in (False, True):
        he = ThresholdBGV(parties=3, flooding=flooding)
        key = he.keygen_first()
        sizes[flooding] = len(he.encrypt(he.serialize_public(key), [1, 2, 3])[0])
    assert sizes[True] > sizes[False] * 1.4


def test_batching_prime_and_overflow_guard():
    p = batching_prime(1 << 14, 31)
    assert p % (2 * (1 << 14)) == 1 and p.bit_length() == 32
    with pytest.raises(ValueError, match="exact signed sum"):
        SchemeBGV(clip_bound=1., scale_bits=29, authenticator=InsecureTestAuth()).setup(SITES)


def test_stages_and_rekey_and_clip():
    agg = make()
    agg.aggregate_round(0, UPDATES)
    assert len(ledger_stages(agg.stats())) == 4
    first = agg._joint["A"]
    agg.rekey()
    assert agg.epoch == 1 and agg._joint["A"] != first
    agg.aggregate_round(1, UPDATES)
    before = agg.stats()
    with pytest.raises(ValueError):
        agg.aggregate_round(2, dict(UPDATES, A=np.array([2., 0., 0.])))
    assert agg.stats() == before and 2 not in agg.used_rounds
