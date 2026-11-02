"""Scheme C: additive secret sharing with sealed pairwise delivery."""
import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from conftest import InsecureTestAuth
from pqfl.aggregators.scheme_c import SchemeC
from pqfl.logging.metrics import ByteCounter
from pqfl.masking.sharing import open_share, seal_share, split_shares

SITES = ["A", "B", "C"]
UPDATES = {"A": np.array([0.1, 0.25, -0.3]), "B": np.array([0.4, -0.5, 0.6]),
           "C": np.array([-0.7, 0.8, 0.9])}


def make(fake_kem_factory, sites=SITES):
    agg = SchemeC(fake_kem_factory(7), clip_bound=1., scale_bits=12, modulus_bits=32,
                  authenticator=InsecureTestAuth())
    agg.setup(sites)
    return agg


def test_shares_sum_to_input_and_are_not_the_input():
    quantized = np.arange(1000, dtype=np.int64)
    shares = split_shares(quantized, 3, 32)
    np.testing.assert_array_equal(sum(shares) % 2 ** 32, quantized)
    assert not any(np.array_equal(share, quantized) for share in shares)
    assert all(share.min() >= 0 and share.max() < 2 ** 32 for share in shares)


def test_sealed_share_roundtrip_and_tamper_or_context_rejected():
    seed = bytes(range(32))
    sealed = seal_share(seed, 0, 1, "A", "B", b"payload")
    assert open_share(seed, 0, 1, "A", "B", sealed) == b"payload"
    tampered = bytes([sealed[0] ^ 1]) + sealed[1:]
    for args, message in [((0, 1, "A", "B"), tampered), ((1, 1, "A", "B"), sealed),
                          ((0, 2, "A", "B"), sealed), ((0, 1, "B", "A"), sealed)]:
        with pytest.raises(InvalidTag):
            open_share(seed, *args, message)


def test_reducer_never_receives_single_site_value(fake_kem_factory, monkeypatch):
    seen = []
    original = ByteCounter.send

    def send(counter, sender, receiver, message):
        seen.append((sender, receiver, message))
        return original(counter, sender, receiver, message)

    monkeypatch.setattr(ByteCounter, "send", send)
    agg = make(fake_kem_factory)
    seen.clear()
    np.testing.assert_allclose(agg.aggregate_round(0, UPDATES), sum(UPDATES.values()),
                               rtol=0, atol=3 * 0.5 * 2 ** -12 + 1e-12)
    from pqfl.masking.core import quantize
    from pqfl.masking.wire import serialize_masked
    plain = {site: serialize_masked(quantize(u, 12, 32), 32) for site, u in UPDATES.items()}
    to_reducer = [m for s, r, m in seen if r == "reducer"]
    assert len(to_reducer) == 3 and not set(to_reducer) & set(plain.values())
    site_to_site = [m for s, r, m in seen if s != "reducer" and r != "reducer"]
    assert len(site_to_site) == 6 and not set(site_to_site) & set(plain.values())


def test_round_traffic_is_quadratic_in_sites_unlike_a(fake_kem_factory):
    agg = make(fake_kem_factory)
    agg.aggregate_round(0, UPDATES)
    pairs = [t for t in agg.stats().transfers if t.receiver != "reducer" and t.sender != "reducer"]
    assert len(pairs) == len(SITES) * (len(SITES) - 1)
    assert all(t.size == 3 * 4 + 16 for t in pairs)  # 3 words * 4 bytes + AEAD tag


def test_corrupted_share_in_transit_aborts(fake_kem_factory, monkeypatch):
    agg = make(fake_kem_factory)
    original = ByteCounter.send

    def send(counter, sender, receiver, message):
        if sender == "A" and receiver == "B":
            message = bytes([message[0] ^ 1]) + message[1:]
        return original(counter, sender, receiver, message)

    monkeypatch.setattr(ByteCounter, "send", send)
    with pytest.raises(InvalidTag):
        agg.aggregate_round(0, UPDATES)


def test_rekey_changes_epoch_and_sum_stays_correct(fake_kem_factory):
    agg = make(fake_kem_factory)
    agg.aggregate_round(0, UPDATES)
    agg.rekey()
    assert agg.epoch == 1
    np.testing.assert_allclose(agg.aggregate_round(1, UPDATES), sum(UPDATES.values()),
                               rtol=0, atol=3 * 0.5 * 2 ** -12 + 1e-12)


def test_clip_bound_enforced(fake_kem_factory):
    agg = make(fake_kem_factory)
    with pytest.raises(ValueError, match="clip_bound"):
        agg.aggregate_round(0, dict(UPDATES, A=np.array([2., 0., 0.])))


@pytest.mark.parametrize("bad", [np.array([2., 0., 0.]), np.array([np.nan, 0., 0.])])
def test_invalid_update_is_rejected_before_the_round_is_consumed(fake_kem_factory, bad):
    agg = make(fake_kem_factory)
    with pytest.raises(ValueError, match="clip_bound or is not finite"):
        agg.aggregate_round(0, dict(UPDATES, B=bad))
    np.testing.assert_allclose(agg.aggregate_round(0, UPDATES), sum(UPDATES.values()), atol=1e-3)
