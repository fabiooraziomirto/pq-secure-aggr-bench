"""Synthetic functional/regression checks, not benchmarks or security proofs."""
from collections import Counter
from copy import deepcopy
import json
import math

import numpy as np
import pytest

import pqfl.aggregators.scheme_a as scheme_module
from pqfl.aggregators.errors import AggregateOverflowError, KeyConfirmationError
from pqfl.aggregators.scheme_a import SchemeA
from pqfl.kem.hybrid import HybridKEM, _pack, _unpack
from pqfl.logging.metrics import JSONLWriter, Metrics
from pqfl.masking import confirmation
from pqfl.masking.core import pairwise_mask, quantize
from pqfl.masking.wire import deserialize_masked, serialize_masked

SITES = ["A", "B", "C"]


def make(factory, *, writer=None, scale_bits=12, modulus_bits=32, clip_bound=None):
    agg = SchemeA(factory(42), scale_bits=scale_bits, modulus_bits=modulus_bits,
                  clip_bound=clip_bound, writer=writer)
    agg.setup(SITES)
    return agg


def writer_at(path):
    return JSONLWriter(path, run_id="synthetic", config_hash="test-only",
                       scheme="A", dp_level="none", run_seed=42)


def sum_deltas(rows, field, actors):
    return {actor: sum(row[field][actor] for row in rows) for actor in actors}


def test_large_tensor_accuracy(fake_kem_factory):
    rng = np.random.default_rng(42)
    updates = {site: rng.uniform(-1, 1, 100000) for site in SITES}
    scale_bits = 12
    agg = make(fake_kem_factory, scale_bits=scale_bits, clip_bound=1.)
    result = agg.aggregate_round(0, updates)
    tolerance = len(SITES) * 0.5 * 2 ** (-scale_bits) + np.finfo(float).eps
    assert np.max(np.abs(result - sum(updates.values()))) <= tolerance


def test_masked_vectors_hide_coordinates_and_balance_bits(fake_kem_factory):
    agg = make(fake_kem_factory)
    quantized = quantize(np.random.default_rng(42).uniform(-1, 1, 100000), 12, 32)
    sigma = math.sqrt(0.5 * (1 - 0.5) / quantized.size)
    for site in SITES:
        masked = agg._masked_site(site, 0, quantized)
        assert np.count_nonzero(masked == quantized) <= 1
        bit_means = np.array([np.mean((masked >> bit) & 1) for bit in range(32)])
        assert np.all(np.abs(bit_means - 0.5) <= 5 * sigma)


def test_mask_domains_include_pair_and_epoch():
    args = (b"INSECURE test-only seed", 7)
    original = pairwise_mask(*args, "A", "B", (128,), 32, epoch=0)
    other_pair = pairwise_mask(*args, "A", "C", (128,), 32, epoch=0)
    other_epoch = pairwise_mask(*args, "A", "B", (128,), 32, epoch=1)
    assert not np.array_equal(original, other_pair)
    assert not np.array_equal(original, other_epoch)


def test_rekey_preserves_round_uniqueness(fake_kem_factory):
    agg = make(fake_kem_factory)
    updates = {site: np.array([0.25, -0.5]) for site in SITES}
    first = agg.aggregate_round(0, updates)
    old = deepcopy(agg.seeds)
    agg.rekey()
    assert agg.epoch == 1
    assert agg.used_rounds == {0}
    assert all(old[pair] != agg.seeds[pair] for pair in old)
    assert len(agg.kem.key_material) == 2 * (len(SITES) - 1)
    np.testing.assert_array_equal(agg.aggregate_round(1, updates), first)
    with pytest.raises(ValueError, match="never reused"):
        agg.aggregate_round(0, updates)


def faulty_kem(factory):
    class WrongDecapsulation:
        """INSECURE test-only fault injection, never a cryptographic backend."""
        fail = True

        def __init__(self):
            self.fake = factory(42)

        def keygen(self):
            return self.fake.keygen()

        def encaps(self, public):
            return self.fake.encaps(public)

        def decaps(self, secret, ciphertext):
            recovered = self.fake.decaps(secret, ciphertext)
            return bytes([recovered[0] ^ 1]) + recovered[1:] if self.fail else recovered

    return WrongDecapsulation()


def test_wrong_shared_secret_is_detected_by_confirmation(fake_kem_factory):
    kem = faulty_kem(fake_kem_factory)
    agg = SchemeA(kem, scale_bits=12, modulus_bits=32)
    with pytest.raises(KeyConfirmationError, match="^Peer key confirmation failed$"):
        agg.setup(SITES)
    assert not agg.seeds
    with pytest.raises(ValueError, match="successful"):
        agg.aggregate_round(0, {site: np.zeros(1) for site in SITES})


def test_failed_rekey_invalidates_old_keys_and_can_recover(fake_kem_factory):
    kem = faulty_kem(fake_kem_factory)
    kem.fail = False
    agg = SchemeA(kem, scale_bits=12, modulus_bits=32)
    agg.setup(SITES)
    updates = {site: np.zeros(1) for site in SITES}
    agg.aggregate_round(0, updates)
    kem.fail = True
    with pytest.raises(KeyConfirmationError):
        agg.rekey()
    assert agg.epoch == 1
    assert not agg.seeds
    assert agg.used_rounds == {0}
    with pytest.raises(ValueError, match="successful"):
        agg.aggregate_round(1, updates)
    kem.fail = False
    agg.rekey()
    assert agg.epoch == 2
    np.testing.assert_array_equal(agg.aggregate_round(1, updates), np.zeros(1))


def test_both_confirmation_directions_are_verified(fake_kem_factory, monkeypatch):
    original = confirmation.confirmation_tag
    seen_right = 0

    def tamper_right_response(key, role):
        nonlocal seen_right
        tag = original(key, role)
        if role == "right":
            seen_right += 1
            if seen_right == 1:
                return bytes([tag[0] ^ 1]) + tag[1:]
        return tag

    monkeypatch.setattr(confirmation, "confirmation_tag", tamper_right_response)
    with pytest.raises(KeyConfirmationError):
        make(fake_kem_factory)
    assert seen_right == 2


@pytest.mark.parametrize("bound", [-1., float("nan"), float("inf"), True, "1"])
def test_invalid_clip_bound_rejected_at_construction(fake_kem_factory, bound):
    with pytest.raises(ValueError, match="clip_bound"):
        SchemeA(fake_kem_factory(42), scale_bits=0, modulus_bits=8, clip_bound=bound)


def test_clip_bound_checks_minimum_actual_count_and_rounding(fake_kem_factory):
    with pytest.raises(ValueError, match="clip_bound"):
        SchemeA(fake_kem_factory(42), scale_bits=0, modulus_bits=8, clip_bound=64.)
    agg = SchemeA(fake_kem_factory(42), scale_bits=0, modulus_bits=8, clip_bound=60.)
    with pytest.raises(ValueError, match="clip_bound"):
        agg.setup(SITES)
    assert len(agg.kem.key_material) == 0
    rounded = SchemeA(fake_kem_factory(42), scale_bits=0, modulus_bits=4, clip_bound=2.6)
    with pytest.raises(ValueError, match="Rounded clip_bound"):
        rounded.setup(SITES)
    safe = make(fake_kem_factory, scale_bits=0, modulus_bits=8, clip_bound=40.)
    np.testing.assert_array_equal(safe.aggregate_round(0, {s: np.array([40.]) for s in SITES}), [120.])
    with pytest.raises(ValueError, match="exceeds clip_bound"):
        safe.aggregate_round(1, {s: np.array([40.1]) for s in SITES})


@pytest.mark.parametrize("value", [100., -50.])
def test_simulator_detects_signed_aggregate_overflow(fake_kem_factory, value):
    agg = make(fake_kem_factory, scale_bits=0, modulus_bits=8)
    with pytest.raises(AggregateOverflowError):
        agg.aggregate_round(0, {s: np.array([value]) for s in SITES})
    assert agg.used_rounds == {0}


def test_overflow_guard_uses_unbounded_integer_sum(fake_kem_factory):
    sites = [str(i) for i in range(10)]
    agg = SchemeA(fake_kem_factory(42), scale_bits=0, modulus_bits=62)
    agg.setup(sites)
    with pytest.raises(AggregateOverflowError):
        agg.aggregate_round(0, {s: np.array([2 ** 60], dtype=np.float64) for s in sites})


@pytest.mark.parametrize("bits", [1, 7, 8, 9, 16, 31, 32, 62])
def test_wire_round_trip_and_size(bits):
    values = np.array([[0, 1, (1 << bits) - 1], [1, 0, 1]], dtype=np.int64).T
    assert not values.flags.c_contiguous
    message = serialize_masked(values, bits)
    assert len(message) == values.size * ((bits + 7) // 8)
    np.testing.assert_array_equal(deserialize_masked(message, values.shape, bits), values)
    with pytest.raises(ValueError, match="length"):
        deserialize_masked(message[:-1], values.shape, bits)
    with pytest.raises(ValueError, match="length"):
        deserialize_masked(message + b"\0", values.shape, bits)


def test_wire_empty_scalar_endianness_and_rejections():
    for values in (np.zeros((0, 3), dtype=np.int64), np.array(123, dtype=np.int64)):
        np.testing.assert_array_equal(deserialize_masked(serialize_masked(values, 9), values.shape, 9), values)
    assert serialize_masked(np.array([256, 511]), 9) == b"\x00\x01\xff\x01"
    with pytest.raises(ValueError, match="high bits"):
        deserialize_masked(b"\x00\x02", (1,), 9)
    for values in (np.array([-1]), np.array([512]), np.array([0.5])):
        with pytest.raises(ValueError, match="canonical"):
            serialize_masked(values, 9)


def test_payload_lengths_stats_deltas_and_per_site_masks(tmp_path, fake_kem_factory, monkeypatch):
    path = tmp_path / "run.jsonl"
    agg = make(fake_kem_factory, writer=writer_at(path), modulus_bits=17)
    initial = deepcopy(agg.setup_stats)
    assert len(agg.kem.key_material) == len(SITES) - 1
    serialized = []
    masks_called = []
    original_serialize = scheme_module.serialize_masked
    original_mask = scheme_module.pairwise_mask

    def capture_serialized(*args, **kwargs):
        message = original_serialize(*args, **kwargs)
        serialized.append(message)
        return message

    def capture_mask(seed, round_idx, left, right, shape, modulus_bits, *, epoch):
        masks_called.append((left, right, epoch, round_idx))
        return original_mask(seed, round_idx, left, right, shape, modulus_bits, epoch=epoch)

    monkeypatch.setattr(scheme_module, "serialize_masked", capture_serialized)
    monkeypatch.setattr(scheme_module, "pairwise_mask", capture_mask)
    updates = {site: np.array([0.25, -0.5]) for site in SITES}
    agg.aggregate_round(0, updates)
    stats = agg.stats()
    assert len(serialized) == len(SITES) + 1
    uploads, result_message = serialized[:-1], serialized[-1]
    for site, upload in zip(SITES, uploads):
        assert stats.bytes_sent[site] == len(upload)
        assert stats.bytes_received[site] == len(result_message)
    assert stats.bytes_received["reducer"] == sum(len(message) for message in uploads)
    assert stats.bytes_sent["reducer"] == sum(len(result_message) for _ in SITES)
    assert Counter(masks_called) == Counter({("A", "B", 0, 0): 2, ("A", "C", 0, 0): 2,
                                            ("B", "C", 0, 0): 2})
    agg.rekey()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for epoch, round_idx, expected in [(0, None, initial), (0, 0, stats), (1, None, agg.rekey_stats)]:
        selected = [row for row in rows if row["epoch"] == epoch and row["round"] == round_idx]
        for field in ("bytes_sent", "bytes_received"):
            assert sum_deltas(selected, field, agg.actors) == getattr(expected, field)
            assert all(value >= 0 for row in selected for value in row[field].values())
        assert all(row["phase"] != "comm" for row in selected)
        assert all(row["site"] in agg.actors for row in selected)
    round_rows = [row for row in rows if row["round"] == 0]
    assert {row["site"] for row in round_rows if row["phase"] == "crypto"} == set(SITES)


def test_metrics_deltas_follow_instance_not_site(tmp_path):
    path = tmp_path / "metrics.jsonl"
    metrics = Metrics(["A", "B", "reducer"], writer_at(path), epoch=3)
    with metrics.phase_timer("compute", site="A"):
        metrics.counter.transfer("A", "B", len(b"public payload"))
    with metrics.phase_timer("crypto", site="B"):
        pass
    with metrics.phase_timer("compute", site=None):
        metrics.counter.transfer("B", "reducer", len(b"another public payload"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(value == 0 for value in rows[1]["bytes_sent"].values())
    assert rows[-1]["site"] is None
    assert all(row["epoch"] == 3 and row["schema_version"] == 2 for row in rows)
    assert sum_deltas(rows, "bytes_sent", metrics.counter.value.bytes_sent) == metrics.counter.value.bytes_sent
    assert sum_deltas(rows, "bytes_received", metrics.counter.value.bytes_received) == metrics.counter.value.bytes_received


def test_pairwise_mask_regression_vector():
    # Regression vector generated by this implementation, NOT an independent check.
    expected = np.array([802328967, 3518897045, 2835203192, 461098831,
                         554616088, 2373655481, 1380080994, 2087473387], dtype=np.int64)
    actual = pairwise_mask(b"INSECURE regression seed", 7, "A", "B", (8,), 32, epoch=2)
    np.testing.assert_array_equal(actual, expected)


def test_hybrid_rejects_low_order_x25519_public_key(fake_kem_factory):
    kem = HybridKEM(fake_kem_factory(42))
    public, _ = kem.keygen()
    algorithm, _, pq_public = _unpack(public, b"pk", 3)
    low_order = _pack(b"pk", algorithm, bytes(32), pq_public)
    with pytest.raises(ValueError):
        kem.encaps(low_order)


@pytest.mark.parametrize("rekey", [False, True])
def test_setup_and_rekey_count_concrete_confirmation_messages(fake_kem_factory, monkeypatch, rekey):
    kem = fake_kem_factory(42)
    public_ciphertexts = []
    tags = []
    original_encaps = kem.encaps
    original_tag = confirmation.confirmation_tag

    def capture_encaps(public):
        ciphertext, shared = original_encaps(public)
        public_ciphertexts.append((public, ciphertext))
        return ciphertext, shared

    def capture_tag(key, role):
        tag = original_tag(key, role)
        tags.append((role, tag))
        return tag

    monkeypatch.setattr(kem, "encaps", capture_encaps)
    monkeypatch.setattr(confirmation, "confirmation_tag", capture_tag)
    agg = SchemeA(kem, scale_bits=12, modulus_bits=32)
    agg.setup(SITES)
    if rekey:
        public_ciphertexts.clear()
        tags.clear()
        agg.rekey()
    sent = dict.fromkeys(agg.actors, 0)
    received = dict.fromkeys(agg.actors, 0)
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    assert len(public_ciphertexts) == len(pairs)
    for index, (left, right) in enumerate(pairs):
        public, ciphertext = public_ciphertexts[index]
        roles_and_tags = tags[index * 4:(index + 1) * 4]
        assert [role for role, _ in roles_and_tags] == ["left", "left", "right", "right"]
        left_tag, right_tag = roles_and_tags[0][1], roles_and_tags[2][1]
        assert len(left_tag) == len(right_tag) == 32
        for sender, receiver, message in [(right, left, public), (left, right, ciphertext),
                                          (left, right, left_tag), (right, left, right_tag)]:
            sent[sender] += len(message)
            received[receiver] += len(message)
    assert agg.stats().bytes_sent == sent
    assert agg.stats().bytes_received == received
