import base64
import json
from pathlib import Path
import yaml
import numpy as np
import pytest

from pqfl.aggregators.scheme_a import SchemeA
from conftest import InsecureTestAuth
from pqfl.aggregators.scheme_b import SchemeB
from pqfl.aggregators.scheme_c import SchemeC
from pqfl.channel.hybrid import HybridChannel
from pqfl.kem.backends import OpenSSLKEM
from pqfl.logging.metrics import ByteCounter, JSONLWriter
from pqfl.masking.core import quantize, dequantize, pairwise_mask, recover_masks

SITES = ["A", "B", "C"]
UPDATES = {"A": np.array([0.25, -1.5]), "B": np.array([2., 0.75]),
           "C": np.array([-0.5, 3.])}


def make(factory, order=SITES, seed=42, writer=None):
    agg = SchemeA(factory(seed), authenticator=InsecureTestAuth(), clip_bound=4., scale_bits=12, modulus_bits=32, writer=writer)
    agg.setup(order)
    return agg


def test_masked_sum_matches_plaintext(fake_kem_factory):
    agg = make(fake_kem_factory)
    np.testing.assert_array_equal(agg.aggregate_round(0, UPDATES), sum(UPDATES.values()))
    assert agg.stats().bytes_sent == dict(A=8, B=8, C=8, reducer=24)
    assert agg.stats().bytes_received == dict(A=8, B=8, C=8, reducer=24)
    assert agg.setup_stats.bytes_sent["A"] > 0


def test_masks_change_across_rounds():
    first = pairwise_mask(b"test-only-seed", 0, "A", "B", (128,), 32, epoch=0)
    second = pairwise_mask(b"test-only-seed", 1, "A", "B", (128,), 32, epoch=0)
    assert not np.array_equal(first, second)


def test_site_order_invariant(fake_kem_factory):
    first = make(fake_kem_factory).aggregate_round(0, UPDATES)
    second = make(fake_kem_factory, ["C", "A", "B"]).aggregate_round(0, dict(reversed(list(UPDATES.items()))))
    np.testing.assert_array_equal(first, second)


def test_seed_determinism(fake_kem_factory):
    first, second = make(fake_kem_factory), make(fake_kem_factory)
    assert first.seeds == second.seeds
    for pair in first.seeds:
        np.testing.assert_array_equal(
            pairwise_mask(first.seeds[pair], 0, *pair, (8,), 32, epoch=0),
            pairwise_mask(second.seeds[pair], 0, *pair, (8,), 32, epoch=0))
    np.testing.assert_array_equal(first.aggregate_round(0, UPDATES), second.aggregate_round(0, UPDATES))


def test_quantization_error_bound():
    scale_bits, modulus_bits = 10, 32
    values = np.array([-3.14159, -0.0004, 0., 0.99999, 12.3456])
    tolerance = 0.5 / (2 ** scale_bits) + np.finfo(float).eps
    restored = dequantize(quantize(values, scale_bits, modulus_bits), scale_bits, modulus_bits)
    assert np.max(np.abs(restored - values)) <= tolerance


def test_logs_contain_no_secrets(tmp_path, fake_kem_factory, monkeypatch):
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    derived_keys = []
    original_derive = HKDF.derive

    def capture_derive(self, material):
        result = original_derive(self, material)
        derived_keys.append(result)
        return result

    monkeypatch.setattr(HKDF, "derive", capture_derive)
    from pqfl.masking import confirmation
    tags = []
    original_tag = confirmation.confirmation_tag

    def capture_tag(*args):
        tag = original_tag(*args)
        tags.append(tag)
        return tag

    monkeypatch.setattr(confirmation, "confirmation_tag", capture_tag)
    path = tmp_path / "run.jsonl"
    config = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    config.update(run_id="unit-test", dp_level="none", run_seed=42)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    writer = JSONLWriter.from_config(path, config_path)
    agg = make(fake_kem_factory, writer=writer)
    agg.aggregate_round(0, UPDATES)
    text = path.read_text()
    rows = [json.loads(line) for line in text.splitlines()]
    assert len(rows) == 29
    required = {"run_id", "config_hash", "scheme", "dp_level", "run_seed", "schema_version", "round", "phase", "site", "epoch",
                "duration_s", "bytes_sent", "bytes_received", "peak_mem_gpu_mb",
                "peak_mem_host_mb", "timestamp", "library_versions"}
    for row in rows:
        assert set(row) == required | ({"env"} if row is rows[0] else set())
        if row is rows[0]:
            assert {"python", "cryptography", "cpu_model", "liboqs", "enabled_kems", "enabled_sigs"} <= row["env"].keys()
        assert row["run_seed"] == config["run_seed"]
        assert row["schema_version"] == 2
        assert row["site"] in [*SITES, "reducer"]
        assert row["epoch"] == 0
        assert row["phase"] != "comm"
        assert "seed" not in row
        assert row["duration_s"] >= 0
    # Scan every generated key/shared secret/ciphertext and each derived mask.
    material = list(agg.kem.generated_material)
    for pair, seed in agg.seeds.items():
        material.append(pairwise_mask(seed, 0, *pair, (2,), 32, epoch=0).tobytes())
    material.extend(derived_keys)
    material.extend(tags)
    assert tags and all(len(tag) == 32 for tag in tags)
    for seed in material:
        assert seed.hex() not in text
        assert base64.b64encode(seed).decode("ascii") not in text
        assert str(seed) not in text
    for forbidden in ("secret", "public_key", "private_key", "mask", "ciphertext", "tag"):
        assert forbidden not in text


def test_round_reuse_dropout_and_shapes_rejected(fake_kem_factory):
    agg = make(fake_kem_factory)
    agg.aggregate_round(0, UPDATES)
    with pytest.raises(ValueError):
        agg.aggregate_round(0, UPDATES)
    with pytest.raises(ValueError):
        agg.aggregate_round(1, {"A": UPDATES["A"]})
    with pytest.raises(ValueError):
        agg.aggregate_round(1, dict(UPDATES, C=np.zeros(3)))
    result = agg.stats()
    result.bytes_sent["A"] = -1
    assert agg.stats().bytes_sent["A"] >= 0


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [2**20]])
def test_quantization_range_rejected(values):
    with pytest.raises(ValueError):
        quantize(np.array(values), 12, 32)


def test_byte_counter():
    counter = ByteCounter(SITES)
    counter.transfer("A", "B", 123)
    assert counter.value.bytes_sent["A"] == counter.value.bytes_received["B"] == 123
    with pytest.raises(ValueError):
        counter.transfer("A", "B", -1)


def test_unimplemented_crypto_fails_closed_and_stubs_are_labelled():
    for backend in (OpenSSLKEM,):
        with pytest.raises(NotImplementedError):
            backend()
    for scheme in (SchemeB(), SchemeC()):
        assert scheme.security_status == "INSECURE_STUB"
        scheme.setup(SITES)
        np.testing.assert_array_equal(scheme.aggregate_round(0, UPDATES), sum(UPDATES.values()))
    with pytest.raises(NotImplementedError):
        HybridChannel().establish()
    with pytest.raises(NotImplementedError):
        recover_masks()
