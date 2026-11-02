"""Classical X25519 KEM: the deliberately quantum-vulnerable baseline of the HNDL experiments."""
import numpy as np
import pytest
import yaml

from conftest import InsecureTestAuth
from pqfl.aggregators.scheme_a import SchemeA
from pqfl.config import load_run_config
from pqfl.kem.classical import ClassicalX25519KEM
from pqfl.runner import build_aggregator


def test_round_trip_and_fresh_secrets():
    kem = ClassicalX25519KEM()
    public, secret = kem.keygen()
    first, shared = kem.encaps(public)
    second, other = kem.encaps(public)
    assert kem.decaps(secret, first) == shared and len(shared) == 32
    assert first != second and shared != other


def test_it_is_labelled_quantum_vulnerable():
    kem = ClassicalX25519KEM()
    assert kem.quantum_safe is False and "classical" in kem.mechanism


def test_malformed_or_foreign_material_is_rejected():
    kem = ClassicalX25519KEM()
    public, secret = kem.keygen()
    ciphertext, _ = kem.encaps(public)
    with pytest.raises(ValueError):
        kem.encaps(public[:-1])
    with pytest.raises(ValueError):
        kem.decaps(secret, ciphertext + b"x")
    with pytest.raises(ValueError):
        kem.decaps(ClassicalX25519KEM().keygen()[1][:-1], ciphertext)


def test_scheme_a_aggregates_with_it():
    aggregator = SchemeA(ClassicalX25519KEM(), scale_bits=12, modulus_bits=32, clip_bound=1.0,
                         authenticator=InsecureTestAuth())
    aggregator.setup(["a", "b", "c"])
    rng = np.random.default_rng(0)
    updates = {s: rng.uniform(-0.5, 0.5, 8) for s in "abc"}
    np.testing.assert_allclose(aggregator.aggregate_round(0, updates), sum(updates.values()), atol=3 * 2 ** -13)


BASE = dict(run_id="t", sites=["a", "b", "c"], clip_bound=1.0, scale_bits=12, modulus_bits=32, ring_dim=16384,
            scaling_mod_size=50, dp_level="none", run_seed=0, rekey_every_rounds=None,
            auth={"method": "none_insecure"})


def config(tmp_path, **changes):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({**BASE, **changes}))
    return load_run_config(path)


def test_config_selects_the_kem_and_keeps_old_hashes(tmp_path):
    default = config(tmp_path)
    assert default.kem == "hybrid"
    assert config(tmp_path, kem="hybrid").kem == "hybrid"
    classical = config(tmp_path, kem="x25519_classical")
    assert classical.kem == "x25519_classical" and classical.config_hash("A") != default.config_hash("A")
    with pytest.raises(ValueError, match="kem"):
        config(tmp_path, kem="rsa")


@pytest.mark.parametrize("scheme", ["A", "C", "A-dropout"])
def test_runner_builds_kem_schemes_with_the_classical_kem(tmp_path, scheme):
    aggregator = build_aggregator(scheme, config(tmp_path, kem="x25519_classical"), ["a", "b", "c"], None)
    assert isinstance(aggregator.kem, ClassicalX25519KEM)


@pytest.mark.parametrize("scheme", ["N", "N-FP", "B", "B-BGV"])
def test_the_kem_choice_is_refused_where_no_kem_is_used(tmp_path, scheme):
    with pytest.raises(ValueError, match="kem"):
        build_aggregator(scheme, config(tmp_path, kem="x25519_classical"), ["a", "b", "c"], None)
