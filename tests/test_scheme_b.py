"""Scheme B: threshold CKKS (OpenFHE); skipped when the openfhe wheel is absent."""
import numpy as np
import pytest

from conftest import InsecureTestAuth
from pqfl.aggregators.scheme_b import SchemeB
from pqfl.he.ckks import pack, unpack
from pqfl.logging.metrics import ByteCounter

pytestmark = pytest.mark.requires_openfhe
SITES = ["A", "B", "C"]
UPDATES = {"A": np.array([0.1, 0.25, -0.3]), "B": np.array([0.4, -0.5, 0.6]),
           "C": np.array([-0.7, 0.8, 0.9])}
TOL = 1e-8  # CKKS with 50-bit scaling: observed error ~1e-12 for values <= 1


def make(sites=SITES):
    agg = SchemeB(clip_bound=1., authenticator=InsecureTestAuth())
    agg.setup(sites)
    return agg


def test_pack_roundtrip_and_malformed():
    assert unpack(pack([b"a", b"", b"ccc"])) == [b"a", b"", b"ccc"]
    with pytest.raises(ValueError):
        unpack(pack([b"a"]) + b"x")


def test_sum_within_declared_tolerance_and_shape():
    out = make().aggregate_round(0, {k: v.reshape(1, 3) for k, v in UPDATES.items()})
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out, sum(UPDATES.values()).reshape(1, 3), rtol=0, atol=TOL)


def test_multi_ciphertext_vector_larger_than_slots():
    rng = np.random.default_rng(1)
    updates = {s: rng.uniform(-1, 1, 20000) for s in SITES}  # 3 chunks of 8192 slots
    np.testing.assert_allclose(make().aggregate_round(0, updates), sum(updates.values()),
                               rtol=0, atol=TOL)


def test_reducer_receives_only_ciphertexts_not_plaintext_values(monkeypatch):
    seen = []
    original = ByteCounter.send

    def send(counter, sender, receiver, message):
        seen.append((sender, receiver, message))
        return original(counter, sender, receiver, message)

    monkeypatch.setattr(ByteCounter, "send", send)
    agg = make()
    seen.clear()
    agg.aggregate_round(0, UPDATES)
    for site, update in UPDATES.items():
        plain = np.asarray(update, dtype="<f8").tobytes()
        assert all(plain not in m for s, r, m in seen if s == site)
    uploads = [m for s, r, m in seen if r == "reducer" and s in SITES]
    assert len(uploads) == 6 and all(len(m) > 100_000 for m in uploads)


def test_rekey_refreshes_joint_key_and_keeps_sum():
    agg = make()
    first = agg._joint["A"]
    agg.aggregate_round(0, UPDATES)
    agg.rekey()
    assert agg.epoch == 1 and agg._joint["A"] != first
    np.testing.assert_allclose(agg.aggregate_round(1, UPDATES), sum(UPDATES.values()), rtol=0, atol=TOL)


def test_clip_bound_and_nonfinite_rejected_before_traffic():
    agg = make()
    before = agg.stats()
    for bad in (np.array([2., 0., 0.]), np.array([np.nan, 0., 0.])):
        with pytest.raises(ValueError):
            agg.aggregate_round(0, dict(UPDATES, A=bad))
    assert agg.stats() == before and not agg.used_rounds


def test_authentication_failure_aborts_setup():
    class Reject:
        def issue(self, site_id, epoch, public_key):
            from pqfl.auth import AuthProof
            return AuthProof()

        def verify(self, *args):
            raise RuntimeError("rejected")

    with pytest.raises(RuntimeError, match="rejected"):
        SchemeB(clip_bound=1., authenticator=Reject()).setup(SITES)
