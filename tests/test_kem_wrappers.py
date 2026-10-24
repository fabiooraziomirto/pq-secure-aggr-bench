"""Lightweight adapter/combiner tests; these do not certify real ML-KEM."""
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from pqfl.kem import HybridKEM
from pqfl.kem import backends
from pqfl.kem.hybrid import _pack, _unpack
from pqfl.aggregators.scheme_a import SchemeA
import numpy as np


def module_with(mechanisms):
    class DetailsOnly:
        details = {"length_secret_key": 1, "length_public_key": 1184,
                   "length_ciphertext": 1088, "length_shared_secret": 32}

        def __init__(self, mechanism):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return SimpleNamespace(KeyEncapsulation=DetailsOnly,
                           get_enabled_kem_mechanisms=lambda: mechanisms,
                           oqs_version=lambda: "test-double", oqs_python_version=lambda: "test-double")


def test_missing_backend_fails_closed(monkeypatch):
    monkeypatch.setattr(backends.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(backends.BackendUnavailable, match="No automatic installation"):
        backends.LibOQSKEM()
    with pytest.raises(backends.BackendUnavailable):
        HybridKEM()


def test_no_native_library_means_no_binding_import(monkeypatch):
    monkeypatch.setattr(backends.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(backends.ctypes.util, "find_library", lambda _: None)
    monkeypatch.delenv("OQS_INSTALL_PATH", raising=False)
    def missing_library(_):
        raise OSError("test-only missing native library")
    monkeypatch.setattr(backends.ctypes, "CDLL", missing_library)
    def forbidden_import(_):
        pytest.fail("Binding import must not run without native prerequisites")
    monkeypatch.setattr(backends.importlib, "import_module", forbidden_import)
    with pytest.raises(backends.BackendUnavailable):
        backends.LibOQSKEM()


def test_capability_selection(monkeypatch):
    monkeypatch.setattr(backends, "load_oqs", lambda: module_with(["Kyber768", "ML-KEM-768"]))
    assert backends.LibOQSKEM().mechanism == "ML-KEM-768"
    monkeypatch.setattr(backends, "load_oqs", lambda: module_with(["Kyber768"]))
    with pytest.raises(backends.BackendUnavailable, match="legacy"):
        backends.LibOQSKEM()
    with pytest.warns(RuntimeWarning, match="NOT standardized"):
        assert backends.LibOQSKEM(allow_legacy=True).mechanism == "Kyber768"
    monkeypatch.setattr(backends, "load_oqs", lambda: module_with([]))
    with pytest.raises(backends.BackendUnavailable, match="Neither"):
        backends.LibOQSKEM()


def test_hybrid_with_test_only_kem(fake_kem_factory):
    kem = HybridKEM(fake_kem_factory(42))
    public, secret = kem.keygen()
    ciphertext, shared = kem.encaps(public)
    assert kem.decaps(secret, ciphertext) == shared
    assert len(shared) == 32
    algorithm, x_public, pq_public = _unpack(public, b"pk", 3)
    _, pq_secret, _ = _unpack(secret, b"sk", 3)
    assert len(x_public) == 32
    assert pq_public == pq_secret  # INSECURE test double only.
    for malformed in (b"", ciphertext[:-1], ciphertext + b"extra"):
        with pytest.raises(ValueError):
            kem.decaps(secret, malformed)
    with pytest.raises(ValueError, match="mechanism"):
        kem.encaps(_pack(b"pk", b"other", x_public, pq_public))


def test_combiner_binds_transcript_and_order():
    # Synthetic non-secret unit inputs, not known-answer cryptographic vectors.
    x, pq = bytes(range(32)), bytes(range(32, 64))
    actual = HybridKEM._combine(x, pq, b"public", b"ciphertext")
    reference = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                     info=_pack(b"combiner", b"public", b"ciphertext")).derive(x + pq)
    assert actual == reference
    assert actual != HybridKEM._combine(pq, x, b"public", b"ciphertext")
    assert actual != HybridKEM._combine(x, pq, b"changed", b"ciphertext")
    assert actual != HybridKEM._combine(x, pq, b"public", b"changed")


def test_hybrid_in_scheme_a(fake_kem_factory):
    agg = SchemeA(HybridKEM(fake_kem_factory(42)), scale_bits=10, modulus_bits=32)
    agg.setup(["A", "B", "C"])
    updates = {site: np.array([0.5, -1.]) for site in ["A", "B", "C"]}
    np.testing.assert_array_equal(agg.aggregate_round(0, updates), sum(updates.values()))


def test_liboqs_adapter_calls(fake_kem_factory, monkeypatch):
    # Model binding calls with an explicitly INSECURE test-only KEM.
    fake = fake_kem_factory(42)
    native = module_with(["ML-KEM-768"])

    class Binding:
        details = {"length_secret_key": 32, "length_public_key": 1184,
                   "length_ciphertext": 1088, "length_shared_secret": 32}

        def __init__(self, algorithm, secret_key=None):
            assert algorithm == "ML-KEM-768"
            self.secret_key = secret_key

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def generate_keypair(self):
            public, self.secret_key = fake.keygen()
            return public + bytes(1184 - len(public))

        def export_secret_key(self):
            return self.secret_key

        def encap_secret(self, public):
            ciphertext, shared = fake.encaps(public)
            return ciphertext + bytes(1088 - len(ciphertext)), shared

        def decap_secret(self, ciphertext):
            return fake.decaps(self.secret_key, ciphertext[:32])

    native.KeyEncapsulation = Binding
    monkeypatch.setattr(backends, "load_oqs", lambda: native)
    kem = backends.LibOQSKEM()
    public, secret = kem.keygen()
    ciphertext, shared = kem.encaps(public)
    assert kem.decaps(secret, ciphertext) == shared
    with pytest.raises(ValueError, match="secret key length"):
        kem.decaps(secret[:-1], ciphertext)
    with pytest.raises(ValueError, match="ciphertext"):
        kem.decaps(secret, ciphertext[:-1])
    with pytest.raises(ValueError, match="public key"):
        kem.encaps(public[:-1])
