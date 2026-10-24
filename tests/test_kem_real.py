"""Cluster-only real KEM checks; no downloaded or fabricated KAT vectors."""
import json
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from pqfl.kem.backends import BackendUnavailable, LibOQSKEM

pytestmark = pytest.mark.requires_oqs
from pqfl.kem.hybrid import HybridKEM


@pytest.fixture
def real_kem():
    try:
        return LibOQSKEM()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))


def test_component_sizes(real_kem):
    public, _ = real_kem.keygen()
    ciphertext, shared = real_kem.encaps(public)
    assert len(public) == 1184
    assert len(ciphertext) == 1088
    assert len(shared) == 32
    x_public = X25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    assert len(x_public) == 32


def test_real_round_trip(real_kem):
    public, secret = real_kem.keygen()
    ciphertext, shared = real_kem.encaps(public)
    if real_kem.decaps(secret, ciphertext) != shared:
        pytest.fail("Real KEM round trip failed; secret values suppressed")


def test_hybrid_round_trip(real_kem):
    kem = HybridKEM(real_kem)
    public, secret = kem.keygen()
    ciphertext, shared = kem.encaps(public)
    assert len(shared) == 32
    assert len(public) == 1259
    assert len(ciphertext) == 1149
    if kem.decaps(secret, ciphertext) != shared:
        pytest.fail("Hybrid KEM round trip failed; secret values suppressed")


def _rsp_vectors(path):
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        line = line.strip()
        if not line:
            if fields:
                if not {"sk", "ct", "ss"} <= fields.keys():
                    raise ValueError("RSP must contain sk, ct, ss in every record")
                yield tuple(bytes.fromhex(fields[key]) for key in ("sk", "ct", "ss"))
                fields = {}
        elif not line.startswith(("#", "[")):
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()


def _document(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    # ACVP may wrap the vector set after a protocol-version object.
    if isinstance(document, list):
        candidates = [item for item in document if isinstance(item, dict) and "testGroups" in item]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one ACVP vector set")
        document = candidates[0]
    return document


def _acvp_vectors(directory):
    prompt = _document(directory / "prompt.json")
    expected = _document(directory / "expectedResults.json")
    if prompt.get("algorithm") != "ML-KEM" or prompt.get("revision") != "FIPS203":
        raise ValueError("ACVP input must declare ML-KEM revision FIPS203")
    results = {(group["tgId"], test["tcId"]): test
               for group in expected["testGroups"] for test in group["tests"]}
    for group in prompt["testGroups"]:
        if group.get("parameterSet") != "ML-KEM-768" or group.get("function") != "decapsulation":
            continue
        for test in group["tests"]:
            answer = results[group["tgId"], test["tcId"]]
            yield tuple(bytes.fromhex(value) for value in (test["dk"], test["c"], answer["k"]))


def test_known_answers(real_kem):
    configured = os.environ.get("PQFL_KAT_DIR")
    if not configured:
        pytest.skip("PQFL_KAT_DIR unset: supply official ML-KEM-768 decapsulation KATs on the cluster")
    directory = Path(configured) / real_kem.mechanism
    if not directory.is_dir():
        pytest.fail("PQFL_KAT_DIR must contain a directory named for the actual KEM mechanism")
    vectors = []
    for path in sorted(directory.glob("*.rsp")):
        vectors.extend(_rsp_vectors(path))
    if (directory / "prompt.json").is_file():
        vectors.extend(_acvp_vectors(directory))
    if not vectors:
        pytest.fail("No supported decapsulation vectors found; see README KAT format")
    for secret, ciphertext, expected in vectors:
        if len(ciphertext) != 1088 or len(expected) != 32:
            pytest.fail("KAT dimensions do not match ML-KEM-768")
        if real_kem.decaps(secret, ciphertext) != expected:
            pytest.fail("Decapsulation KAT failed; secret values suppressed")
