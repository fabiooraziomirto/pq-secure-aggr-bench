"""ML-DSA-65 tests are real or explicitly skipped, never mocked signatures."""
import base64
import json

import numpy as np
import pytest

from pqfl.auth import AuthenticationError, MLDSAAuth
from pqfl.kem.backends import BackendUnavailable, LibOQSKEM, load_oqs
from pqfl.kem.hybrid import HybridKEM
from pqfl.aggregators.scheme_a import SchemeA
from pqfl.logging.metrics import JSONLWriter

pytestmark = pytest.mark.requires_oqs
SITES = ['A', 'B', 'C']


@pytest.fixture
def real_auth():
    try:
        oqs = load_oqs()
        if 'ML-DSA-65' not in oqs.get_enabled_sig_mechanisms():
            pytest.skip('Native liboqs does not enable ML-DSA-65')
        public_keys, secrets = {}, {}
        for site in SITES:
            with oqs.Signature('ML-DSA-65') as signer:
                public_keys[site] = bytes(signer.generate_keypair())
                secrets[site] = bytes(signer.export_secret_key())
        return MLDSAAuth(public_keys, secrets, {site: dict(public_keys) for site in SITES})
    except BackendUnavailable as exc:
        pytest.skip(str(exc))


def test_mldsa_binds_epoch_owner_and_kem_public_key(real_auth):
    public = b'public test input, not a KEM vector'
    proof = real_auth.issue('B', 0, public)
    assert len(proof.signature) == real_auth.signature_bytes == 3309
    real_auth.verify('A', 'B', 0, public, proof)
    for owner, epoch, key in [('B', 1, public), ('B', 0, public + b'changed'), ('C', 0, public)]:
        with pytest.raises(AuthenticationError):
            real_auth.verify('A', owner, epoch, key, proof)


def test_mldsa_wire_bytes_and_no_signature_in_logs(real_auth, tmp_path, monkeypatch):
    try:
        kem = HybridKEM(LibOQSKEM())
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    proofs, public_keys = {}, {}
    original_issue = real_auth.issue
    def issue(owner, epoch, public):
        proof = original_issue(owner, epoch, public)
        proofs[owner], public_keys[owner] = proof, public
        return proof
    monkeypatch.setattr(real_auth, 'issue', issue)
    path = tmp_path / 'run.jsonl'
    writer = JSONLWriter(path, run_id='real-functional', config_hash='test-only', scheme='A',
                         dp_level='none', run_seed=42)
    agg = SchemeA(kem, scale_bits=12, modulus_bits=32, clip_bound=1.,
                  authenticator=real_auth, writer=writer)
    agg.setup(SITES)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    auth_rows = [row for row in rows if row['phase'] == 'auth']
    expected = sum(len(public_keys[right]) + len(proofs[right].identity_public) + len(proofs[right].signature)
                   for left, right in [('A', 'B'), ('A', 'C'), ('B', 'C')])
    assert sum(sum(row['bytes_sent'].values()) for row in auth_rows) == expected
    assert sum(sum(row['bytes_received'].values()) for row in auth_rows) == expected
    assert real_auth.signature_bytes == 3309
    text = path.read_text()
    for proof in proofs.values():
        assert proof.signature.hex() not in text
        assert base64.b64encode(proof.signature).decode() not in text
    updates = {site: np.array([0.25, -0.5]) for site in SITES}
    np.testing.assert_array_equal(agg.aggregate_round(0, updates), sum(updates.values()))
