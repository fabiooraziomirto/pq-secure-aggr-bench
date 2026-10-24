"""Authentication tests use provisioned pins; no automatic trust bootstrap."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from pqfl.auth import AuthenticationError, AuthProof, PinnedFingerprintAuth
from pqfl.auth.mldsa import authentication_message
from pqfl.aggregators.errors import MissingParticipantError
from pqfl.aggregators.scheme_a import SchemeA
from conftest import InsecureTestAuth

SITES = ['A', 'B', 'C']


def provisioned(factory, epochs=(0,)):
    kem = factory(42)
    keys = {(epoch, owner): kem.keygen() for epoch in epochs for owner in SITES[1:]}
    pins = {key: hashlib.sha256(pair[0]).hexdigest() for key, pair in keys.items()}
    auth = PinnedFingerprintAuth({verifier: dict(pins) for verifier in SITES})
    return kem, keys, auth


def test_default_pinning_rejects_unconfigured_keys(fake_kem_factory, monkeypatch):
    kem = fake_kem_factory(42)
    calls = []
    monkeypatch.setattr(kem, 'encaps', lambda public: calls.append(public))
    agg = SchemeA(kem, clip_bound=1., scale_bits=12, modulus_bits=32)
    with pytest.raises(AuthenticationError, match='missing pin'):
        agg.setup(SITES)
    assert calls == []
    assert not agg.seeds


def test_mitm_key_substitution_rejected_before_kem(fake_kem_factory, monkeypatch):
    kem, keys, auth = provisioned(fake_kem_factory)
    calls = []
    monkeypatch.setattr(kem, 'encaps', lambda public: calls.append(public))
    def substituted(owner, epoch):
        public, secret = keys[epoch, owner]
        return bytes([public[0] ^ 1]) + public[1:], secret
    agg = SchemeA(kem, clip_bound=1., scale_bits=12, modulus_bits=32,
                  authenticator=auth, key_provider=substituted)
    with pytest.raises(AuthenticationError):
        agg.setup(SITES)
    assert calls == []
    assert not agg.seeds


def test_fingerprint_mismatch_and_epoch_replay(fake_kem_factory):
    _, keys, auth = provisioned(fake_kem_factory)
    public = keys[0, 'B'][0]
    auth.verify('A', 'B', 0, public, AuthProof())
    with pytest.raises(AuthenticationError):
        auth.verify('A', 'B', 1, public, AuthProof())
    wrong = PinnedFingerprintAuth({'A': {(0, 'B'): '00' * 32}})
    with pytest.raises(AuthenticationError):
        wrong.verify('A', 'B', 0, public, AuthProof())
    with pytest.raises(AuthenticationError):
        auth.verify('A', 'B', 0, public, AuthProof(signature=b'unexpected'))


def test_pinned_config_and_preprovisioned_rekey(fake_kem_factory):
    kem, keys, auth = provisioned(fake_kem_factory, (0, 1))
    config = {verifier: {epoch: {owner: fingerprint for (e, owner), fingerprint in pins.items() if e == epoch}
                         for epoch in (0, 1)} for verifier, pins in auth.registries.items()}
    auth = PinnedFingerprintAuth.from_config(config)
    agg = SchemeA(kem, clip_bound=1., scale_bits=12, modulus_bits=32,
                  authenticator=auth, key_provider=lambda owner, epoch: keys[epoch, owner])
    agg.setup(SITES)
    updates = {site: np.array([0.25, -0.5]) for site in SITES}
    np.testing.assert_array_equal(agg.aggregate_round(0, updates), sum(updates.values()))
    agg.rekey()
    np.testing.assert_array_equal(agg.aggregate_round(1, updates), sum(updates.values()))


def test_clip_bound_required_and_actual_count_validated(fake_kem_factory):
    with pytest.raises(TypeError, match='clip_bound'):
        SchemeA(fake_kem_factory(42), scale_bits=0, modulus_bits=8)
    with pytest.raises(ValueError, match='clip_bound'):
        SchemeA(fake_kem_factory(42), scale_bits=0, modulus_bits=8, clip_bound=None)
    with pytest.raises(ValueError, match='clip_bound'):
        SchemeA(fake_kem_factory(42), scale_bits=0, modulus_bits=8, clip_bound=64.)
    agg = SchemeA(fake_kem_factory(42), scale_bits=0, modulus_bits=8, clip_bound=60.)
    with pytest.raises(ValueError, match='clip_bound'):
        agg.setup(SITES)


def test_all_or_abort_produces_no_partial_result(fake_kem_factory):
    agg = SchemeA(fake_kem_factory(42), scale_bits=12, modulus_bits=32, clip_bound=1.,
                  authenticator=InsecureTestAuth())
    agg.setup(SITES)
    before = deepcopy(agg.stats())
    assert agg.participation_policy == 'all_or_abort'
    with pytest.raises(MissingParticipantError):
        agg.aggregate_round(0, {'A': np.zeros(1), 'B': np.zeros(1)})
    assert agg.stats() == before
    assert not agg.used_rounds


def test_authentication_message_canonical_tuple():
    assert json.loads(authentication_message('A', 7, b'public-test-input')) == [
        'pqfl/auth/v1', 7, 'A', b'public-test-input'.hex()]


def test_mitm_at_delivery_boundary_is_rejected_before_seed(fake_kem_factory, monkeypatch):
    from pqfl.logging.metrics import ByteCounter
    kem, keys, auth = provisioned(fake_kem_factory)
    calls, changed = [], False
    original_send = ByteCounter.send
    def intercepted(counter, sender, receiver, message):
        nonlocal changed
        if not changed and (sender, receiver) == ('B', 'A'):
            changed = True
            message = bytes([message[0] ^ 1]) + message[1:]
        return original_send(counter, sender, receiver, message)
    monkeypatch.setattr(ByteCounter, 'send', intercepted)
    monkeypatch.setattr(kem, 'encaps', lambda public: calls.append(public))
    agg = SchemeA(kem, clip_bound=1., scale_bits=12, modulus_bits=32,
                  authenticator=auth, key_provider=lambda owner, epoch: keys[epoch, owner])
    with pytest.raises(AuthenticationError):
        agg.setup(SITES)
    assert changed and not calls and not agg.seeds
