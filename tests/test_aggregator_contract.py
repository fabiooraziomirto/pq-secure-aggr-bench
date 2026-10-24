"""Shared conformance suite; real A stays a real/skip test, not a fake fallback."""
from collections import Counter
from copy import deepcopy
import json

import numpy as np
import pytest

from conftest import InsecureTestAuth, real_auth_for_sites
from pqfl.aggregators import AGGREGATOR_REGISTRY
from pqfl.aggregators.errors import MissingParticipantError
from pqfl.kem.backends import BackendUnavailable, LibOQSKEM
from pqfl.kem.hybrid import HybridKEM
from pqfl.logging.metrics import ByteCounter, JSONLWriter, RoundStats

SITES = ['A', 'B', 'C']
UPDATES = {'A': np.array([[0.1, 0.25, -0.3]]),
           'B': np.array([[0.4, -0.5, 0.6]]),
           'C': np.array([[-0.7, 0.8, 0.9]])}
TOLERANCE = len(SITES) * 0.5 * 2 ** (-12) + np.finfo(float).eps


@pytest.fixture(params=[
    pytest.param(('A', 'real'), marks=pytest.mark.requires_oqs, id='A-real'),
    pytest.param(('A', 'functional'), id='A-functional-INSECURE'),
    pytest.param(('B', 'stub'), id='B-INSECURE_STUB'),
    pytest.param(('C', 'stub'), id='C-INSECURE_STUB'),
])
def registered(request, fake_kem_factory):
    name, variant = request.param
    assert set(AGGREGATOR_REGISTRY) == {'A', 'B', 'C'}
    cls = AGGREGATOR_REGISTRY[name]
    def create(writer=None):
        if name != 'A':
            return cls(writer=writer)
        if variant == 'real':
            auth = real_auth_for_sites(SITES)
            try:
                kem = HybridKEM(LibOQSKEM())
            except BackendUnavailable as exc:
                pytest.skip(str(exc))
        else:
            kem, auth = fake_kem_factory(42), InsecureTestAuth()
        return cls(kem, clip_bound=1., scale_bits=12, modulus_bits=32,
                   authenticator=auth, writer=writer)
    return name, variant, create


def assert_ledger(stats, actors):
    assert isinstance(stats, RoundStats)
    assert set(stats.bytes_sent) == set(stats.bytes_received) == set(actors)
    for actor in actors:
        assert stats.bytes_sent[actor] == sum(t.size for t in stats.transfers if t.sender == actor)
        assert stats.bytes_received[actor] == sum(t.size for t in stats.transfers if t.receiver == actor)
    assert sum(stats.bytes_sent.values()) == sum(stats.bytes_received.values())
    assert all(t.sender in actors and t.receiver in actors and t.size >= 0 for t in stats.transfers)
    assert all(not hasattr(t, 'payload') for t in stats.transfers)


def test_contract_sum_serialized_transfers_broadcast_and_stats(registered, tmp_path, monkeypatch):
    name, variant, create = registered
    depth, sends, transfers = Counter(), [], []
    original_send, original_transfer = ByteCounter.send, ByteCounter.transfer

    def send(counter, sender, receiver, message):
        assert isinstance(message, bytes)
        sends.append((counter, sender, receiver, len(message)))
        depth[counter] += 1
        try:
            return original_send(counter, sender, receiver, message)
        finally:
            depth[counter] -= 1

    def transfer(counter, sender, receiver, size):
        assert depth[counter] > 0, 'Aggregator transfer bypassed serialized delivery'
        transfers.append((counter, sender, receiver, size))
        return original_transfer(counter, sender, receiver, size)

    monkeypatch.setattr(ByteCounter, 'send', send)
    monkeypatch.setattr(ByteCounter, 'transfer', transfer)
    path = tmp_path / 'contract.jsonl'
    writer = JSONLWriter(path, run_id='contract-only', config_hash='test-only',
                         scheme=f'{name}-{variant}', dp_level='none', run_seed=42)
    agg = create(writer)
    assert agg.setup(SITES) is None
    setup_transfers = list(agg.stats().transfers)
    assert_ledger(agg.stats(), agg.actors)
    np.testing.assert_allclose(agg.aggregate_round(0, UPDATES), sum(UPDATES.values()), rtol=0, atol=TOLERANCE)
    assert sends == transfers
    stats = agg.stats()
    assert_ledger(stats, agg.actors)
    current = [(s, r, n) for counter_id, s, r, n in sends if counter_id is agg.metrics.counter]
    assert current == [(t.sender, t.receiver, t.size) for t in stats.transfers]
    observed_round = {(t.sender, t.receiver) for t in stats.transfers if t.size > 0}
    assert observed_round == agg.communication_edges['round']
    all_events = setup_transfers + stats.transfers
    observed = {(t.sender, t.receiver) for t in all_events if t.size > 0}
    assert observed == agg.communication_edges['setup'] | agg.communication_edges['round']
    for sender, receiver in observed:
        assert sum(t.size for t in all_events if (t.sender, t.receiver) == (sender, receiver)) > 0
    for site in SITES:
        assert (site, 'reducer') in observed_round
        assert ('reducer', site) in observed_round
    assert all(stats.bytes_sent[actor] > 0 and stats.bytes_received[actor] > 0 for actor in agg.actors)
    snapshot = agg.stats()
    snapshot.bytes_sent['A'] = -1
    snapshot.transfers.clear()
    assert agg.stats().bytes_sent['A'] >= 0 and agg.stats().transfers
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    round_rows = [row for row in rows if row['round'] == 0]
    for field in ('bytes_sent', 'bytes_received'):
        assert {actor: sum(row[field][actor] for row in round_rows) for actor in agg.actors} == getattr(stats, field)
    assert all(row['phase'] in {'auth', 'crypto', 'compute', 'comm'} for row in rows)
    assert all(row['phase'] != 'comm' and row['schema_version'] == 2 for row in rows)
    assert 'env' in rows[0]
    if name in {'B', 'C'}:
        assert agg.security_status == 'INSECURE_STUB'


def test_contract_participation_policy(registered):
    _, _, create = registered
    agg = create()
    agg.setup(SITES)
    assert agg.participation_policy == 'all_or_abort'
    before = deepcopy(agg.stats())
    with pytest.raises(MissingParticipantError):
        agg.aggregate_round(0, {site: update for site, update in UPDATES.items() if site != 'C'})
    assert agg.stats() == before
    assert agg.used_rounds == set()
    with pytest.raises(MissingParticipantError):
        agg.aggregate_round(0, dict(UPDATES, unexpected=np.zeros((1, 3))))
    assert agg.stats() == before
    np.testing.assert_allclose(agg.aggregate_round(0, UPDATES), sum(UPDATES.values()), rtol=0, atol=TOLERANCE)


def test_contract_rekey_or_declared_noop(registered):
    _, _, create = registered
    agg = create()
    agg.setup(SITES)
    agg.aggregate_round(0, UPDATES)
    previous_epoch = agg.epoch
    assert agg.rekey() is None
    assert agg.used_rounds == {0}
    assert_ledger(agg.stats(), agg.actors)
    if agg.rekey_policy == 'refresh_all_pairs':
        assert agg.epoch == previous_epoch + 1
        observed = {(t.sender, t.receiver) for t in agg.stats().transfers if t.size > 0}
        assert observed == agg.communication_edges['rekey']
    else:
        assert agg.rekey_policy == 'no_op'
        assert agg.epoch == previous_epoch
        assert sum(agg.stats().bytes_sent.values()) == 0
    with pytest.raises(ValueError, match='never reused'):
        agg.aggregate_round(0, UPDATES)
    np.testing.assert_allclose(agg.aggregate_round(1, UPDATES), sum(UPDATES.values()), rtol=0, atol=TOLERANCE)


def test_contract_shape_error_without_traffic(registered):
    _, _, create = registered
    agg = create()
    agg.setup(SITES)
    before = deepcopy(agg.stats())
    with pytest.raises(ValueError, match='shape'):
        agg.aggregate_round(0, dict(UPDATES, C=np.zeros((3,))))
    assert agg.stats() == before
    assert not agg.used_rounds


def test_counter_delivery_requires_serialized_payload_and_known_actors():
    counter = ByteCounter(['A', 'B'])
    with pytest.raises(TypeError):
        counter.send('A', 'B', np.zeros(3))
    with pytest.raises(ValueError, match='Unknown'):
        counter.send('A', 'unknown', b'public payload')
    with pytest.raises(ValueError, match='Self'):
        counter.send('A', 'A', b'public payload')
    assert not counter.value.transfers
