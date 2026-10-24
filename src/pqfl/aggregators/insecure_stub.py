"""INSECURE_STUB: functional plaintext transport, NOT CKKS or MPC."""
from copy import deepcopy

import numpy as np

from pqfl.aggregators.base import Aggregator
from pqfl.aggregators.errors import MissingParticipantError
from pqfl.logging.metrics import Metrics, RoundStats
from pqfl.masking.core import Tensor


def _serialize_plain(values: Tensor) -> bytes:
    return np.asarray(values, dtype='<f8').tobytes(order='C')


def _deserialize_plain(message: bytes, shape: tuple[int, ...]) -> Tensor:
    values = np.frombuffer(message, dtype='<f8')
    if values.size != int(np.prod(shape)):
        raise ValueError('Unexpected plaintext stub payload length')
    return values.reshape(shape).copy()


class InsecureSumStub(Aggregator):
    """INSECURE_STUB: star topology, float64 sum, no cryptography.

    Serialized payloads are actually delivered to a simulated reducer and the
    serialized result to each site. Byte counts describe this stub only and must
    never be reported as CKKS/MPC measurements or security evidence.
    """

    security_status = 'INSECURE_STUB'
    participation_policy = 'all_or_abort'
    rekey_policy = 'no_op'

    def __init__(self, *, writer=None):
        self.writer = writer
        self.sites = None
        self.actors = []
        self.communication_edges = {}
        self.used_rounds = set()
        self.epoch = 0

    def setup(self, site_ids: list[str]) -> None:
        if self.sites is not None:
            raise ValueError('Use a new instance for a new session')
        if len(site_ids) < 2 or any(not isinstance(site, str) or not site for site in site_ids):
            raise ValueError('Require at least two nonempty site IDs')
        if len(set(site_ids)) != len(site_ids) or 'reducer' in site_ids:
            raise ValueError('Duplicate or reserved site ID')
        self.sites = sorted(site_ids)
        self.actors = [*self.sites, 'reducer']
        edges = {(site, 'reducer') for site in self.sites} | {('reducer', site) for site in self.sites}
        self.communication_edges = {'setup': set(), 'round': edges, 'rekey': set()}
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        with self.metrics.phase_timer('compute', site='reducer'):
            pass  # Setup has no messages, keys or cryptographic work.

    def rekey(self) -> None:
        if self.sites is None:
            raise ValueError('Call setup first')
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        with self.metrics.phase_timer('compute', site='reducer'):
            pass  # Declared no-op: no epoch change and no invented messages.

    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor]) -> Tensor:
        if self.sites is None:
            raise ValueError('Call setup first')
        if type(round_idx) is not int or round_idx < 0 or round_idx in self.used_rounds:
            raise ValueError('Round must be nonnegative and never reused within a session')
        if set(updates) != set(self.sites):
            raise MissingParticipantError('all_or_abort requires exactly the configured participants')
        shape = np.asarray(updates[self.sites[0]]).shape
        if any(np.asarray(value).shape != shape for value in updates.values()):
            raise ValueError('Mismatched tensor shapes')
        if any(not np.all(np.isfinite(value)) for value in updates.values()):
            raise ValueError('Nonfinite plaintext update')
        self.used_rounds.add(round_idx)
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        self.metrics.round_idx = round_idx
        messages = {}
        for site in self.sites:
            with self.metrics.phase_timer('compute', site=site):
                payload = _serialize_plain(updates[site])
                messages[site] = self.metrics.counter.send(site, 'reducer', payload)
        with self.metrics.phase_timer('compute', site='reducer'):
            result = np.zeros(shape, dtype=np.float64)
            for message in messages.values():
                result += _deserialize_plain(message, shape)
            payload = _serialize_plain(result)
            delivered = {site: self.metrics.counter.send('reducer', site, payload) for site in self.sites}
        result = None
        for site in self.sites:
            with self.metrics.phase_timer('compute', site=site):
                decoded = _deserialize_plain(delivered[site], shape)
                if result is None:
                    result = decoded
        return result

    def stats(self) -> RoundStats:
        if self.sites is None:
            raise ValueError('Call setup first')
        return deepcopy(self.metrics.counter.value)
