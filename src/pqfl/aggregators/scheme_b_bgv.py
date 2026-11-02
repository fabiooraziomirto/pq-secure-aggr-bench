import math
from numbers import Real

import numpy as np

from pqfl.aggregators.errors import MissingParticipantError
from pqfl.aggregators.scheme_b import SchemeB
from pqfl.auth import PinnedFingerprintAuth
from pqfl.he.bgv import ThresholdBGV
from pqfl.he.ckks import pack, unpack
from pqfl.logging.metrics import Metrics
from pqfl.masking.core import Tensor


class SchemeBGV(SchemeB):
    """B with threshold BGV: fixed-point integers summed exactly under a joint key.

    Same key chain, authentication, stages and traffic pattern as the CKKS scheme B.
    Multiparty noise flooding is on (in CKKS it had no effect). The only numerical
    error is quantization to 2^-scale_bits; the sum of integers is exact as long as
    n * round(clip_bound * 2^scale_bits) < plaintext_modulus / 2, checked at setup.
    The result travels as int64 (8 bytes per element), like B's float64.
    """

    def __init__(self, *, clip_bound: float, scale_bits: int = 20, ring_dim: int = 1 << 14,
                 plaintext_bits: int = 31, writer=None, authenticator=None):
        if isinstance(clip_bound, bool) or not isinstance(clip_bound, Real):
            raise ValueError("clip_bound must be a finite nonnegative number")
        if not math.isfinite(clip_bound) or clip_bound < 0:
            raise ValueError("clip_bound must be finite and nonnegative")
        if type(scale_bits) is not int or not 0 <= scale_bits < plaintext_bits - 1:
            raise ValueError("scale_bits must lie in [0, plaintext_bits - 1)")
        self.clip_bound, self.scale_bits = clip_bound, scale_bits
        self.ring_dim, self.plaintext_bits = ring_dim, plaintext_bits
        self.he = None
        self.authenticator = authenticator if authenticator is not None else PinnedFingerprintAuth()
        self.writer = writer
        self.sites = None
        self.actors = []
        self.communication_edges = {}
        self.used_rounds = set()
        self.epoch = 0
        self._ready = False

    def setup(self, site_ids: list[str]) -> None:
        if self.sites is not None:
            raise ValueError("Use a new instance for a new session")
        if len(site_ids) < 2:
            raise ValueError("Require at least two nonempty site IDs")
        self.he = ThresholdBGV(parties=len(site_ids), ring_dim=self.ring_dim,
                               plaintext_bits=self.plaintext_bits)
        scaled = int(np.rint(self.clip_bound * 2 ** self.scale_bits))
        if len(site_ids) * scaled >= self.he.plaintext_modulus // 2:
            raise ValueError("clip_bound cannot guarantee an exact signed sum in the plaintext modulus")
        super().setup(site_ids)

    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor]) -> Tensor:
        if not self._ready:
            raise ValueError("A successful setup or rekey is required")
        if type(round_idx) is not int or round_idx < 0 or round_idx in self.used_rounds:
            raise ValueError("Round must be nonnegative and never reused within a session")
        if set(updates) != set(self.sites):
            raise MissingParticipantError("all_or_abort requires exactly the configured participants")
        shape = np.asarray(updates[self.sites[0]]).shape
        if any(np.asarray(u).shape != shape for u in updates.values()):
            raise ValueError("Mismatched tensor shapes")
        for site in self.sites:
            values = np.asarray(updates[site], dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(np.abs(values) > self.clip_bound):
                raise ValueError("Update exceeds clip_bound or is not finite")
        self.used_rounds.add(round_idx)
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        self.metrics.round_idx = round_idx
        send = self.metrics.counter.send
        length = int(np.prod(shape))
        scale = 2 ** self.scale_bits
        uploads = {}
        self.metrics.counter.stage = 0
        for site in self.sites:
            with self.metrics.phase_timer("compute", site=site):
                quantized = np.rint(np.asarray(updates[site], dtype=np.float64).ravel() * scale)
                integers = quantized.astype(np.int64).tolist()
            with self.metrics.phase_timer("crypto", site=site):
                payload = pack(self.he.encrypt(self._joint[site], integers))
            uploads[site] = unpack(send(site, "reducer", payload))
        with self.metrics.phase_timer("crypto", site="reducer"):
            sums_message = pack(self.he.add([uploads[site] for site in self.sites]))
            self.metrics.counter.stage = 1
            sums_delivered = {site: send("reducer", site, sums_message) for site in self.sites}
        partial_messages = {}
        self.metrics.counter.stage = 2
        for index, site in enumerate(self.sites):
            with self.metrics.phase_timer("crypto", site=site):
                partial = self.he.partial_decrypt(self._keys[site], unpack(sums_delivered[site]),
                                                  lead=index == 0)
                payload = pack(partial)
            partial_messages[site] = unpack(send(site, "reducer", payload))
        with self.metrics.phase_timer("crypto", site="reducer"):
            values = self.he.fuse([partial_messages[site] for site in self.sites], length)
        with self.metrics.phase_timer("compute", site="reducer"):
            result_message = np.asarray(values, dtype="<i8").tobytes()
            self.metrics.counter.stage = 3
            delivered = {site: send("reducer", site, result_message) for site in self.sites}
        result = None
        for site in self.sites:
            with self.metrics.phase_timer("compute", site=site):
                decoded = (np.frombuffer(delivered[site], dtype="<i8").reshape(shape) / scale)
                if result is None:
                    result = decoded
                elif not np.array_equal(decoded, result):
                    raise ValueError("Sites decoded different aggregates")
        return result
