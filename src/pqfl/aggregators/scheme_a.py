from copy import deepcopy
from itertools import combinations

import numpy as np

from pqfl.aggregators.base import Aggregator
from pqfl.kem.base import KEM
from pqfl.logging.metrics import Metrics, RoundStats
from pqfl.masking.core import Tensor, dequantize, pairwise_mask, quantize, validate_parameters


class SchemeA(Aggregator):
    """In-process functional simulator, not a distributed secure deployment.

    The coordinator sees all inputs and secrets. Real site isolation and the
    authenticated hybrid transport remain separate integration requirements.
    """

    def __init__(self, kem: KEM, *, scale_bits: int, modulus_bits: int, writer=None):
        validate_parameters(scale_bits, modulus_bits)
        self.kem, self.scale_bits, self.modulus_bits = kem, scale_bits, modulus_bits
        self.writer = writer
        self.sites = None
        self.used_rounds = set()

    def setup(self, site_ids: list[str]) -> None:
        if self.sites is not None:
            raise ValueError("Use a new instance for a new session")
        if len(site_ids) < 2 or any(not isinstance(s, str) or not s for s in site_ids):
            raise ValueError("Require at least two nonempty site IDs")
        if len(set(site_ids)) != len(site_ids):
            raise ValueError("Duplicate site IDs")
        self.sites = sorted(site_ids)
        self.metrics = Metrics(self.sites, self.writer)
        self.seeds = {}
        with self.metrics.phase_timer("crypto"):
            keys = {s: self.kem.keygen() for s in self.sites}
            for left, right in combinations(self.sites, 2):
                public, secret = keys[right]
                ciphertext, seed = self.kem.encaps(public)
                recovered = self.kem.decaps(secret, ciphertext)
                if recovered != seed:
                    raise ValueError("KEM shared secret mismatch")
                self.seeds[left, right] = seed
                self.metrics.counter.transfer(right, left, len(public))
                self.metrics.counter.transfer(left, right, len(ciphertext))
        self.setup_stats = deepcopy(self.metrics.counter.value)

    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor]) -> Tensor:
        if self.sites is None:
            raise ValueError("Call setup first")
        if type(round_idx) is not int or round_idx < 0 or round_idx in self.used_rounds:
            raise ValueError("Round must be nonnegative and never reused within a session")
        if set(updates) != set(self.sites):
            raise ValueError("All configured sites must participate; dropout unsupported")
        shape = np.asarray(updates[self.sites[0]]).shape
        if any(np.asarray(u).shape != shape for u in updates.values()):
            raise ValueError("Mismatched tensor shapes")
        self.used_rounds.add(round_idx)
        self.metrics = Metrics(self.sites, self.writer)
        self.metrics.round_idx = round_idx
        modulus = 2 ** self.modulus_bits
        with self.metrics.phase_timer("compute"):
            masked = {s: quantize(updates[s], self.scale_bits, self.modulus_bits)
                      for s in self.sites}
        with self.metrics.phase_timer("crypto"):
            for (left, right), seed in self.seeds.items():
                mask = pairwise_mask(seed, round_idx, left, right, shape, self.modulus_bits)
                masked[left] = (masked[left] + mask) % modulus
                masked[right] = (masked[right] - mask) % modulus
        with self.metrics.phase_timer("comm"):
            # Logical payload to an external reducer; no sockets or transport overhead.
            for s in self.sites:
                self.metrics.counter.value.bytes_sent[s] += masked[s].size * ((self.modulus_bits + 7) // 8)
        with self.metrics.phase_timer("compute"):
            total = np.zeros(shape, dtype=np.int64)
            for s in self.sites:
                total = (total + masked[s]) % modulus
            return dequantize(total, self.scale_bits, self.modulus_bits)

    def stats(self) -> RoundStats:
        if self.sites is None:
            raise ValueError("Call setup first")
        return deepcopy(self.metrics.counter.value)
