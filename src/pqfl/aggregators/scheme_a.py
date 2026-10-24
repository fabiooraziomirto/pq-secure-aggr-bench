from copy import deepcopy
from itertools import combinations

import numpy as np

from pqfl.aggregators.base import Aggregator
from pqfl.kem.base import KEM
from pqfl.logging.metrics import Metrics, RoundStats
from pqfl.masking.core import Tensor, dequantize, pairwise_mask, quantize, validate_parameters
from pqfl.masking.wire import deserialize_masked, serialize_masked


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
        self.epoch = 0

    def setup(self, site_ids: list[str]) -> None:
        if self.sites is not None:
            raise ValueError("Use a new instance for a new session")
        if len(site_ids) < 2 or any(not isinstance(s, str) or not s for s in site_ids):
            raise ValueError("Require at least two nonempty site IDs")
        if len(set(site_ids)) != len(site_ids) or "reducer" in site_ids:
            raise ValueError("Duplicate or reserved site ID")
        self.sites = sorted(site_ids)
        self.actors = [*self.sites, "reducer"]
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        self.seeds = {}
        self._site_seeds = {site: {} for site in self.sites}
        keys = {}
        # The smallest site ID never receives encapsulations: only n-1 keypairs.
        for site in self.sites[1:]:
            with self.metrics.phase_timer("crypto", site=site):
                keys[site] = self.kem.keygen()
        for left, right in combinations(self.sites, 2):
            public, secret = keys[right]
            with self.metrics.phase_timer("crypto", site=left):
                self.metrics.counter.transfer(right, left, len(public))
                ciphertext, seed = self.kem.encaps(public)
                self.metrics.counter.transfer(left, right, len(ciphertext))
            with self.metrics.phase_timer("crypto", site=right):
                recovered = self.kem.decaps(secret, ciphertext)
                if recovered != seed:
                    raise ValueError("KEM shared secret mismatch")
            self.seeds[left, right] = seed
            self._site_seeds[left][left, right] = seed
            self._site_seeds[right][left, right] = recovered
        self.setup_stats = deepcopy(self.metrics.counter.value)

    def _masked_site(self, site: str, round_idx: int, quantized: Tensor) -> Tensor:
        """Return this site's masked vector from its own quantized input."""
        modulus = 2 ** self.modulus_bits
        masked = quantized.copy()
        with self.metrics.phase_timer("crypto", site=site):
            for (left, right), seed in self._site_seeds[site].items():
                mask = pairwise_mask(seed, round_idx, left, right, quantized.shape, self.modulus_bits)
                masked = (masked + mask if site == left else masked - mask) % modulus
        return masked

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
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        self.metrics.round_idx = round_idx
        modulus = 2 ** self.modulus_bits
        messages = {}
        for site in self.sites:
            with self.metrics.phase_timer("compute", site=site):
                quantized = quantize(updates[site], self.scale_bits, self.modulus_bits)
            masked = self._masked_site(site, round_idx, quantized)
            with self.metrics.phase_timer("compute", site=site):
                messages[site] = serialize_masked(masked, self.modulus_bits)
                self.metrics.counter.transfer(site, "reducer", len(messages[site]))
        with self.metrics.phase_timer("compute", site="reducer"):
            total = np.zeros(shape, dtype=np.int64)
            for message in messages.values():
                total = (total + deserialize_masked(message, shape, self.modulus_bits)) % modulus
            result_message = serialize_masked(total, self.modulus_bits)
            for site in self.sites:
                self.metrics.counter.transfer("reducer", site, len(result_message))
        result = None
        for site in self.sites:
            with self.metrics.phase_timer("compute", site=site):
                received = deserialize_masked(result_message, shape, self.modulus_bits)
                decoded = dequantize(received, self.scale_bits, self.modulus_bits)
                if result is None:
                    result = decoded
        return result

    def stats(self) -> RoundStats:
        if self.sites is None:
            raise ValueError("Call setup first")
        return deepcopy(self.metrics.counter.value)
