from copy import deepcopy
from itertools import combinations
import hmac
import math
from numbers import Real
from typing import Callable

from pqfl.auth import Authenticator, AuthProof, PinnedFingerprintAuth

import numpy as np

from pqfl.aggregators.base import Aggregator
from pqfl.aggregators.errors import AggregateOverflowError, KeyConfirmationError, MissingParticipantError
from pqfl.masking import confirmation
from pqfl.kem.base import KEM
from pqfl.logging.metrics import Metrics, RoundStats
from pqfl.masking.core import Tensor, dequantize, pairwise_mask, quantize, validate_parameters
from pqfl.masking.wire import deserialize_masked, serialize_masked


class SchemeA(Aggregator):
    """In-process functional simulator, not a distributed secure deployment.

    The coordinator sees all inputs and secrets. Real site isolation and the
    authenticated hybrid transport remain separate integration requirements.
    """

    participation_policy = "all_or_abort"
    rekey_policy = "refresh_all_pairs"
    security_status = "FUNCTIONAL_SIMULATOR"

    def __init__(self, kem: KEM, *, scale_bits: int, modulus_bits: int, clip_bound: float,
                 writer=None, authenticator: Authenticator | None = None,
                 key_provider: Callable[[str, int], tuple[bytes, bytes]] | None = None):
        validate_parameters(scale_bits, modulus_bits)
        self.kem, self.scale_bits, self.modulus_bits = kem, scale_bits, modulus_bits
        self.authenticator = authenticator if authenticator is not None else PinnedFingerprintAuth()
        self.key_provider = key_provider
        self.clip_bound = clip_bound
        if isinstance(clip_bound, bool) or not isinstance(clip_bound, Real):
            raise ValueError("clip_bound must be a finite nonnegative number")
        if not math.isfinite(clip_bound) or clip_bound < 0:
            raise ValueError("clip_bound must be finite and nonnegative")
        # At construction n is unknown. Reject even the minimum n=2 if unsafe;
        # setup checks the actual participant count before generating keys.
        self._validate_clip_count(2)
        self.writer = writer
        self.sites = None
        self.actors = []
        self.communication_edges = {}
        self.used_rounds = set()
        self.epoch = 0
        self._ready = False

    def _validate_clip_count(self, count: int) -> None:
        scaled_bound = self.clip_bound * (2 ** self.scale_bits)
        limit = 2 ** (self.modulus_bits - 1)
        if count * scaled_bound >= limit:
            raise ValueError("clip_bound cannot guarantee a signed aggregate without overflow")
        # Include rounding in the conservative production-side bound as well.
        if count * int(np.rint(scaled_bound)) >= limit:
            raise ValueError("Rounded clip_bound cannot guarantee a signed aggregate")

    def setup(self, site_ids: list[str]) -> None:
        if self.sites is not None:
            raise ValueError("Use a new instance for a new session")
        if len(site_ids) < 2 or any(not isinstance(s, str) or not s for s in site_ids):
            raise ValueError("Require at least two nonempty site IDs")
        if len(set(site_ids)) != len(site_ids) or "reducer" in site_ids:
            raise ValueError("Duplicate or reserved site ID")
        self._validate_clip_count(len(site_ids))
        self.sites = sorted(site_ids)
        self.actors = [*self.sites, "reducer"]
        pair_edges = {(left, right) for left in self.sites for right in self.sites if left != right}
        round_edges = {(site, "reducer") for site in self.sites} | {("reducer", site) for site in self.sites}
        self.communication_edges = {"setup": pair_edges, "rekey": set(pair_edges), "round": round_edges}
        self._exchange_keys()
        self.setup_stats = deepcopy(self.metrics.counter.value)

    def rekey(self) -> None:
        if self.sites is None:
            raise ValueError("Call setup first")
        self.epoch += 1
        self._exchange_keys()
        self.rekey_stats = deepcopy(self.metrics.counter.value)

    def _exchange_keys(self) -> None:
        # Invalidate the previous epoch before attempting a replacement. A failed
        # confirmation must never leave old keys usable under the new epoch.
        self._ready = False
        self.seeds = {}
        self._site_seeds = {}
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        seeds = {}
        site_seeds = {site: {} for site in self.sites}
        keys = {}
        # The smallest site ID never receives encapsulations: only n-1 keypairs.
        for site in self.sites[1:]:
            with self.metrics.phase_timer("crypto", site=site):
                keys[site] = (self.key_provider(site, self.epoch) if self.key_provider
                              else self.kem.keygen())
        proofs = {}
        for site in self.sites[1:]:
            with self.metrics.phase_timer("auth", site=site):
                proofs[site] = self.authenticator.issue(site, self.epoch, keys[site][0])
        for left, right in combinations(self.sites, 2):
            public, secret = keys[right]
            with self.metrics.phase_timer("auth", site=left):
                proof = proofs[right]
                public = self.metrics.counter.send(right, left, public)
                identity = (self.metrics.counter.send(right, left, proof.identity_public)
                            if proof.identity_public else b"")
                signature = (self.metrics.counter.send(right, left, proof.signature)
                             if proof.signature else b"")
                received_proof = AuthProof(identity, signature)
                self.authenticator.verify(left, right, self.epoch, public, received_proof)
            with self.metrics.phase_timer("crypto", site=left):
                ciphertext, seed = self.kem.encaps(public)
                ciphertext = self.metrics.counter.send(left, right, ciphertext)
                left_key = confirmation.confirmation_key(seed, self.epoch, left, right, public, ciphertext)
                left_tag = confirmation.confirmation_tag(left_key, "left")
                left_tag = self.metrics.counter.send(left, right, left_tag)
            with self.metrics.phase_timer("crypto", site=right):
                recovered = self.kem.decaps(secret, ciphertext)
                right_key = confirmation.confirmation_key(recovered, self.epoch, left, right,
                                                          public, ciphertext)
                expected_left = confirmation.confirmation_tag(right_key, "left")
                if not hmac.compare_digest(left_tag, expected_left):
                    raise KeyConfirmationError("Peer key confirmation failed")
                right_tag = confirmation.confirmation_tag(right_key, "right")
                right_tag = self.metrics.counter.send(right, left, right_tag)
            with self.metrics.phase_timer("crypto", site=left):
                expected_right = confirmation.confirmation_tag(left_key, "right")
                if not hmac.compare_digest(right_tag, expected_right):
                    raise KeyConfirmationError("Peer key confirmation failed")
            seeds[left, right] = seed
            site_seeds[left][left, right] = seed
            site_seeds[right][left, right] = recovered
        self.seeds = seeds
        self._site_seeds = site_seeds
        self._ready = True

    def _masked_site(self, site: str, round_idx: int, quantized: Tensor) -> Tensor:
        """Return this site's masked vector from its own quantized input."""
        modulus = 2 ** self.modulus_bits
        masked = quantized.copy()
        with self.metrics.phase_timer("crypto", site=site):
            for (left, right), seed in self._site_seeds[site].items():
                mask = pairwise_mask(seed, round_idx, left, right, quantized.shape, self.modulus_bits,
                                     epoch=self.epoch)
                masked = (masked + mask if site == left else masked - mask) % modulus
        return masked

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
        self.used_rounds.add(round_idx)
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        self.metrics.round_idx = round_idx
        modulus = 2 ** self.modulus_bits
        messages = {}
        # Simulator-only: a real reducer cannot inspect the plaintext sum. Python
        # integers avoid overflow in the safety check even for many participants.
        signed_total = np.zeros(shape, dtype=object)
        for site in self.sites:
            with self.metrics.phase_timer("compute", site=site):
                values = np.asarray(updates[site], dtype=np.float64)
                if self.clip_bound is not None and np.any(np.abs(values) > self.clip_bound):
                    raise ValueError("Update exceeds clip_bound")
                quantized = quantize(values, self.scale_bits, self.modulus_bits)
                signed = np.where(quantized >= modulus // 2, quantized - modulus, quantized)
                signed_total += signed.astype(object)
            masked = self._masked_site(site, round_idx, quantized)
            with self.metrics.phase_timer("compute", site=site):
                payload = serialize_masked(masked, self.modulus_bits)
                messages[site] = self.metrics.counter.send(site, "reducer", payload)
        with self.metrics.phase_timer("compute", site="reducer"):
            if np.any(signed_total < -modulus // 2) or np.any(signed_total >= modulus // 2):
                raise AggregateOverflowError("Signed aggregate exceeds the selected modular range")
            total = np.zeros(shape, dtype=np.int64)
            for message in messages.values():
                total = (total + deserialize_masked(message, shape, self.modulus_bits)) % modulus
            result_message = serialize_masked(total, self.modulus_bits)
            delivered = {site: self.metrics.counter.send("reducer", site, result_message)
                         for site in self.sites}
        result = None
        for site in self.sites:
            with self.metrics.phase_timer("compute", site=site):
                received = deserialize_masked(delivered[site], shape, self.modulus_bits)
                decoded = dequantize(received, self.scale_bits, self.modulus_bits)
                if result is None:
                    result = decoded
        return result

    def stats(self) -> RoundStats:
        if self.sites is None:
            raise ValueError("Call setup first")
        return deepcopy(self.metrics.counter.value)
