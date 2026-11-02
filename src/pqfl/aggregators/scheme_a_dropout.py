import json

import numpy as np

from pqfl.aggregators.errors import AggregateOverflowError, MissingParticipantError
from pqfl.aggregators.scheme_a import SchemeA
from pqfl.logging.metrics import Metrics
from pqfl.masking.core import Tensor, dequantize, mask_from_key, mask_key, quantize
from pqfl.masking.wire import deserialize_masked, serialize_masked


class InsufficientParticipantsError(ValueError):
    """Too few survivors for the quorum; raised before any traffic or round consumption."""


class SchemeADropout(SchemeA):
    """A with recovery from a site that crashes before uploading its masked vector.

    Survivors already masked against every peer, so the sum holds uncancelled masks of the
    dropped sites. The reducer asks each survivor for this round's mask keys of the pairs it
    shares with the dropped sites, recomputes those masks and removes them; the result is the
    survivors' sum, and the dropped sites leave the session (their pair seeds are deleted).
    Only per-round keys are revealed, never the epoch seeds: earlier uploads of the dropped
    sites in the same epoch stay masked, also for whoever records the (unencrypted) reveal.

    Threat model: crash-stop dropout before upload and a reducer that does not lie about it.
    A reducer that falsely declares a live site dropped could collect all of its round keys
    and unmask it once its vector arrives; defending against that needs the double masking
    of Bonawitz et al. (2017), which is not implemented. Survivors must answer the request.
    """

    participation_policy = "quorum"

    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor], *,
                        dropped=frozenset(), min_survivors: int = 2) -> Tensor:
        dropped = frozenset(dropped)
        if not dropped:
            return super().aggregate_round(round_idx, updates)
        if not self._ready:
            raise ValueError("A successful setup or rekey is required")
        if type(round_idx) is not int or round_idx < 0 or round_idx in self.used_rounds:
            raise ValueError("Round must be nonnegative and never reused within a session")
        if set(updates) != set(self.sites):
            raise MissingParticipantError("updates must cover exactly the active sites")
        if not dropped <= set(self.sites):
            raise ValueError("Dropped sites must be active sites")
        survivors = [site for site in self.sites if site not in dropped]
        if len(survivors) < max(2, min_survivors):
            raise InsufficientParticipantsError("Fewer survivors than the quorum")
        shape = np.asarray(updates[self.sites[0]]).shape
        if any(np.asarray(u).shape != shape for u in updates.values()):
            raise ValueError("Mismatched tensor shapes")
        for site in survivors:
            values = np.asarray(updates[site], dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(np.abs(values) > self.clip_bound):
                raise ValueError("Update exceeds clip_bound or is not finite")
        self.used_rounds.add(round_idx)
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        self.metrics.round_idx = round_idx
        send = self.metrics.counter.send
        modulus = 2 ** self.modulus_bits
        signed_total = np.zeros(shape, dtype=object)
        messages = {}
        self.metrics.counter.stage = 0
        for site in survivors:
            with self.metrics.phase_timer("compute", site=site):
                values = np.asarray(updates[site], dtype=np.float64)
                quantized = quantize(values, self.scale_bits, self.modulus_bits)
                signed = np.where(quantized >= modulus // 2, quantized - modulus, quantized)
                signed_total += signed.astype(object)
            masked = self._masked_site(site, round_idx, quantized)   # masks against every peer
            with self.metrics.phase_timer("compute", site=site):
                messages[site] = send(site, "reducer", serialize_masked(masked, self.modulus_bits))
        # Detection (a timeout at the reducer) is a deployment parameter, not simulated time.
        dropped_sorted = sorted(dropped)
        self.metrics.counter.stage = 1
        with self.metrics.phase_timer("compute", site="reducer"):
            request = json.dumps(dropped_sorted).encode()
            requests = {site: send("reducer", site, request) for site in survivors}
        self.metrics.counter.stage = 2
        reveals = {}
        for site in survivors:
            with self.metrics.phase_timer("crypto", site=site):
                wanted = json.loads(requests[site])
                payload = b"".join(mask_key(self._site_seeds[site][pair], round_idx, *pair, epoch=self.epoch)
                                   for pair in (tuple(sorted((site, d))) for d in wanted))
            reveals[site] = send(site, "reducer", payload)
        with self.metrics.phase_timer("crypto", site="reducer"):
            total = np.zeros(shape, dtype=np.int64)
            for message in messages.values():
                total = (total + deserialize_masked(message, shape, self.modulus_bits)) % modulus
            for site in survivors:
                width = len(reveals[site]) // len(dropped_sorted)
                for k, d in enumerate(dropped_sorted):
                    left, right = sorted((site, d))
                    mask = mask_from_key(reveals[site][k * width:(k + 1) * width], shape, self.modulus_bits)
                    # A survivor added +mask when it is the left site and -mask otherwise.
                    total = (total - mask) % modulus if site == left else (total + mask) % modulus
            if np.any(signed_total < -modulus // 2) or np.any(signed_total >= modulus // 2):
                raise AggregateOverflowError("Signed aggregate exceeds the selected modular range")
            result_message = serialize_masked(total, self.modulus_bits)
        self.metrics.counter.stage = 3
        with self.metrics.phase_timer("compute", site="reducer"):
            delivered = {site: send("reducer", site, result_message) for site in survivors}
        result = None
        for site in survivors:
            with self.metrics.phase_timer("compute", site=site):
                decoded = dequantize(deserialize_masked(delivered[site], shape, self.modulus_bits),
                                     self.scale_bits, self.modulus_bits)
                if result is None:
                    result = decoded
                elif not np.array_equal(decoded, result):
                    raise ValueError("Sites decoded different aggregates")
        self._remove_sites(dropped)
        return result

    def _remove_sites(self, dropped) -> None:
        """Drop sites from the session so later rounds mask over survivors only."""
        self.sites = [site for site in self.sites if site not in dropped]
        self.actors = [*self.sites, "reducer"]
        for site in dropped:
            self._site_seeds.pop(site, None)
        for seeds in self._site_seeds.values():
            for pair in [p for p in seeds if p[0] in dropped or p[1] in dropped]:
                del seeds[pair]
        self.seeds = {pair: seed for pair, seed in self.seeds.items()
                      if pair[0] not in dropped and pair[1] not in dropped}
        keep = lambda edges: {(a, b) for a, b in edges if a not in dropped and b not in dropped}
        self.communication_edges = {phase: keep(edges) for phase, edges in self.communication_edges.items()}
