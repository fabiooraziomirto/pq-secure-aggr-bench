import numpy as np

from pqfl.aggregators.errors import AggregateOverflowError, MissingParticipantError
from pqfl.aggregators.scheme_a import SchemeA
from pqfl.logging.metrics import Metrics
from pqfl.masking.core import Tensor, dequantize, quantize
from pqfl.masking.sharing import open_share, seal_share, split_shares
from pqfl.masking.wire import deserialize_masked, serialize_masked


class SchemeC(SchemeA):
    """Additive secret sharing with the same authenticated PQ pairwise setup as A.

    Each site splits its quantized update into one additive share per site and
    sends every peer its share sealed (ChaCha20-Poly1305) under a key derived from
    the pairwise KEM secret. Each site then sends the sum of the shares it holds to
    the reducer, which adds the partial sums and broadcasts the result. The
    reducer never sees a single-site value. Colluding reducer plus up to n-2 sites
    learn only what the aggregate already reveals; reducer plus n-1 sites expose
    the remaining site, as for any sum.

    Same simulator caveats as A: in-process, the coordinator holds all secrets.
    """

    def setup(self, site_ids: list[str]) -> None:
        super().setup(site_ids)
        site_pairs = {(left, right) for left in self.sites for right in self.sites if left != right}
        self.communication_edges["round"] = site_pairs | self.communication_edges["round"]

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
        modulus = 2 ** self.modulus_bits
        # Simulator-only overflow check: a real reducer cannot see the plaintext sum.
        signed_total = np.zeros(shape, dtype=object)
        delivered = {site: {} for site in self.sites}
        own = {}
        self.metrics.counter.stage = 0
        for site in self.sites:
            with self.metrics.phase_timer("compute", site=site):
                values = np.asarray(updates[site], dtype=np.float64)
                quantized = quantize(values, self.scale_bits, self.modulus_bits)
                signed = np.where(quantized >= modulus // 2, quantized - modulus, quantized)
                signed_total += signed.astype(object)
            with self.metrics.phase_timer("crypto", site=site):
                shares = dict(zip(self.sites, split_shares(quantized, len(self.sites), self.modulus_bits)))
                own[site] = shares.pop(site)
                sealed = {peer: seal_share(self._site_seeds[site][tuple(sorted((site, peer)))],
                                           self.epoch, round_idx, site, peer,
                                           serialize_masked(share, self.modulus_bits))
                          for peer, share in shares.items()}
            for peer, message in sealed.items():
                delivered[peer][site] = self.metrics.counter.send(site, peer, message)
        partial_messages = {}
        self.metrics.counter.stage = 1
        for site in self.sites:
            with self.metrics.phase_timer("crypto", site=site):
                partial = own[site]
                for sender, message in delivered[site].items():
                    seed = self._site_seeds[site][tuple(sorted((site, sender)))]
                    plain = open_share(seed, self.epoch, round_idx, sender, site, message)
                    partial = (partial + deserialize_masked(plain, shape, self.modulus_bits)) % modulus
            with self.metrics.phase_timer("compute", site=site):
                payload = serialize_masked(partial, self.modulus_bits)
                partial_messages[site] = self.metrics.counter.send(site, "reducer", payload)
        with self.metrics.phase_timer("compute", site="reducer"):
            if np.any(signed_total < -modulus // 2) or np.any(signed_total >= modulus // 2):
                raise AggregateOverflowError("Signed aggregate exceeds the selected modular range")
            total = np.zeros(shape, dtype=np.int64)
            for message in partial_messages.values():
                total = (total + deserialize_masked(message, shape, self.modulus_bits)) % modulus
            result_message = serialize_masked(total, self.modulus_bits)
            self.metrics.counter.stage = 2
            broadcast = {site: self.metrics.counter.send("reducer", site, result_message)
                         for site in self.sites}
        result = None
        for site in self.sites:
            with self.metrics.phase_timer("compute", site=site):
                decoded = dequantize(deserialize_masked(broadcast[site], shape, self.modulus_bits),
                                     self.scale_bits, self.modulus_bits)
                if result is None:
                    result = decoded
                elif not np.array_equal(decoded, result):
                    raise ValueError("Sites decoded different aggregates")
        return result
