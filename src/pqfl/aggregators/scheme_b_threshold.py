import json
from types import SimpleNamespace

import numpy as np

from pqfl.aggregators.errors import MissingParticipantError
from pqfl.aggregators.scheme_a import SchemeA
from pqfl.aggregators.scheme_a_dropout import InsufficientParticipantsError
from pqfl.aggregators.scheme_b_bgv import SchemeBGV
from pqfl.he.ckks import pack, unpack
from pqfl.logging.metrics import Metrics
from pqfl.masking.core import Tensor
from pqfl.masking.sharing import open_share, seal_share


class SchemeBGVThreshold(SchemeBGV):
    """B-BGV with t-of-n recovery of a crashed site, through OpenFHE Shamir key sharing.

    Setup adds a pairwise KEM handshake (as in A) and each site's Shamir shares of its secret
    key for every peer. If sites crash before uploading, the lowest-ID survivor (the leader)
    collects t shares per dropped key from other survivors over the sealed pairwise
    channels, reconstructs the dropped keys, and computes their partial decryptions; the
    reducer fuses all n partials into the sum of the survivors. Reconstructed keys stay at
    the leader, so the dropped sites remain absent in later rounds at the cost of one
    extra partial decryption each.

    Limits, stated openly. The binding cannot serialize key shares (DCRTPoly is opaque), so
    each share travels as a size-faithful placeholder: real AEAD-sealed bytes of the exact
    serialized key size, with the share object handed over by reference in-process. Byte and
    time costs are faithful; the share contents are simulated. Semi-honest and crash-stop:
    the leader learns the dropped sites' secret keys. Needs BGV (threshold is not verified
    for CKKS here).
    """

    participation_policy = "quorum"
    rekey_policy = "refresh_joint_key"

    def __init__(self, kem, *, clip_bound: float, threshold: int, **kwargs):
        super().__init__(clip_bound=clip_bound, **kwargs)
        self.kem, self.key_provider = kem, None
        if type(threshold) is not int or threshold < 2:
            raise ValueError("threshold must be an integer of at least 2")
        self.threshold = threshold
        self.absent = set()
        self._recovered = {}

    def setup(self, site_ids: list[str]) -> None:
        if len(site_ids) < self.threshold:
            raise ValueError("threshold cannot exceed the number of sites")
        super().setup(site_ids)

    def rekey(self) -> None:
        if self.absent:
            raise ValueError("Rekey with absent sites is not supported here")
        super().rekey()

    def _exchange_keys(self) -> None:
        SchemeA._exchange_keys(self)          # pairwise KEM + authentication; resets self.metrics
        self._ready = False
        self._keys = {}
        self._recovered = {}
        n = len(self.sites)
        send = self.metrics.counter.send
        previous, public = None, None
        for index, site in enumerate(self.sites):
            with self.metrics.phase_timer("crypto", site=site):
                self._keys[site] = (self.he.keygen_first() if index == 0 else self.he.keygen_join(previous))
                public = self.he.serialize_public(self._keys[site])
            if index + 1 < n:
                self.metrics.counter.stage = 3 + index
                with self.metrics.phase_timer("auth", site=site):
                    proof = self.authenticator.issue(site, self.epoch, public)
                with self.metrics.phase_timer("auth", site=self.sites[index + 1]):
                    previous = self._send_authenticated(site, self.sites[index + 1], public, proof)
        with self.metrics.phase_timer("auth", site=self.sites[-1]):
            proof = self.authenticator.issue(self.sites[-1], self.epoch, public)
        self._joint = {self.sites[-1]: public}
        self.metrics.counter.stage = 3 + n - 1
        for site in self.sites[:-1]:
            with self.metrics.phase_timer("auth", site=site):
                self._joint[site] = self._send_authenticated(self.sites[-1], site, public, proof)
        # Shamir shares of every site's secret key, sealed per pair (size-faithful placeholders).
        of = self.he.of
        self._share_bytes = len(of.Serialize(self._keys[self.sites[0]].secretKey, of.BINARY))
        self._held = {site: {} for site in self.sites}
        self.metrics.counter.stage = 3 + n
        for index, site in enumerate(self.sites):
            with self.metrics.phase_timer("crypto", site=site):
                shares = self.he.cc.ShareKeys(self._keys[site].secretKey, n, self.threshold, index + 1, "shamir")
            for peer_index, peer in enumerate(self.sites):
                if peer == site:
                    continue
                self._held[peer][site] = self._transfer_share(site, peer, shares[peer_index + 1], round_tag=-1)
        self._ready = True

    def _transfer_share(self, sender, receiver, share, *, round_tag):
        """Send a sealed placeholder of the true share size; hand over the object by reference."""
        seed = self._site_seeds[sender][tuple(sorted((sender, receiver)))]
        with self.metrics.phase_timer("crypto", site=sender):
            sealed = seal_share(seed, self.epoch, round_tag, sender, receiver, bytes(self._share_bytes))
        delivered = self.metrics.counter.send(sender, receiver, sealed)
        with self.metrics.phase_timer("crypto", site=receiver):
            open_share(seed, self.epoch, round_tag, sender, receiver, delivered)
        return share

    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor], *, dropped=frozenset()) -> Tensor:
        dropped = frozenset(dropped)
        active = [site for site in self.sites if site not in self.absent]
        if not self._ready:
            raise ValueError("A successful setup or rekey is required")
        if type(round_idx) is not int or round_idx < 0 or round_idx in self.used_rounds:
            raise ValueError("Round must be nonnegative and never reused within a session")
        if set(updates) != set(active):
            raise MissingParticipantError("updates must cover exactly the active sites")
        if not dropped <= set(active):
            raise ValueError("Dropped sites must be active sites")
        survivors = [site for site in active if site not in dropped]
        if len(survivors) < self.threshold:
            raise InsufficientParticipantsError("Fewer survivors than the decryption threshold")
        shape = np.asarray(updates[active[0]]).shape
        if any(np.asarray(u).shape != shape for u in updates.values()):
            raise ValueError("Mismatched tensor shapes")
        for site in survivors:
            values = np.asarray(updates[site], dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(np.abs(values) > self.clip_bound):
                raise ValueError("Update exceeds clip_bound or is not finite")
        self.used_rounds.add(round_idx)
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        self.metrics.round_idx = round_idx
        send, counter = self.metrics.counter.send, self.metrics.counter
        length, scale = int(np.prod(shape)), 2 ** self.scale_bits
        uploads = {}
        counter.stage = 0
        for site in survivors:
            with self.metrics.phase_timer("compute", site=site):
                integers = np.rint(np.asarray(updates[site], dtype=np.float64).ravel() * scale).astype(np.int64).tolist()
            with self.metrics.phase_timer("crypto", site=site):
                payload = pack(self.he.encrypt(self._joint[site], integers))
            uploads[site] = unpack(send(site, "reducer", payload))
        with self.metrics.phase_timer("crypto", site="reducer"):
            sums_message = pack(self.he.add([uploads[site] for site in survivors]))
            counter.stage = 1
            sums = {site: send("reducer", site, sums_message) for site in survivors}
        counter.stage = 2
        partials = {}
        for site in survivors:
            with self.metrics.phase_timer("crypto", site=site):
                payload = pack(self.he.partial_decrypt(self._keys[site], unpack(sums[site]),
                                                       lead=site == self.sites[0]))
            partials[site] = unpack(send(site, "reducer", payload))
        absent_now = sorted(self.absent | dropped)
        if absent_now:
            leader, stage = survivors[0], 3
            fresh = [d for d in absent_now if d not in self._recovered]
            if fresh:
                counter.stage = stage
                with self.metrics.phase_timer("compute", site="reducer"):
                    request = json.dumps(fresh).encode()
                    for site in survivors:
                        send("reducer", site, request)
                stage += 1
                counter.stage = stage
                for d in fresh:
                    held = {leader: self._held[leader][d]}
                    for helper in survivors[1:self.threshold]:
                        held[helper] = self._transfer_share(helper, leader, self._held[helper][d], round_tag=-2)
                    with self.metrics.phase_timer("crypto", site=leader):
                        recovered = self.he.cc.KeyGen().secretKey
                        shares = {self.sites.index(h) + 1: s for h, s in held.items()}
                        self.he.cc.RecoverSharedKey(recovered, shares, len(self.sites), self.threshold, "shamir")
                    self._recovered[d] = SimpleNamespace(secretKey=recovered)
                stage += 1
            counter.stage = stage
            with self.metrics.phase_timer("crypto", site=leader):
                extra = {d: self.he.partial_decrypt(self._recovered[d], unpack(sums[leader]),
                                                    lead=d == self.sites[0]) for d in absent_now}
                payload = pack([part for d in absent_now for part in extra[d]])
            flat = unpack(send(leader, "reducer", payload))
            chunks = len(unpack(sums_message))      # ciphertexts per partial decryption
            for k, d in enumerate(absent_now):
                partials[d] = flat[k * chunks:(k + 1) * chunks]
            stage += 1
        else:
            stage = 3
        with self.metrics.phase_timer("crypto", site="reducer"):
            values = self.he.fuse([partials[site] for site in self.sites], length)
        counter.stage = stage
        with self.metrics.phase_timer("compute", site="reducer"):
            result_message = np.asarray(values, dtype="<i8").tobytes()
            delivered = {site: send("reducer", site, result_message) for site in survivors}
        result = None
        for site in survivors:
            with self.metrics.phase_timer("compute", site=site):
                decoded = np.frombuffer(delivered[site], dtype="<i8").reshape(shape) / scale
                if result is None:
                    result = decoded
                elif not np.array_equal(decoded, result):
                    raise ValueError("Sites decoded different aggregates")
        self.absent |= dropped
        return result
