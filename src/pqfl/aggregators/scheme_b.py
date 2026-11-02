from copy import deepcopy
import math
from numbers import Real

import numpy as np

from pqfl.aggregators.base import Aggregator
from pqfl.aggregators.errors import MissingParticipantError
from pqfl.auth import Authenticator, AuthProof, PinnedFingerprintAuth
from pqfl.he.ckks import ThresholdCKKS, pack, unpack
from pqfl.logging.metrics import Metrics, RoundStats
from pqfl.masking.core import Tensor


class SchemeB(Aggregator):
    """Threshold CKKS: n-of-n joint key, homomorphic sum at the reducer.

    The first site (smallest ID) generates a key pair; each later site derives a
    joint key from the previous public key (sequential chain), and the last site
    broadcasts the final public key. Public keys travel with authentication
    proofs. Rounds: sites upload ciphertexts, the reducer adds them and
    broadcasts the sum, sites return partial decryptions, the reducer fuses them
    and broadcasts the plaintext sum. No pairwise KEM or secure channel is needed.

    In-process simulator, semi-honest, n-of-n (no dropout). Partial decryptions
    are not noise-flooded (see pqfl.he.ckks), so threshold-decryption security
    still needs expert review. CKKS is approximate: error is declared by tests.
    """

    participation_policy = "all_or_abort"
    rekey_policy = "refresh_joint_key"
    security_status = "FUNCTIONAL_SIMULATOR"

    def __init__(self, *, clip_bound: float, ring_dim: int = 1 << 14, scaling_mod_size: int = 50,
                 writer=None, authenticator: Authenticator | None = None):
        if isinstance(clip_bound, bool) or not isinstance(clip_bound, Real):
            raise ValueError("clip_bound must be a finite nonnegative number")
        if not math.isfinite(clip_bound) or clip_bound < 0:
            raise ValueError("clip_bound must be finite and nonnegative")
        self.clip_bound = clip_bound
        self.he = ThresholdCKKS(ring_dim=ring_dim, scaling_mod_size=scaling_mod_size)
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
        if len(site_ids) < 2 or any(not isinstance(s, str) or not s for s in site_ids):
            raise ValueError("Require at least two nonempty site IDs")
        if len(set(site_ids)) != len(site_ids) or "reducer" in site_ids:
            raise ValueError("Duplicate or reserved site ID")
        self.sites = sorted(site_ids)
        self.actors = [*self.sites, "reducer"]
        chain = {(a, b) for a, b in zip(self.sites, self.sites[1:])}
        final = {(self.sites[-1], site) for site in self.sites[:-1]}
        round_edges = {(site, "reducer") for site in self.sites} | {("reducer", site) for site in self.sites}
        self.communication_edges = {"setup": chain | final, "rekey": chain | final, "round": round_edges}
        self._exchange_keys()
        self.setup_stats = deepcopy(self.metrics.counter.value)

    def rekey(self) -> None:
        if self.sites is None:
            raise ValueError("Call setup first")
        self.epoch += 1
        self._exchange_keys()
        self.rekey_stats = deepcopy(self.metrics.counter.value)

    def _send_authenticated(self, sender: str, receiver: str, public: bytes, proof: AuthProof) -> bytes:
        send = self.metrics.counter.send
        received = send(sender, receiver, public)
        identity = send(sender, receiver, proof.identity_public) if proof.identity_public else b""
        signature = send(sender, receiver, proof.signature) if proof.signature else b""
        self.authenticator.verify(receiver, sender, self.epoch, received, AuthProof(identity, signature))
        return received

    def _exchange_keys(self) -> None:
        self._ready = False
        self._keys = {}
        self.metrics = Metrics(self.actors, self.writer, epoch=self.epoch)
        previous = None
        for index, site in enumerate(self.sites):
            with self.metrics.phase_timer("crypto", site=site):
                self._keys[site] = (self.he.keygen_first() if index == 0
                                    else self.he.keygen_join(previous))
                public = self.he.serialize_public(self._keys[site])
            if index + 1 < len(self.sites):
                self.metrics.counter.stage = index
                with self.metrics.phase_timer("auth", site=site):
                    proof = self.authenticator.issue(site, self.epoch, public)
                with self.metrics.phase_timer("auth", site=self.sites[index + 1]):
                    previous = self._send_authenticated(site, self.sites[index + 1], public, proof)
        with self.metrics.phase_timer("auth", site=self.sites[-1]):
            proof = self.authenticator.issue(self.sites[-1], self.epoch, public)
        self._joint = {self.sites[-1]: public}
        self.metrics.counter.stage = len(self.sites) - 1
        for site in self.sites[:-1]:
            with self.metrics.phase_timer("auth", site=site):
                self._joint[site] = self._send_authenticated(self.sites[-1], site, public, proof)
        self._ready = True

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
        uploads = {}
        self.metrics.counter.stage = 0
        for site in self.sites:
            with self.metrics.phase_timer("crypto", site=site):
                cts = self.he.encrypt(self._joint[site], np.asarray(updates[site], dtype=np.float64).ravel())
                payload = pack(cts)
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
            result_message = np.asarray(values, dtype="<f8").tobytes()
            self.metrics.counter.stage = 3
            delivered = {site: send("reducer", site, result_message) for site in self.sites}
        result = None
        for site in self.sites:
            with self.metrics.phase_timer("compute", site=site):
                decoded = np.frombuffer(delivered[site], dtype="<f8").reshape(shape).copy()
                if result is None:
                    result = decoded
                elif not np.array_equal(decoded, result):
                    raise ValueError("Sites decoded different aggregates")
        return result

    def stats(self) -> RoundStats:
        if self.sites is None:
            raise ValueError("Call setup first")
        return deepcopy(self.metrics.counter.value)
