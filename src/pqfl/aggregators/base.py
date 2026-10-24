from abc import ABC, abstractmethod

from pqfl.masking.core import Tensor
from pqfl.logging.metrics import RoundStats


class Aggregator(ABC):
    """Common measured sum contract for schemes A, B and C.

    setup(site_ids) is called once and declares actors (including any reducer),
    participation_policy, rekey_policy and directed communication_edges per
    operation: setup, round, rekey. rekey either refreshes keys or is a declared
    no-op. Round indices are unique across the session, including after rekey.

    aggregate_round returns the SUM, preserving tensor shape, within the scheme's
    documented numerical tolerance. all_or_abort raises MissingParticipantError
    for missing/extra sites before output, traffic or round consumption.

    stats returns an isolated RoundStats snapshot for the last operation, with
    sent/received totals for every actor and a payload-free transfer ledger.
    Every actual protocol message crosses ByteCounter.send, which uses len of
    serialized bytes and records the same transfer. Include result broadcast;
    exclude transport framing. No synthetic full-mesh messages are required.

    Shared phases: auth, crypto, compute, comm. Use per-actor nonnested timers;
    comm is only for real transport. Metrics log byte deltas in schema_version=2.
    INSECURE_STUB implementations must declare that status explicitly.
    """

    actors: list[str]
    participation_policy: str
    rekey_policy: str
    communication_edges: dict[str, set[tuple[str, str]]]

    @abstractmethod
    def setup(self, site_ids: list[str]) -> None: ...

    @abstractmethod
    def rekey(self) -> None: ...

    @abstractmethod
    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor]) -> Tensor: ...

    @abstractmethod
    def stats(self) -> RoundStats: ...
