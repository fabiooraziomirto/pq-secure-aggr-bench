from abc import ABC, abstractmethod
from pqfl.masking.core import Tensor
from pqfl.logging.metrics import RoundStats


class Aggregator(ABC):
    """Same API for schemes A, B, C so costs are measured identically."""

    @abstractmethod
    def setup(self, site_ids: list[str]) -> None: ...

    @abstractmethod
    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor]) -> Tensor: ...

    @abstractmethod
    def stats(self) -> RoundStats: ...
