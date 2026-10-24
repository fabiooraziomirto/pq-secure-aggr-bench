from pqfl.aggregators.base import Aggregator
from pqfl.masking.core import Tensor
from pqfl.logging.metrics import RoundStats


class SchemeC(Aggregator):
    """Pending colleague implementation; use shared Metrics and byte semantics."""

    def setup(self, site_ids: list[str]) -> None:
        raise NotImplementedError("Scheme C pending")

    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Scheme C pending")

    def stats(self) -> RoundStats:
        raise NotImplementedError("Scheme C pending")
