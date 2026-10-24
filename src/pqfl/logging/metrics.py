import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version


@dataclass
class RoundStats:
    bytes_sent: dict[str, int] = field(default_factory=dict)
    bytes_received: dict[str, int] = field(default_factory=dict)


class ByteCounter:
    def __init__(self, site_ids):
        self.value = RoundStats(dict.fromkeys(site_ids, 0), dict.fromkeys(site_ids, 0))

    def transfer(self, sender: str, receiver: str, size: int):
        if type(size) is not int or size < 0:
            raise ValueError("Byte count must be nonnegative")
        self.value.bytes_sent[sender] += size
        self.value.bytes_received[receiver] += size


class JSONLWriter:
    """Allowlisted metadata only; no caller-supplied arbitrary payloads."""

    def __init__(self, path, *, run_id, config_hash, scheme, dp_level, run_seed):
        if type(run_seed) is not int or run_seed < 0:
            raise ValueError("Set run_seed to a nonnegative experiment seed in the configuration")
        self.path = path
        self.metadata = dict(run_id=run_id, config_hash=config_hash,
                             scheme=scheme, dp_level=dp_level, run_seed=run_seed, schema_version=1)
        # One writer owns one new run file; never append to another execution.
        with open(path, "x", encoding="utf-8"):
            pass

    @classmethod
    def from_config(cls, path, config_path):
        """Read public experiment metadata; never use run_seed for real KEMs."""
        import yaml

        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a mapping")
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode()
        return cls(path, run_id=config["run_id"],
                   config_hash=hashlib.sha256(canonical).hexdigest(),
                   scheme=config["scheme"], dp_level=config["dp_level"],
                   run_seed=config["run_seed"])

    def record(self, *, round_idx, phase, duration_s, stats):
        if phase not in {"compute", "crypto", "comm"}:
            raise ValueError("Unknown phase")
        row = dict(self.metadata, round=round_idx, phase=phase,
                   duration_s=duration_s, bytes_sent=stats.bytes_sent,
                   bytes_received=stats.bytes_received, peak_mem_gpu_mb=None,
                   peak_mem_host_mb=None, timestamp=datetime.now(timezone.utc).isoformat(),
                   library_versions={name: version(name) for name in ("numpy", "cryptography")})
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


class Metrics:
    def __init__(self, site_ids, writer=None):
        self.counter = ByteCounter(site_ids)
        self.writer = writer
        self.round_idx = None

    @contextmanager
    def phase_timer(self, phase):
        if phase not in {"compute", "crypto", "comm"}:
            raise ValueError("Unknown phase")
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.writer is not None:
                self.writer.record(round_idx=self.round_idx, phase=phase,
                                   duration_s=time.perf_counter() - start,
                                   stats=self.counter.value)
