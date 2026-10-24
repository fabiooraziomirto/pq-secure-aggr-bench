import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version


@dataclass
class RoundStats:
    bytes_sent: dict[str, int] = field(default_factory=dict)
    bytes_received: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class HandshakeSummary:
    """Public aggregate measurements, with no field for cryptographic material."""

    operation: str
    mechanism: str
    repetitions: int
    warmup: int
    percentiles_s: dict[str, float]
    kind: str = field(default="sample_summary", init=False)
    percentile_method: str = field(default="linear", init=False)
    timing_scope: str = field(default="in_process_kem_api", init=False)
    byte_scope: str = field(default="logical_payload_per_handshake", init=False)

    def __post_init__(self):
        if self.operation not in {"keygen", "encaps", "decaps", "handshake"}:
            raise ValueError("Unknown handshake operation")
        if self.repetitions <= 0 or self.warmup < 0:
            raise ValueError("Invalid sample counts")
        if any(not isinstance(value, (int, float)) or value < 0
               for value in self.percentiles_s.values()):
            raise ValueError("Percentile durations must be nonnegative numbers")


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

    def __init__(self, path, *, run_id, config_hash, scheme, dp_level, run_seed, library_versions=None):
        if type(run_seed) is not int or run_seed < 0:
            raise ValueError("Set run_seed to a nonnegative experiment seed in the configuration")
        self.library_versions = dict(library_versions or {})
        self._env_written = False
        self.path = path
        self.metadata = dict(run_id=run_id, config_hash=config_hash,
                             scheme=scheme, dp_level=dp_level, run_seed=run_seed, schema_version=2)
        # One writer owns one new run file; never append to another execution.
        with open(path, "x", encoding="utf-8"):
            pass

    @classmethod
    def from_config(cls, path, config_path, *, library_versions=None):
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
                   run_seed=config["run_seed"], library_versions=library_versions)

    def record(self, *, round_idx, phase, duration_s, stats, site=None, epoch=0, summary=None):
        if phase not in {"auth", "compute", "crypto", "comm"}:
            raise ValueError("Unknown phase")
        if site is not None and not isinstance(site, str):
            raise ValueError("site must be a string or None")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        row = dict(self.metadata, round=round_idx, phase=phase, site=site, epoch=epoch,
                   duration_s=duration_s, bytes_sent=stats.bytes_sent,
                   bytes_received=stats.bytes_received, peak_mem_gpu_mb=None,
                   peak_mem_host_mb=None, timestamp=datetime.now(timezone.utc).isoformat(),
                   library_versions={**{name: version(name) for name in ("numpy", "cryptography")},
                                     **self.library_versions})
        include_env = not self._env_written and round_idx is None and epoch == 0
        if include_env:
            from pqfl.logging.environment import environment_metadata
            row["env"] = environment_metadata()
        if summary is not None:
            if not isinstance(summary, HandshakeSummary):
                raise TypeError("summary must be a HandshakeSummary, not an arbitrary payload")
            row["summary"] = asdict(summary)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if include_env:
            self._env_written = True


class Metrics:
    def __init__(self, site_ids, writer=None, *, epoch=0):
        self.epoch = epoch
        self.counter = ByteCounter(site_ids)
        self.writer = writer
        self.round_idx = None
        self._last_written = RoundStats(dict.fromkeys(site_ids, 0), dict.fromkeys(site_ids, 0))

    @contextmanager
    def phase_timer(self, phase, *, site=None):
        if phase not in {"auth", "compute", "crypto", "comm"}:
            raise ValueError("Unknown phase")
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.writer is not None:
                current = self.counter.value
                delta = RoundStats(
                    {actor: size - self._last_written.bytes_sent[actor]
                     for actor, size in current.bytes_sent.items()},
                    {actor: size - self._last_written.bytes_received[actor]
                     for actor, size in current.bytes_received.items()})
                self.writer.record(round_idx=self.round_idx, phase=phase,
                                   duration_s=time.perf_counter() - start,
                                   stats=delta, site=site, epoch=self.epoch)
                self._last_written = RoundStats(dict(current.bytes_sent), dict(current.bytes_received))
