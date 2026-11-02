"""Validated public run configuration; secrets never belong here."""
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real

import yaml

SCHEMES = ("A", "B", "B-BGV", "C", "N", "N-FP", "STUB", "A-dropout", "B-BGV-threshold")
AUTH_METHODS = ("mldsa_ephemeral_simulation", "none_insecure")


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    sites: tuple[str, ...]
    clip_bound: float
    scale_bits: int
    modulus_bits: int
    ring_dim: int
    scaling_mod_size: int
    dp_level: str
    run_seed: int
    rekey_every_rounds: int | None
    auth_method: str
    canonical: bytes
    bgv_scale_bits: int = 20
    bgv_plaintext_bits: int = 31
    dropout_threshold: int | None = None

    def config_hash(self, scheme: str) -> str:
        return hashlib.sha256(self.canonical + scheme.encode()).hexdigest()


def _require(config: dict, key: str, kind):
    value = config.get(key)
    if value is None or (isinstance(value, str) and "TODO" in value):
        raise ValueError(f"Config field '{key}' is missing or still TODO")
    if isinstance(value, bool) or not isinstance(value, kind):
        raise ValueError(f"Config field '{key}' has the wrong type")
    return value


def load_run_config(path) -> RunConfig:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    sites = _require(config, "sites", list)
    if len(sites) < 2 or any(not isinstance(s, str) or not s for s in sites) or len(set(sites)) != len(sites):
        raise ValueError("sites must be at least two distinct nonempty strings")
    clip_bound = _require(config, "clip_bound", Real)
    if not math.isfinite(clip_bound) or clip_bound <= 0:
        raise ValueError("clip_bound must be positive and finite")
    run_seed = _require(config, "run_seed", int)
    if run_seed < 0:
        raise ValueError("run_seed must be nonnegative")
    rekey = config.get("rekey_every_rounds")
    if rekey is not None and (isinstance(rekey, bool) or not isinstance(rekey, int) or rekey <= 0):
        raise ValueError("rekey_every_rounds must be null or a positive integer")
    auth = _require(config, "auth", dict)
    method = _require(auth, "method", str)
    if method not in AUTH_METHODS:
        raise ValueError(f"auth.method must be one of {AUTH_METHODS}")
    threshold = config.get("dropout_threshold")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 2):
        raise ValueError("dropout_threshold must be null or an integer of at least 2")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return RunConfig(run_id=_require(config, "run_id", str), sites=tuple(sites),
                     clip_bound=float(clip_bound), scale_bits=_require(config, "scale_bits", int),
                     modulus_bits=_require(config, "modulus_bits", int),
                     ring_dim=_require(config, "ring_dim", int),
                     scaling_mod_size=_require(config, "scaling_mod_size", int),
                     dp_level=_require(config, "dp_level", str), run_seed=run_seed,
                     rekey_every_rounds=rekey, auth_method=method, canonical=canonical,
                     bgv_scale_bits=config.get("bgv_scale_bits", 20),
                     bgv_plaintext_bits=config.get("bgv_plaintext_bits", 31),
                     dropout_threshold=threshold)
