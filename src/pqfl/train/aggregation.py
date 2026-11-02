"""Aggregators for the training loop, built like in the cost campaign (real KEM and ML-DSA identities).

No torch here: this module also runs where torch is absent. `clip_bound` is the per-element limit B_inf
of the secure schemes; the loop clamps every update to it before aggregating (post-processing), so the
fixed-point range of A, C and B-BGV is guaranteed.
"""
import os

import yaml

from pqfl.config import load_run_config
from pqfl.logging.metrics import JSONLWriter
from pqfl.runner import build_aggregator


def build_training_aggregator(scheme: str, sites: list[str], *, clip_bound: float, out_dir: str, seed: int,
                              dp_level: str = "none", run_id: str = "train", ring_dim: int = 16384):
    """Aggregator already set up for `sites`, logging phase timers and byte deltas to out_dir/aggregation*.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    config = dict(run_id=run_id, sites=list(sites), clip_bound=float(clip_bound), scale_bits=24, modulus_bits=40,
                  ring_dim=ring_dim, scaling_mod_size=50, dp_level=dp_level, run_seed=seed, rekey_every_rounds=None,
                  auth={"method": "mldsa_ephemeral_simulation"}, bgv_scale_bits=20, bgv_plaintext_bits=31)
    path = os.path.join(out_dir, "runconfig.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    cfg = load_run_config(path)
    log, counter = os.path.join(out_dir, "aggregation.jsonl"), 1
    while os.path.exists(log):          # a resumed run starts a new aggregator session and a new log
        log, counter = os.path.join(out_dir, f"aggregation-{counter}.jsonl"), counter + 1
    writer = JSONLWriter(log, run_id=run_id, config_hash=cfg.config_hash(scheme), scheme=scheme,
                         dp_level=dp_level, run_seed=seed)
    aggregator = build_aggregator(scheme, cfg, list(sites), writer)
    aggregator.setup(list(sites))
    return aggregator
