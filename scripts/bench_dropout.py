#!/usr/bin/env python3
"""Opt-in dropout recovery benchmark (exploratory experiment of the protocol).

Each session: setup, a normal round, a round where one site crashes before uploading, and
the next round with the survivors. Records per-round bytes and timer sums, and the staged
ledgers of those rounds so the network share can be replayed with scripts/net_replay.py.
The detection timeout is a deployment parameter and is NOT included: recovery time =
timeout + the measured extra work and network of the dropout round + the next round.
Simulator numbers, serial in-process; not a distributed latency.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from pqfl.analysis import load_rows, per_round_samples
from pqfl.config import load_run_config
from pqfl.kem.backends import BackendUnavailable, LibOQSKEM
from pqfl.kem.hybrid import HybridKEM
from pqfl.logging.metrics import JSONLWriter
from pqfl.net.replay import ledger_stages
from pqfl.runner import build_authenticator, site_ids


def build(scheme, cfg, sites, threshold, writer):
    auth = build_authenticator(cfg, sites)
    kem = HybridKEM(LibOQSKEM())
    if scheme == "A":
        from pqfl.aggregators.scheme_a_dropout import SchemeADropout
        return SchemeADropout(kem, clip_bound=cfg.clip_bound, scale_bits=cfg.scale_bits,
                              modulus_bits=cfg.modulus_bits, authenticator=auth, writer=writer)
    from pqfl.aggregators.scheme_b_threshold import SchemeBGVThreshold
    return SchemeBGVThreshold(kem, clip_bound=cfg.clip_bound, threshold=threshold,
                              scale_bits=cfg.bgv_scale_bits, ring_dim=cfg.ring_dim,
                              plaintext_bits=cfg.bgv_plaintext_bits, authenticator=auth, writer=writer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scheme", choices=("A", "B-BGV"), required=True)
    parser.add_argument("--sites", type=int, required=True)
    parser.add_argument("--threshold", type=int, help="Decryption threshold t (B-BGV only)")
    parser.add_argument("--drop-index", type=int, required=True, help="Index of the site that crashes (0 = lead)")
    parser.add_argument("--vector-size", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True, help="Independent sessions")
    parser.add_argument("--output", required=True, help="New JSON file; never overwritten")
    args = parser.parse_args()
    if args.scheme == "B-BGV" and not args.threshold:
        parser.error("--threshold is required for B-BGV")
    cfg = load_run_config(args.config)
    sites = site_ids(args.sites)
    victim = sites[args.drop_index]
    rng = np.random.default_rng(cfg.run_seed)
    samples = {"normal": [], "dropout": [], "next": []}
    ledgers, setups = {}, []
    try:
        for session in range(args.repetitions):
            with tempfile.TemporaryDirectory() as directory:
                writer = JSONLWriter(Path(directory) / "s.jsonl", run_id=f"dropout-{args.scheme}", config_hash="-",
                                     scheme=args.scheme, dp_level=cfg.dp_level, run_seed=cfg.run_seed)
                agg = build(args.scheme, cfg, sites, args.threshold, writer)
                agg.setup(sites)
                ledgers["setup"] = ledger_stages(agg.stats())
                updates = lambda who: {s: rng.uniform(-cfg.clip_bound, cfg.clip_bound, args.vector_size) for s in who}
                agg.aggregate_round(0, updates(sites))
                ledgers["normal"] = ledger_stages(agg.stats())
                agg.aggregate_round(1, updates(sites), dropped={victim})
                ledgers["dropout"] = ledger_stages(agg.stats())
                survivors = [s for s in sites if s != victim]
                agg.aggregate_round(2, updates(survivors))
                ledgers["next"] = ledger_stages(agg.stats())
                setup, rounds = per_round_samples(load_rows(Path(directory) / "s.jsonl"))
                setups.append(setup)
                for name, sample in zip(("normal", "dropout", "next"), rounds):
                    samples[name].append(sample)
    except BackendUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    result = dict(scheme=args.scheme, sites=sites, dropped=victim, threshold=args.threshold,
                  vector_size=args.vector_size, sessions=args.repetitions, setup=setups,
                  rounds=samples, ledgers=ledgers)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle)
    median = lambda name, key: float(np.median([s[key] for s in samples[name]]))
    print(f"{args.scheme} n={args.sites} drop={victim}: bytes normal/dropout/next = "
          f"{median('normal', 'bytes'):,.0f} / {median('dropout', 'bytes'):,.0f} / {median('next', 'bytes'):,.0f}; "
          f"cpu_total_s = {median('normal', 'cpu_total_s'):.3f} / {median('dropout', 'cpu_total_s'):.3f} / "
          f"{median('next', 'cpu_total_s'):.3f}")


if __name__ == "__main__":
    main()
