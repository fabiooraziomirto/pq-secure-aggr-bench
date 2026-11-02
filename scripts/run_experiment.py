#!/usr/bin/env python3
"""Opt-in simulator cost run for one scheme on synthetic updates.

Never invoked by installation or tests. Simulator numbers are development
measurements, not distributed or network results.
"""
import argparse

from pqfl.config import SCHEMES, load_run_config
from pqfl.kem.backends import BackendUnavailable
from pqfl.runner import run_cell, site_ids


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--output", required=True, help="New JSONL file; never overwritten")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True, help="Unlogged rounds before measuring")
    parser.add_argument("--vector-size", type=int, required=True)
    parser.add_argument("--ledger-output", help="Optional new JSON transfer ledger for net_replay.py")
    parser.add_argument("--sites-count", type=int, help="Use this many generated sites instead of the config's")
    parser.add_argument("--rekey-every", type=int, help="Rekey every K rounds (overrides the config)")
    parser.add_argument("--peak-memory-output", help="Optional new JSON file with the process peak RSS")
    args = parser.parse_args()
    if args.rounds <= 0 or args.vector_size <= 0 or args.warmup < 0:
        parser.error("rounds and vector-size must be positive, warmup nonnegative")
    return args


def main():
    args = parse_args()
    cfg = load_run_config(args.config)
    try:
        sites = site_ids(args.sites_count) if args.sites_count else list(cfg.sites)
        run_cell(args.scheme, cfg, sites=sites, vector_size=args.vector_size,
                 rounds=args.rounds, warmup=args.warmup, output=args.output,
                 ledger_output=args.ledger_output, rekey_every=args.rekey_every,
                 peak_memory_output=args.peak_memory_output)
    except BackendUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    print(f"{args.scheme}: {args.warmup}+{args.rounds} rounds of {args.vector_size} elements -> {args.output}")


if __name__ == "__main__":
    main()
