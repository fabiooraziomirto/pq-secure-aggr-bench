#!/usr/bin/env python3
"""Opt-in simulator sweep over schemes x site counts x vector sizes.

Each cell is a fresh session with its own JSONL file in --output-dir plus one
summary.csv row. Every count, warm-up and percentile is explicit. Not a network
benchmark: timings are serial in-process simulator time.
"""
import argparse
import csv
from pathlib import Path

from pqfl.analysis import load_rows, per_round_samples, percentiles
from pqfl.config import SCHEMES, load_run_config
from pqfl.kem.backends import BackendUnavailable
from pqfl.runner import run_cell, site_ids


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; never reused")
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, required=True)
    parser.add_argument("--sites", type=int, nargs="+", required=True)
    parser.add_argument("--vector-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--percentiles", type=float, nargs="+", required=True)
    parser.add_argument("--skip", nargs="*", default=[], metavar="SCHEME:SITES:SIZE",
                        help="Cells to omit, e.g. B:10:1000000 (documented in the summary)")
    args = parser.parse_args()
    if args.repetitions <= 0 or args.warmup < 0 or any(n < 2 for n in args.sites) \
            or any(v <= 0 for v in args.vector_sizes):
        parser.error("invalid repetitions, warmup, sites or vector sizes")
    if any(not 0 <= p <= 100 for p in args.percentiles):
        parser.error("percentiles must lie in [0, 100]")
    return args


def main():
    args = parse_args()
    cfg = load_run_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True)  # fails if it exists: no mixing of executions
    fields = ["scheme", "sites", "vector_size", "repetitions", "warmup", "setup_bytes", "setup_s"]
    for metric in ("bytes", "cpu_total_s", "site_max_s", "reducer_s", "crypto_s"):
        fields += [f"{metric}_p{p:g}" for p in args.percentiles]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scheme in args.schemes:
            for count in args.sites:
                for size in args.vector_sizes:
                    if f"{scheme}:{count}:{size}" in args.skip:
                        continue
                    path = out / f"{scheme}-n{count}-v{size}.jsonl"
                    try:
                        run_cell(scheme, cfg, sites=site_ids(count), vector_size=size,
                                 rounds=args.repetitions, warmup=args.warmup, output=path)
                    except BackendUnavailable as exc:
                        raise SystemExit(str(exc)) from exc
                    setup, samples = per_round_samples(load_rows(path))
                    row = dict(scheme=scheme, sites=count, vector_size=size,
                               repetitions=args.repetitions, warmup=args.warmup,
                               setup_bytes=setup["bytes"], setup_s=f"{setup['time_s']:.6f}")
                    for metric in ("bytes", "cpu_total_s", "site_max_s", "reducer_s", "crypto_s"):
                        for point, value in percentiles([s[metric] for s in samples], args.percentiles).items():
                            row[f"{metric}_p{float(point):g}"] = f"{value:.6f}"
                    writer.writerow(row)
                    handle.flush()
                    print(f"done {scheme} n={count} size={size}", flush=True)


if __name__ == "__main__":
    main()
