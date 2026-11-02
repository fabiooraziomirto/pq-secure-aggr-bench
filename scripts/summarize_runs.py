#!/usr/bin/env python3
"""Summarize run JSONL files into a markdown table with explicit percentiles."""
import argparse

from pqfl.analysis import load_rows, per_round_samples, percentiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--percentiles", type=float, nargs="+", required=True)
    args = parser.parse_args()
    if any(not 0 <= p <= 100 for p in args.percentiles):
        parser.error("percentiles must lie in [0, 100]")
    cols = [f"p{p:g}" for p in args.percentiles]
    print("| scheme | n rounds | setup bytes | round bytes (median) | "
          + " | ".join(f"crypto s {c}" for c in cols) + " | "
          + " | ".join(f"site-max s {c}" for c in cols) + " |")
    print("| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in cols * 2) + " |")
    for path in args.files:
        rows = load_rows(path)
        setup, samples = per_round_samples(rows)
        crypto = percentiles([s["crypto_s"] for s in samples], args.percentiles)
        site = percentiles([s["site_max_s"] for s in samples], args.percentiles)
        median_bytes = percentiles([s["bytes"] for s in samples], [50])["50"]
        print(f"| {rows[0]['scheme']} | {len(samples)} | {setup['bytes']:,} | {median_bytes:,.0f} | "
              + " | ".join(f"{v:.4f}" for v in crypto.values()) + " | "
              + " | ".join(f"{v:.4f}" for v in site.values()) + " |")


if __name__ == "__main__":
    main()
