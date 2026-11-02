#!/usr/bin/env python3
"""Summarize training runs (utility, attacks, H3/H4/H5) from a directory of run folders (opt-in, no GPU)."""
import argparse
import sys

from pqfl.training_report import load_runs, markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="directory with one folder per run (summary.json, final.jsonl)")
    parser.add_argument("--rounds", type=int, default=4, help="keep runs with this number of rounds (0: all)")
    parser.add_argument("--output", help="new markdown file; never overwritten (default: stdout)")
    args = parser.parse_args()
    rows = load_runs(args.runs, rounds=args.rounds or None)
    if not rows:
        sys.exit("no finished runs found")
    text = markdown(rows)
    if args.output:
        with open(args.output, "x", encoding="utf-8") as handle:
            handle.write(text)
        print(f"{len(rows)} runs summarized in {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
