#!/usr/bin/env python3
"""Export MedMCQA rows (as JSONL) for the slices of the site-split manifest (opt-in; needs pandas)."""
import argparse
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--slice", required=True, choices=("pool", "holdout", "reference", "validation"))
    parser.add_argument("--output", required=True, help="New JSONL file; never overwritten")
    args = parser.parse_args()
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    if args.slice == "validation":
        frame = pd.read_parquet(args.validation)
        ids = set(manifest["validation_ids"])
    else:
        frame = pd.read_parquet(args.train)
        ids = {i for site in manifest[args.slice] for i in site}
    frame = frame[frame["id"].isin(ids)]
    clean = lambda v: None if isinstance(v, float) and v != v else v
    with open(args.output, "x", encoding="utf-8") as handle:
        for row in frame.to_dict("records"):
            handle.write(json.dumps({k: clean(v) for k, v in row.items()}, default=str) + "\n")
    print(f"{len(frame)} rows of {args.slice} written to {args.output} (expected {len(ids)})")


if __name__ == "__main__":
    main()
