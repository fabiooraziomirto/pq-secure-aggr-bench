#!/usr/bin/env python3
"""Build the MedMCQA site-split manifest from the official parquet files (opt-in).

Needs pandas and pyarrow only here, not in the package. Writes ids and counts, never text.
"""
import argparse

import pandas as pd

from pqfl.data.medmcqa import build_manifest, dump_manifest, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True, help="New JSON file; never overwritten")
    parser.add_argument("--sites", type=int, required=True)
    parser.add_argument("--pool", type=int, required=True, help="Member examples per site")
    parser.add_argument("--holdout", type=int, required=True, help="Non-member examples per site")
    parser.add_argument("--reference", type=int, required=True, help="Reference-model examples per site")
    parser.add_argument("--split-seed", type=int, required=True)
    args = parser.parse_args()
    train = pd.read_parquet(args.train, columns=["id", "subject_name"])
    validation = pd.read_parquet(args.validation, columns=["id"])
    records = [(str(i), str(s)) for i, s in zip(train["id"], train["subject_name"])
               if s != "Unknown"]
    manifest = build_manifest(records, [str(i) for i in validation["id"]], sites=args.sites,
                              pool=args.pool, holdout=args.holdout, reference=args.reference,
                              split_seed=args.split_seed)
    manifest["source"] = dict(dataset="openlifescienceai/medmcqa", train_sha256=file_sha256(args.train),
                              validation_sha256=file_sha256(args.validation),
                              train_rows=len(train), unknown_subject_excluded=int((train["subject_name"] == "Unknown").sum()))
    dump_manifest(manifest, args.output)
    for index, subjects in enumerate(manifest["subjects_by_site"]):
        print(f"site {index}: {manifest['train_examples_by_site'][index]} examples; subjects: {', '.join(subjects)}")


if __name__ == "__main__":
    main()
