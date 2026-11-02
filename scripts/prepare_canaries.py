#!/usr/bin/env python3
"""Generate the canary manifest (random digits only; opt-in, never overwrites)."""
import argparse
import json

from pqfl.data.canaries import generate_canaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New JSON file; never overwritten")
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--sites", type=int, required=True)
    parser.add_argument("--levels", required=True, help="repetitions:count pairs, e.g. 0:10,1:10,4:10,16:10")
    parser.add_argument("--case-digits", type=int, required=True)
    parser.add_argument("--secret-digits", type=int, required=True)
    parser.add_argument("--references", type=int, required=True)
    args = parser.parse_args()
    levels = {int(a): int(b) for a, b in (pair.split(":") for pair in args.levels.split(","))}
    manifest = generate_canaries(split_seed=args.split_seed, sites=args.sites, levels=levels,
                                 case_digits=args.case_digits, secret_digits=args.secret_digits,
                                 references=args.references)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
    inserted = sum(c["repetitions"] for c in manifest["canaries"])
    print(f"{len(manifest['canaries'])} canaries; {inserted} inserted copies in total")


if __name__ == "__main__":
    main()
