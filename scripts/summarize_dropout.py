#!/usr/bin/env python3
"""Markdown table of the dropout benchmark results (read-only)."""
import argparse
import glob

from pqfl.dropout_report import markdown, summarize
from pqfl.net.profiles import load_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Directory with the bench_dropout JSON files")
    parser.add_argument("--profiles-file", default="configs/network_profiles.yaml")
    parser.add_argument("--profiles", nargs="+", required=True)
    args = parser.parse_args()
    profiles = {name: load_profiles(args.profiles_file)[name] for name in args.profiles}
    print(markdown([summarize(path, profiles) for path in sorted(glob.glob(f"{args.results}/*.json"))], args.profiles))


if __name__ == "__main__":
    main()
