#!/usr/bin/env python3
"""Write report.md for a finished campaign directory (read-only on the results)."""
import argparse

from pqfl.report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output", required=True, help="New markdown file; never overwritten")
    args = parser.parse_args()
    report = build_report(args.campaign_dir)
    with open(args.output, "x", encoding="utf-8") as handle:
        handle.write(report)
    print(f"report written to {args.output} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
