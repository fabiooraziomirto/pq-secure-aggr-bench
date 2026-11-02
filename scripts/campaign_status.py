#!/usr/bin/env python3
"""Print campaign progress and a remaining-time estimate from state.json."""
import argparse
import json
import time

from pqfl.campaign import eta


def fmt(seconds):
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state")
    args = parser.parse_args()
    data = json.load(open(args.state, encoding="utf-8"))
    report = eta(data)
    print(f"elapsed {fmt(report['elapsed_s'])}, estimated remaining {fmt(report['remaining_s'])}")
    for phase, p in sorted(report["phases"].items()):
        print(f"  {phase}: {p['done']}/{p['total']} done, {p['skipped']} skipped, {p['failed']} failed, "
              f"obs/est ratio {p['ratio']:.2f}, remaining {fmt(p['remaining_s'])}")
    running = [k for k, c in data["cells"].items() if c["status"] == "running"]
    print("  running:", ", ".join(running) or "-")
    for key, cell in data["cells"].items():
        if cell["status"] == "failed":
            print("  FAILED", key, cell.get("error", "")[:160])
    print("  updated", time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
