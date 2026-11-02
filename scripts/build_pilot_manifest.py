#!/usr/bin/env python3
"""Checksum manifest of the pilot evidence (result files, configs, specs), for provenance (opt-in, read-only).

Writes a new JSON file listing path, size and sha256 of every file selected by the patterns, plus the Python
environment's key library versions. It never overwrites an existing manifest.
"""
import argparse
import glob
import hashlib
import importlib.metadata as md
import json
import os
import time

PATTERNS = ["runs/campaign-3/report.md", "runs/campaign-3/spec.yaml", "runs/campaign-3/state.json",
            "runs/campaign-3/compute/summary.csv", "runs/campaign-3/final-status.txt", "runs/campaign-3/perturbations.log",
            "runs/campaign-4/report.md", "runs/campaign-4/spec.yaml", "runs/campaign-4/state.json",
            "runs/campaign-4/compute/summary.csv", "runs/campaign-4/final-status.txt",
            "runs/campaign-5/report.md", "runs/campaign-5/spec.yaml", "runs/campaign-5/state.json",
            "runs/campaign-5/final-status.txt", "runs/campaign-5/load.log",
            "runs/campaign-6/report.md", "runs/campaign-6/spec.yaml", "runs/campaign-6/state.json",
            "runs/campaign-6/final-status.txt", "runs/campaign-6/load.log",
            "runs/campaign-3/net/*/*.json", "runs/campaign-4/net/*/*.json", "runs/campaign-5/net/*/*.json",
            "runs/campaign-6/net/*/*.json", "runs/campaign-4/load.log", "runs/campaign-4/compute/*.csv",
            "runs/campaign-5/compute/*.csv", "runs/campaign-6/compute/*.csv", "runs/pilot/code-snapshot.sha256",
            "runs/old/README.md",
            "runs/pilot/*/config.json", "runs/pilot/*/summary.json", "runs/pilot/*/final.jsonl", "runs/pilot/*/rounds.jsonl",
            "runs/pilot-q08b/*/summary.json", "runs/pilot-q08b/*/final.jsonl", "runs/pilot-q08b/*/config.json",
            "runs/pilot-q08b/*.jsonl", "runs/scoring-v2/*", "runs/scoring.jsonl", "runs/scoring-rounds.jsonl",
            "runs/gpu-queue/done/*.json", "runs/dropout/summary.md", "runs/tls-baseline.json",
            "configs/*.yaml", "configs/data/*.json", "docs/decisions.md", "docs/results_dossier.md", "docs/pilot_ledger.md"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New JSON file; never overwritten")
    args = parser.parse_args()
    files = sorted({f for pattern in PATTERNS for f in glob.glob(pattern) if os.path.isfile(f)})
    libs = {}
    for name in ("numpy", "cryptography", "liboqs-python", "openfhe", "torch", "transformers", "peft", "PyYAML"):
        try:
            libs[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.uname().nodename, python_libs=libs,
                    files=[dict(path=f, bytes=os.path.getsize(f), sha256=sha256(f)) for f in files])
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(f"{len(files)} files listed in {args.output}")


if __name__ == "__main__":
    main()
