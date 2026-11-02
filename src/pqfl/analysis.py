"""Reduce run JSONL files to per-round cost samples and percentiles."""
import json
from collections import defaultdict

import numpy as np


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def per_round_samples(rows):
    """Return setup totals and one sample per logged round.

    Time fields are sums of the in-process timers: `cpu_total_s` over all actors,
    `site_max_s` the busiest single site, `reducer_s` the reducer. Serial
    simulation time is not a distributed wall-clock.
    """
    setup = dict(bytes=0, time_s=0.0)
    rounds = defaultdict(lambda: dict(bytes=0, phase=defaultdict(float), actor=defaultdict(float)))
    for row in rows:
        total = sum(row["bytes_sent"].values())
        if row["round"] is None:
            if row.get("epoch", 0) == 0:
                setup["bytes"] += total
                setup["time_s"] += row["duration_s"]
            continue
        bucket = rounds[row["round"]]
        bucket["bytes"] += total
        bucket["phase"][row["phase"]] += row["duration_s"]
        bucket["actor"][row["site"]] += row["duration_s"]
    samples = []
    for index in sorted(rounds):
        bucket = rounds[index]
        sites = [t for actor, t in bucket["actor"].items() if actor != "reducer"]
        samples.append(dict(bytes=bucket["bytes"],
                            cpu_total_s=sum(bucket["actor"].values()),
                            site_max_s=max(sites) if sites else 0.0,
                            reducer_s=bucket["actor"].get("reducer", 0.0),
                            crypto_s=bucket["phase"].get("crypto", 0.0),
                            compute_s=bucket["phase"].get("compute", 0.0)))
    return setup, samples


def percentiles(values, points):
    return {str(p): float(np.percentile(values, p, method="linear")) for p in points}


def summary_row(scheme, sites, vector_size, repetitions, warmup, path, points):
    """One summary.csv row for a cell JSONL: setup totals and per-round percentiles."""
    setup, samples = per_round_samples(load_rows(path))
    row = dict(scheme=scheme, sites=sites, vector_size=vector_size, repetitions=repetitions,
               warmup=warmup, setup_bytes=setup["bytes"], setup_s=f"{setup['time_s']:.6f}")
    for metric in ("bytes", "cpu_total_s", "site_max_s", "reducer_s", "crypto_s", "compute_s"):
        for point, value in percentiles([s[metric] for s in samples], points).items():
            row[f"{metric}_p{float(point):g}"] = f"{value:.6f}"
    return row


def handshake_samples(rows):
    """One sample per rekey (round None, epoch >= 1): bytes and summed timer seconds."""
    grouped = defaultdict(lambda: dict(bytes=0, time_s=0.0, crypto_s=0.0, auth_s=0.0))
    for row in rows:
        if row["round"] is None and row.get("epoch", 0) >= 1:
            bucket = grouped[row["epoch"]]
            bucket["bytes"] += sum(row["bytes_sent"].values())
            bucket["time_s"] += row["duration_s"]
            if row["phase"] in ("crypto", "auth"):
                bucket[f"{row['phase']}_s"] += row["duration_s"]
    return [grouped[epoch] for epoch in sorted(grouped)]
