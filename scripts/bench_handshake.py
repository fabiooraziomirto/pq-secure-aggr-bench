#!/usr/bin/env python3
"""Opt-in KEM microbenchmark. Never invoked by installation or tests.

All sample/warmup/percentile choices must be explicit. Not a network benchmark.
"""
import argparse
import time

import numpy as np

from pqfl.kem.backends import BackendUnavailable, LibOQSKEM
from pqfl.kem.hybrid import HybridKEM
from pqfl.logging.metrics import HandshakeSummary, JSONLWriter, RoundStats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Public YAML with an integer run_seed")
    parser.add_argument("--output", required=True, help="New JSONL file; never overwritten")
    parser.add_argument("--kem", choices=("mlkem", "hybrid"), required=True)
    parser.add_argument("--repetitions", type=int, required=True, help="TODO: choose on cluster")
    parser.add_argument("--warmup", type=int, required=True, help="TODO: choose on cluster")
    parser.add_argument("--percentiles", type=float, nargs="+", required=True,
                        help="TODO: choose percentiles in [0, 100]")
    args = parser.parse_args()
    if args.repetitions <= 0 or args.warmup < 0:
        parser.error("repetitions must be positive and warmup nonnegative")
    if any(not 0 <= p <= 100 for p in args.percentiles):
        parser.error("percentiles must lie in [0, 100]")
    return args


def sample(kem):
    # Include wrapper serialization and combiner work; exclude Python imports,
    # library selection, file I/O, equality validation and statistics reduction.
    start = time.perf_counter_ns()
    public, secret = kem.keygen()
    keygen_end = time.perf_counter_ns()
    ciphertext, shared = kem.encaps(public)
    encaps_end = time.perf_counter_ns()
    recovered = kem.decaps(secret, ciphertext)
    decaps_end = time.perf_counter_ns()
    if recovered != shared:
        raise RuntimeError("KEM round trip failed; no secret values will be logged")
    durations = dict(keygen=(keygen_end - start) / 1e9,
                     encaps=(encaps_end - keygen_end) / 1e9,
                     decaps=(decaps_end - encaps_end) / 1e9,
                     handshake=(decaps_end - start) / 1e9)
    return durations, (len(public), len(ciphertext))


def payload_stats(operation, lengths):
    public_size, ciphertext_size = lengths
    sent = {"initiator": 0, "responder": 0}
    received = {"initiator": 0, "responder": 0}
    if operation in {"encaps", "handshake"}:
        sent["initiator"] = received["responder"] = ciphertext_size
    if operation == "handshake":
        sent["responder"] = received["initiator"] = public_size
    return RoundStats(sent, received)


def main():
    args = parse_args()
    try:
        pq = LibOQSKEM()  # No legacy opt-in for a benchmark labelled ML-KEM.
    except BackendUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    kem = HybridKEM(pq) if args.kem == "hybrid" else pq
    # Validate configuration and reserve the output file before any measurements.
    writer = JSONLWriter.from_config(args.output, args.config,
                                    library_versions=kem.library_versions)
    for _ in range(args.warmup):
        sample(kem)
    samples = {name: [] for name in ("keygen", "encaps", "decaps", "handshake")}
    lengths = None
    for _ in range(args.repetitions):
        durations, current_lengths = sample(kem)
        if lengths is not None and lengths != current_lengths:
            raise RuntimeError("Variable payload sizes need a per-sample accounting policy")
        lengths = current_lengths
        for name, duration in durations.items():
            samples[name].append(duration)
    for operation, values in samples.items():
        summary = HandshakeSummary(operation=operation,
                       mechanism=kem.mechanism, repetitions=args.repetitions,
                       warmup=args.warmup,
                       percentiles_s={str(p): float(np.percentile(values, p, method="linear"))
                                      for p in args.percentiles})
        writer.record(round_idx=None, phase="crypto", duration_s=float(np.median(values)),
                      stats=payload_stats(operation, lengths), summary=summary)


if __name__ == "__main__":
    main()
