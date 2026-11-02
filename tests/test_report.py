import csv
import json
import os
import time
from datetime import datetime

import pytest

from pqfl.report import build_report, overlaps, parse_windows

SUMMARY = ["group", "peak_host_rss_mb", "scheme", "sites", "vector_size", "repetitions", "warmup", "setup_bytes",
           "setup_s", "bytes_p50", "bytes_p95", "bytes_p99", "cpu_total_s_p50", "cpu_total_s_p95", "cpu_total_s_p99",
           "site_max_s_p50", "site_max_s_p95", "site_max_s_p99", "reducer_s_p50", "reducer_s_p95", "reducer_s_p99",
           "crypto_s_p50", "crypto_s_p95", "crypto_s_p99", "compute_s_p50", "compute_s_p95", "compute_s_p99"]
HANDSHAKE = ["group", "scheme", "sites", "rekey_every", "handshakes", "bytes_p50", "bytes_p95", "bytes_p99",
             "time_s_p50", "time_s_p95", "time_s_p99", "crypto_s_p50", "crypto_s_p95", "crypto_s_p99",
             "auth_s_p50", "auth_s_p95", "auth_s_p99"]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, 1.0) for k in header})


def make_campaign(tmp_path, perturb_cell=False):
    root = tmp_path / "camp"
    (root / "compute").mkdir(parents=True)
    (root / "net" / "wan").mkdir(parents=True)
    write_csv(root / "compute" / "summary.csv", SUMMARY, [
        dict(group="main", scheme="A", sites=3, vector_size=100000, repetitions=30, warmup=3, setup_bytes=23199,
             bytes_p50=3000000, peak_host_rss_mb=120.5, crypto_s_p50=0.01),
        dict(group="scaling", scheme="C", sites=5, vector_size=100000, repetitions=30, warmup=3, setup_bytes=77330,
             bytes_p50=15000000, peak_host_rss_mb=130.0)])
    write_csv(root / "compute" / "handshake.csv", HANDSHAKE, [
        dict(group="handshake-every-1", scheme="A", sites=3, rekey_every=1, handshakes=30, bytes_p50=23199, time_s_p50=0.003)])
    wire = {"iterations": 4, "actors": {"S0000": {"rx_bytes": 3_300_000, "tx_bytes": 3_100_000 * 4 // 3},
                                        "reducer": {"rx_bytes": 5, "tx_bytes": 9_000_000}}}
    result = dict(profile="wan", scheme="A", sites=["S0000", "S0001", "S0002"], vector_size=100000,
                  repetitions=3, warmup=1, tcp={}, setup_network_s=[0.15, 0.16, 0.14],
                  round_network_s=[0.30, 0.32, 0.28], wire_bytes={"setup": wire, "round": {"iterations": 4, "actors": {
                      "S0000": {"rx_bytes": 0, "tx_bytes": 4_000_000}, "S1": {"rx_bytes": 0, "tx_bytes": 8_000_000}}}})
    file = root / "net" / "wan" / "main__A-n3-v100000.json"
    file.write_text(json.dumps(result))
    end = time.time() if not perturb_cell else time.mktime((2026, 10, 30, 22, 0, 0, 0, 0, -1))
    os.utime(file, (end, end))
    (root / "state.json").write_text(json.dumps({"cells": {"net:main:wan:A-n3-v100000": {"seconds": 60.0}}}))
    (root / "perturbations.log").write_text(
        "21:55:10 pilot 4B-Base: download then GPU run (may perturb)\n22:07:20 pilot 4B-Base finished (exit 0)\n"
        "23:58:37 GPU worker started, reserving all available memory\n")
    return root


def test_windows_parse_and_overlap_use_the_clock_not_the_day_of_the_log():
    windows = parse_windows(open_lines := ["21:55:10 pilot (may perturb)", "22:07:20 pilot finished (exit 0)",
                                           "23:58:37 GPU worker started, reserving"])
    assert windows == [("21:55:10", "22:07:20", "pilot (may perturb)")]
    t = lambda h, m: time.mktime((2026, 10, 30, h, m, 0, 0, 0, -1))
    assert overlaps((t(22, 0), t(22, 5)), windows) == ["pilot (may perturb)"]
    assert overlaps((t(21, 0), t(21, 50)), windows) == []
    assert overlaps((t(21, 50), t(21, 56)), windows)                       # partial overlap counts
    midnight = time.mktime((2026, 10, 31, 0, 30, 0, 0, 0, -1))
    assert overlaps((midnight, midnight + 60), windows) == []              # day before has no overlap here


def test_report_contains_every_section_with_the_right_numbers(tmp_path):
    report = build_report(str(make_campaign(tmp_path)))
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert heading in report
    assert "| A | 100000 | 30 | 23,199 | 3.00 |" in report and "120.5" in report
    assert "| C | 5 | 100000 |" in report                                   # scaling table
    assert "| wan | main | A | 100000 | 3 | 0.300/0.318/0.320 | 0.150 |" in report
    assert "| A | 1 | 30 | 23,199 | 0.0030 | wan | 0.150 | 0.153 | 15 |" in report   # round time for < 1 %
    assert "lan | n/a" in report                                            # no lan result in the fixture


def test_wire_to_payload_ratio_and_perturbation_flag(tmp_path):
    report = build_report(str(make_campaign(tmp_path, perturb_cell=True)))
    assert "| 1.000 | ⚠ |" in report                                       # (4e6+8e6) / 4 iterations / 3e6 payload
    assert "net wan main A-n3-v100000: overlaps pilot 4B-Base: download then GPU run (may perturb)" in report
    clean = build_report(str(make_campaign(tmp_path / "clean")))
    assert "no cell overlaps a window" in clean
