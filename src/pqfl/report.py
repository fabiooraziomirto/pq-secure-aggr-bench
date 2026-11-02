"""Markdown report from a finished campaign directory (summary.csv, handshake.csv, net/*, state.json).

Cells that overlap a perturbation window of perturbations.log (HH:MM:SS lines: a start line, then a
line containing "finished") are flagged, never dropped. Network percentiles come from the raw
samples, so their sample counts are shown: wan cells of B and B-BGV have few repetitions.
"""
import csv
import glob
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np

SCHEME_ORDER = {"N": 0, "N-FP": 0.5, "A": 1, "B": 2, "B-BGV": 3, "C": 4}


def parse_windows(lines):
    """[(start 'HH:MM:SS', end 'HH:MM:SS', label)]; lines about the GPU worker are point events."""
    windows, open_window = [], None
    for line in lines:
        stamp, _, text = line.strip().partition(" ")
        if not stamp or ":" not in stamp:
            continue
        if "finished" in text:
            if open_window:
                windows.append((open_window[0], stamp, open_window[1]))
                open_window = None
        elif "GPU worker started" not in text:
            open_window = (stamp, text)
    return windows


def _epoch(day, hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return time.mktime((day.year, day.month, day.day, h, m, s, 0, 0, -1))


def overlaps(interval, windows):
    """Labels of the windows (on the day of the interval or the day before) that intersect it."""
    start, end = interval
    days = {datetime.fromtimestamp(end).date(), datetime.fromtimestamp(start).date(),
            datetime.fromtimestamp(start).date() - timedelta(days=1)}
    hits = []
    for first, last, label in windows:
        for day in days:
            if _epoch(day, first) < end and start < _epoch(day, last):
                hits.append(label)
                break
    return hits


def _rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row, key):
    return float(row[key])


def _pct(values):
    return [float(np.percentile(values, p, method="linear")) for p in (50, 95, 99)]


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def _key(row):
    return (int(row["sites"]) if "sites" in row else 0, int(row["vector_size"]), SCHEME_ORDER.get(row["scheme"], 9))


def load_net(campaign):
    """[(profile, group, cell_id, result dict, interval or None)] for every net result file."""
    state = {}
    state_path = os.path.join(campaign, "state.json")
    if os.path.exists(state_path):
        state = json.load(open(state_path, encoding="utf-8"))["cells"]
    out = []
    for path in sorted(glob.glob(os.path.join(campaign, "net", "*", "*.json"))):
        profile = os.path.basename(os.path.dirname(path))
        group, cid = os.path.basename(path)[:-5].split("__", 1)
        result = json.load(open(path, encoding="utf-8"))
        seconds = (state.get(f"net:{group}:{profile}:{cid}") or {}).get("seconds")
        end = os.path.getmtime(path)
        out.append((profile, group, cid, result, (end - seconds, end) if seconds else None))
    return out


def _skipped_line(campaign: str) -> str:
    path = os.path.join(campaign, "state.json")
    cells = json.load(open(path, encoding="utf-8")).get("cells", {}) if os.path.exists(path) else {}
    skipped = sorted(cid for cid, cell in cells.items() if cell.get("status") == "skipped")
    return ("- Skipped cells (reason in state.json): " + ", ".join(skipped) + "; extrapolate if needed."
            if skipped else "- No cell was skipped.")


def _load_line(campaign: str) -> str:
    path = os.path.join(campaign, "load.log")
    if os.path.exists(path):
        rows = [line.split() for line in open(path, encoding="utf-8").read().splitlines()[1:] if line.strip()]
        if rows:
            load = float(np.mean([float(r[1]) for r in rows]))
            busy = float(np.mean([float(r[3]) for r in rows]))
            return (f"- The host is shared (load average {load:.1f} on average, cpu busy {100 * busy:.0f} % on average "
                    "during the campaign): timings are not those of a dedicated node.")
    return "- The host is shared: timings are not those of a dedicated node."


def build_report(campaign: str) -> str:
    summary = _rows(os.path.join(campaign, "compute", "summary.csv"))
    handshake_path = os.path.join(campaign, "compute", "handshake.csv")
    handshake = _rows(handshake_path) if os.path.exists(handshake_path) else []     # absent without rekey groups
    net = load_net(campaign)
    log_path = os.path.join(campaign, "perturbations.log")
    windows = parse_windows(open(log_path, encoding="utf-8").read().splitlines()) if os.path.exists(log_path) else []
    main = {(r["scheme"], int(r["sites"]), int(r["vector_size"])): r for r in summary if r["group"] == "main"}
    lines = ["# Campaign report", "",
             "Serial in-process simulator times for compute (not a distributed wall clock); network times "
             "are replays of the transfer ledger between agents. Percentiles of compute cells use all "
             "logged rounds; see the repetitions column. Cells overlapping a perturbation window are "
             "marked with ⚠ and listed in section 5.", ""]
    lines += ["## 1. Cost per round, three sites (group main)", ""]
    rows = []
    for r in sorted((r for r in summary if r["group"] == "main"), key=_key):
        rows.append([r["scheme"], r["vector_size"], r["repetitions"], f"{int(float(r['setup_bytes'])):,}",
                     f"{_f(r, 'bytes_p50') / 1e6:.2f}",
                     "/".join(f"{_f(r, f'crypto_s_p{p}'):.3f}" for p in (50, 95, 99)),
                     f"{_f(r, 'site_max_s_p50'):.3f}", f"{_f(r, 'reducer_s_p50'):.3f}", r["peak_host_rss_mb"]])
    lines += _table(["scheme", "elements", "rounds", "setup bytes", "round MB (p50)", "crypto s p50/p95/p99",
                     "site-max s p50", "reducer s p50", "peak RSS MB (whole process)"], rows)
    lines += ["", "## 2. Scalability in the number of sites (group scaling, compute only)", ""]
    rows = [[r["scheme"], r["sites"], r["vector_size"], f"{int(float(r['setup_bytes'])):,}", f"{_f(r, 'bytes_p50') / 1e6:.2f}",
             f"{_f(r, 'crypto_s_p50'):.3f}", f"{_f(r, 'cpu_total_s_p50'):.3f}", r["peak_host_rss_mb"]]
            for r in sorted((r for r in summary if r["group"] == "scaling"), key=_key)]
    lines += _table(["scheme", "sites", "elements", "setup bytes", "round MB (p50)", "crypto s p50",
                     "cpu total s p50", "peak RSS MB"], rows)
    lines += ["", "## 3. Network time per round (replay)", ""]
    rows, flagged = [], []
    for profile, group, cid, result, interval in sorted(net, key=lambda x: (x[0], x[1], _key({"scheme": x[3]["scheme"], "sites": len(x[3]["sites"]), "vector_size": x[3]["vector_size"]}))):
        n = len(result["sites"])
        payload = main.get((result["scheme"], n, result["vector_size"]))
        wire = result["wire_bytes"]["round"]
        tx = sum(a["tx_bytes"] for a in wire["actors"].values() if a["tx_bytes"] is not None) / wire["iterations"]
        ratio = f"{tx / _f(payload, 'bytes_p50'):.3f}" if payload else "n/a"
        samples = result["round_network_s"]
        p50, p95, p99 = _pct(samples)
        hits = overlaps(interval, windows) if interval else []
        if hits:
            flagged.append((f"net {profile} {group} {cid}", hits))
        rows.append([profile, group, result["scheme"], result["vector_size"], len(samples),
                     f"{p50:.3f}/{p95:.3f}/{p99:.3f}", f"{float(np.median(result['setup_network_s'])):.3f}", ratio,
                     "⚠" if hits else ""])
    lines += _table(["profile", "group", "scheme", "elements", "samples", "round net s p50/p95/p99",
                     "setup net s p50", "wire/payload", "perturbed"], rows)
    lines += ["", "wire/payload = interface transmit bytes per replay iteration (all actors, includes TCP/IP "
              "overhead and control traffic) over the payload bytes of one round.", ""]
    lines += ["## 4. Handshake cost (numerator of H6)", ""]
    rows = []
    net_setup = {(p, r["scheme"], r["vector_size"]): float(np.median(r["setup_network_s"]))
                 for p, g, c, r, _ in net if g == "main"}
    for r in sorted(handshake, key=lambda r: (r["rekey_every"], SCHEME_ORDER.get(r["scheme"], 9))):
        compute = _f(r, "time_s_p50")
        for profile in ("lan", "wan"):
            network = net_setup.get((profile, r["scheme"], 100000))
            total = compute + (network or 0.0)
            rows.append([r["scheme"], r["rekey_every"], r["handshakes"], f"{_f(r, 'bytes_p50'):,.0f}",
                         f"{compute:.4f}", profile, "n/a" if network is None else f"{network:.3f}",
                         f"{total:.3f}", f"{100 * total:.0f}"])
    lines += _table(["scheme", "rekey every", "handshakes", "bytes p50", "compute s p50", "profile",
                     "network s p50", "total s", "round time (s) above which it is < 1 %"], rows)
    lines += ["", "The denominator of H6 is the time of a full round including training, which is not measured "
              "here: the last column is the round time that would make the handshake cost under 1 %.", ""]
    lines += ["## 5. Perturbation windows and flagged cells", ""]
    lines += [f"- {a}-{b}: {label}" for a, b, label in windows] or ["- none recorded"]
    lines += [""] + ([f"- {cell}: overlaps {', '.join(hits)}" for cell, hits in flagged] or ["- no cell overlaps a window"])
    lines += ["", "## 6. Limits", "",
              _skipped_line(campaign),
              "- Heavy wan cells use fewer repetitions (see samples); their p95/p99 are coarse.",
              _load_line(campaign),
              "- Peak RSS is that of the whole simulator process, not per actor; GPU memory is not measured here.", ""]
    return "\n".join(lines)
