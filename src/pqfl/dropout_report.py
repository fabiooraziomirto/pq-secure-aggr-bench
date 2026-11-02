"""Summaries of bench_dropout results: cost of recovering from a crashed site.

In each session the round after the crash has the same participants as the crash round but
needs no recovery, so (crash round - next round) isolates the recovery work and traffic.
Network times are MODELLED from the staged ledgers (per-flow model, not replayed); the
detection timeout is not included.
"""
import json

import numpy as np

from pqfl.campaign import ledger_net_estimate_s


def summarize(path: str, profiles: dict[str, dict]) -> dict:
    result = json.load(open(path, encoding="utf-8"))
    rounds = result["rounds"]
    paired = lambda key: [d[key] - n[key] for d, n in zip(rounds["dropout"], rounds["next"])]
    med = lambda values: float(np.median(values))
    out = dict(scheme=result["scheme"], dropped=result["dropped"], lead_dropped=result["dropped"] == result["sites"][0],
               vector_size=result["vector_size"], sessions=result["sessions"], threshold=result["threshold"],
               setup_bytes=med([s["bytes"] for s in result["setup"]]), setup_s=med([s["time_s"] for s in result["setup"]]),
               stages={k: len(v) for k, v in result["ledgers"].items()})
    for name in ("normal", "dropout", "next"):
        out[f"{name}_bytes"] = med([r["bytes"] for r in rounds[name]])
        out[f"{name}_cpu_s"] = med([r["cpu_total_s"] for r in rounds[name]])
    out["recovery_extra_bytes"] = med(paired("bytes"))
    out["recovery_extra_cpu_s"] = med(paired("cpu_total_s"))
    out["recovery_extra_cpu_s_spread"] = [float(np.percentile(paired("cpu_total_s"), q)) for q in (5, 95)]
    for label, profile in profiles.items():
        out[f"net_{label}_dropout_round_s"] = ledger_net_estimate_s(result["ledgers"]["dropout"], profile)
        out[f"net_{label}_next_round_s"] = ledger_net_estimate_s(result["ledgers"]["next"], profile)
        out[f"net_{label}_recovery_extra_s"] = out[f"net_{label}_dropout_round_s"] - out[f"net_{label}_next_round_s"]
    return out


def markdown(summaries: list[dict], profiles: list[str]) -> str:
    head = ["scheme", "dropped", "elements", "sessions", "setup MB", "round MB normal/crash/next", "stages n/c/x",
            "extra bytes (crash-next)", "extra cpu s p50 [p5,p95]"] + [f"modelled extra net s {p}" for p in profiles]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for s in sorted(summaries, key=lambda s: (s["scheme"], s["vector_size"], not s["lead_dropped"])):
        lo, hi = s["recovery_extra_cpu_s_spread"]
        lines.append("| " + " | ".join([
            s["scheme"], "lead" if s["lead_dropped"] else "non-lead", str(s["vector_size"]), str(s["sessions"]),
            f"{s['setup_bytes'] / 1e6:.2f}",
            f"{s['normal_bytes'] / 1e6:.2f}/{s['dropout_bytes'] / 1e6:.2f}/{s['next_bytes'] / 1e6:.2f}",
            f"{s['stages']['normal']}/{s['stages']['dropout']}/{s['stages']['next']}",
            f"{s['recovery_extra_bytes']:,.0f}", f"{s['recovery_extra_cpu_s']:.3f} [{lo:.3f},{hi:.3f}]"]
            + [f"{s[f'net_{p}_recovery_extra_s']:.2f}" for p in profiles]) + " |")
    return "\n".join(lines)
