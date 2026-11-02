"""Campaign planning: cells, analytic cost estimates and a resumable state file.

Estimates are rough analytic models used for skip rules and ETAs; real runs
record actual times, and the ETA is rescaled by observed/estimated ratios.
"""
import json
import math
import os
import time

ROUNDS_OVERHEAD = 1.0
CT_BYTES, SUM_BYTES, PARTIAL_BYTES, SLOTS = 788_000, 850_000, 394_000, 8192
FIXED_POINT_BYTES = 5  # modulus_bits=40
BGV_CT_BYTES, BGV_PARTIAL_BYTES, BGV_SLOTS = 1_311_945, 656_560, 16384


def compute_estimate_s(scheme: str, n: int, size: int, rounds: int) -> float:
    """Wall seconds for one compute cell (setup plus `rounds` rounds)."""
    if scheme in ("B", "B-BGV"):
        return n * 0.1 + rounds * 8e-6 * n * size
    if scheme == "C":
        return n * n * 0.0005 + rounds * (1e-7 * n * n * size + 2.5e-7 * n * size)
    if scheme == "A":
        return n * n * 0.0005 + rounds * 2.5e-7 * n * size
    if scheme == "N-FP":
        return rounds * 2.5e-7 * n * size   # quantize, pack, sum, unpack: A without masks
    return rounds * 5e-8 * n * size  # N and STUB


def stage_egress_bytes(scheme: str, n: int, size: int) -> list[float]:
    """Largest single-sender egress per stage of one round (bytes)."""
    word = FIXED_POINT_BYTES * size
    if scheme == "A":
        return [word, n * word]
    if scheme == "C":
        return [(n - 1) * (word + 16), word, n * word]
    if scheme == "B-BGV":
        chunks = math.ceil(size / BGV_SLOTS)
        return [chunks * BGV_CT_BYTES, n * chunks * BGV_CT_BYTES, chunks * BGV_PARTIAL_BYTES, n * 8 * size]
    if scheme == "B":
        chunks = math.ceil(size / SLOTS)
        return [chunks * CT_BYTES, n * chunks * SUM_BYTES, chunks * PARTIAL_BYTES, n * 8 * size]
    word = FIXED_POINT_BYTES if scheme == "N-FP" else 4 if scheme == "N" else 8
    return [word * size, n * word * size]


MSS_BYTES = 1448
MATHIS_C = 1.65   # fitted on the 18 wan cells of campaign 3 (rms log error 0.14); see docs/decisions.md


def stage_flows(scheme: str, n: int, size: int) -> list[tuple[int, int]]:
    """(bytes of one flow, concurrent flows leaving the same sender) for each stage of a round."""
    word = FIXED_POINT_BYTES * size
    if scheme == "N":
        return [(4 * size, 1), (4 * size, n)]
    if scheme in ("A", "N-FP"):
        return [(word, 1), (word, n)]
    if scheme == "C":
        return [(word + 16, n - 1), (word, 1), (word, n)]
    if scheme == "B":
        chunks = math.ceil(size / SLOTS)
        return [(chunks * CT_BYTES, 1), (chunks * SUM_BYTES, n), (chunks * PARTIAL_BYTES, 1), (8 * size, n)]
    if scheme == "B-BGV":
        chunks = math.ceil(size / BGV_SLOTS)
        return [(chunks * BGV_CT_BYTES, 1), (chunks * BGV_CT_BYTES, n),
                (chunks * BGV_PARTIAL_BYTES, 1), (8 * size, n)]
    return [(8 * size, 1), (8 * size, n)]


def net_rep_estimate_s(scheme, n, size, profile: dict) -> float:
    """Seconds for one replayed round. Each stage takes its largest flow over the flow rate plus
    ~6 RTT (TCP slow start, measured). The flow rate is the link rate shared by the flows leaving
    one sender, capped when the profile has loss by the Mathis limit C * MSS / (RTT * sqrt(p)):
    on wan the per-flow loss limit (about 1.4 MB/s), not the 100 Mbit/s link, decides the time.
    `lan` assumes 5 Gbit/s. Validated on lan, metro (no loss) and wan; the constrained profile
    (0.5 % loss) is an extrapolation."""
    rate = (profile["rate_mbit"] if profile["rate_mbit"] is not None else 5000) * 125_000
    rtt = profile["rtt_ms"] / 1000
    loss = profile["loss_pct"] / 100
    cap = math.inf if loss == 0 or rtt == 0 else MATHIS_C * MSS_BYTES / (rtt * math.sqrt(loss))
    return sum(flow_bytes / min(rate / flows, cap) + 6 * rtt + 0.005
               for flow_bytes, flows in stage_flows(scheme, n, size))


def ledger_net_estimate_s(stages, profile: dict) -> float:
    """Modelled network time of a staged ledger (list of stages of (sender, receiver, size)),
    with the per-flow model of `net_rep_estimate_s`: flows leaving one sender share its link."""
    rate = (profile["rate_mbit"] if profile["rate_mbit"] is not None else 5000) * 125_000
    rtt = profile["rtt_ms"] / 1000
    loss = profile["loss_pct"] / 100
    cap = math.inf if loss == 0 or rtt == 0 else MATHIS_C * MSS_BYTES / (rtt * math.sqrt(loss))
    total = 0.0
    for stage in stages:
        if not stage:                       # a stage without flows costs no network time
            continue
        per_sender = {}
        for sender, _, size in stage:
            per_sender[sender] = per_sender.get(sender, 0) + 1
        slowest = max(size / min(rate / per_sender[sender], cap) for sender, _, size in stage)
        biggest = max(size for _, _, size in stage)
        # Round trips of slow start: about 1 for a tiny message, up to 6 for hundreds of KB (the value
        # calibrated in net_rep_estimate_s), initial window 10 segments.
        total += slowest + min(6.0, 1.0 + math.log2(1 + biggest / (10 * MSS_BYTES))) * rtt + 0.005
    return total


def cell_id(scheme, n, size) -> str:
    return f"{scheme}-n{n}-v{size}"


class State:
    """JSON state, rewritten atomically after every change so it is always readable."""

    def __init__(self, path):
        self.path = path
        self.data = {"started": time.time(), "cells": {}}

    @classmethod
    def load(cls, path):
        state = cls(path)
        with open(path, encoding="utf-8") as handle:
            state.data = json.load(handle)
        return state

    def add(self, key, phase, est_s, detail=None):
        self.data["cells"].setdefault(key, {"phase": phase, "est_s": est_s, "status": "pending",
                                            "seconds": None, "detail": detail or {}})

    def update(self, key, **fields):
        self.data["cells"][key].update(fields)
        self.save()

    def save(self):
        temporary = f"{self.path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle)
        os.replace(temporary, self.path)


def eta(data: dict, now: float | None = None) -> dict:
    """Remaining-time estimate, rescaled per phase by observed/estimated ratios."""
    now = time.time() if now is None else now
    cells = data["cells"].values()
    totals = {}
    for phase in {c["phase"] for c in cells}:
        group = [c for c in cells if c["phase"] == phase]
        done = [c for c in group if c["status"] == "done" and c["seconds"] and c["est_s"]]
        ratio = (sum(c["seconds"] for c in done) / sum(c["est_s"] for c in done)) if done else 1.0
        remaining = sum(c["est_s"] for c in group if c["status"] == "pending") * ratio
        totals[phase] = dict(total=len(group), done=sum(c["status"] == "done" for c in group),
                             skipped=sum(c["status"] == "skipped" for c in group),
                             failed=sum(c["status"] == "failed" for c in group),
                             ratio=ratio, remaining_s=remaining)
    return dict(phases=totals, elapsed_s=now - data["started"],
                remaining_s=sum(p["remaining_s"] for p in totals.values()))


def load_spec(path) -> dict:
    """Campaign spec: policy, percentiles and groups of cells (see configs/campaign_reduced.yaml)."""
    import yaml
    with open(path, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    policy = spec["policy"]
    for key in ("repetitions", "warmup", "min_repetitions", "cell_budget_s", "net_cap_s"):
        if key not in policy or isinstance(policy[key], bool) or not isinstance(policy[key], (int, float)) \
                or policy[key] < 0:
            raise ValueError(f"policy.{key} must be a nonnegative number")
    for key in ("repetitions", "warmup", "min_repetitions"):
        if type(policy[key]) is not int:
            raise ValueError(f"policy.{key} must be an integer")
    if not 1 <= policy["min_repetitions"] <= policy["repetitions"]:
        raise ValueError("min_repetitions must lie in [1, repetitions]")
    names = [g["name"] for g in spec["groups"]]
    if len(set(names)) != len(names):
        raise ValueError("Group names must be unique")
    for group in spec["groups"]:
        for key in ("schemes", "sites", "vector_sizes"):
            if not group.get(key):
                raise ValueError(f"group {group['name']}: {key} must be a nonempty list")
        group.setdefault("profiles", [])
        group.setdefault("skip", [])
        group.setdefault("rekey_every_rounds", None)
        group.setdefault("compute", True)
        group.setdefault("ledger_from", group["name"])
        if not group["compute"] and group["ledger_from"] == group["name"]:
            raise ValueError(f"group {group['name']}: compute false needs ledger_from another group")
    return spec


def repetitions_for(est_per_rep_s: float, policy: dict) -> tuple[int, int]:
    """(repetitions, warmup): full counts for cheap cells; heavy cells shrink to fit the
    per-cell budget (never below min_repetitions, warm-up 1). Bytes are deterministic;
    only timing statistics get wider."""
    full = policy["repetitions"]
    fit = int(policy["cell_budget_s"] / max(est_per_rep_s, 1e-9))
    reps = max(policy["min_repetitions"], min(full, fit))
    return reps, (policy["warmup"] if reps == full else min(1, policy["warmup"]))
