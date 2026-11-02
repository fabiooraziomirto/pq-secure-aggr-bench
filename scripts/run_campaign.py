#!/usr/bin/env python3
"""Measurement campaign from a spec file: compute cells (one fresh subprocess each, for a
clean peak-memory reading), then network replay through the Docker compose per profile.
Opt-in; takes hours. State in <output-dir>/state.json (resume with --resume; progress with
scripts/campaign_status.py). Skipped cells are recorded with their reason.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from pqfl.analysis import handshake_samples, load_rows, percentiles, summary_row
from pqfl.campaign import (State, cell_id, compute_estimate_s, eta, load_spec, net_rep_estimate_s,
                           repetitions_for)
from pqfl.config import SCHEMES
from pqfl.net.compose import agent_arguments, campaign_compose
from pqfl.net.profiles import load_profiles
from pqfl.runner import site_ids

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
COMPOSE_PROJECT = "pqfl-campaign"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True, help="Directory under runs/")
    parser.add_argument("--profiles-file", default="configs/network_profiles.yaml")
    parser.add_argument("--image", default="pqfl-crypto")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_plan(spec, profiles):
    """Cells as (key, phase, est_s, skip_reason, detail); repetitions fixed here."""
    policy, cells = spec["policy"], []
    for group in spec["groups"]:
        for scheme in group["schemes"]:
            if scheme not in SCHEMES:
                raise ValueError(f"unknown scheme {scheme}")
            for n in group["sites"]:
                for size in group["vector_sizes"]:
                    cid = cell_id(scheme, n, size)
                    skip = f"{scheme}:{n}:{size}" in group["skip"]
                    est_round = compute_estimate_s(scheme, n, size, 1)
                    reps, warmup = repetitions_for(est_round, policy)
                    if group["compute"]:
                        cells.append((f"compute:{group['name']}:{cid}", "compute",
                                      compute_estimate_s(scheme, n, size, reps + warmup) + 5,  # +5: process start
                                      "skipped in spec" if skip else None,
                                      dict(group=group["name"], scheme=scheme, n=n, size=size, reps=reps,
                                           warmup=warmup, rekey=group["rekey_every_rounds"])))
                    for profile in group["profiles"]:
                        est_rep = net_rep_estimate_s(scheme, n, size, profiles[profile])
                        reps, warmup = repetitions_for(est_rep, policy)
                        reason = "skipped in spec" if skip else None
                        if not reason and est_rep * (policy["min_repetitions"] + 1) > policy["net_cap_s"]:
                            reason = f"estimated {est_rep:.0f}s per round exceeds the cap"
                        cells.append((f"net:{group['name']}:{profile}:{cid}", "net",
                                      est_rep * (reps + warmup) + 5, reason,
                                      dict(group=group["name"], ledger_group=group["ledger_from"],
                                           scheme=scheme, n=n, size=size,
                                           profile=profile, reps=reps, warmup=warmup,
                                           est_rep_s=est_rep)))
    return cells


def fmt(seconds):
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"


def print_plan(cells):
    for phase in ("compute", "net"):
        group = [c for c in cells if c[1] == phase]
        run = [c for c in group if not c[3]]
        reduced = [c for c in run if c[4]["reps"] < 30]
        print(f"{phase}: {len(run)} to run ({len(reduced)} with reduced repetitions), "
              f"{len(group) - len(run)} skipped, estimated {fmt(sum(c[2] for c in run))}")
    for key, _, _, reason, _ in cells:
        if reason and key.startswith("net:"):
            print(f"  skipped {key}: {reason}")


def loadavg():
    """System load at a cell boundary (1 and 5 minute averages), for later filtering."""
    with open("/proc/loadavg", encoding="ascii") as handle:
        one, five = handle.read().split()[:2]
    return [float(one), float(five)]


def write_row(path, row):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_compute(args, spec, state, out):
    (out / "compute").mkdir(exist_ok=True)
    points = spec["percentiles"]
    for key, cell in list(state.data["cells"].items()):
        if cell["phase"] != "compute" or cell["status"] in ("done", "skipped"):
            continue
        d = cell["detail"]
        base = out / "compute" / f"{d['group']}__{cell_id(d['scheme'], d['n'], d['size'])}"
        partial = {s: Path(f"{base}.partial{s}") for s in (".jsonl", "-ledger.json", "-mem.json")}
        for path in partial.values():
            path.unlink(missing_ok=True)
        command = [sys.executable, str(REPO / "scripts/run_experiment.py"), "--config", args.config,
                   "--scheme", d["scheme"], "--output", str(partial[".jsonl"]),
                   "--rounds", str(d["reps"]), "--warmup", str(d["warmup"]),
                   "--vector-size", str(d["size"]), "--sites-count", str(d["n"]),
                   "--ledger-output", str(partial["-ledger.json"]),
                   "--peak-memory-output", str(partial["-mem.json"])]
        if d["rekey"]:
            command += ["--rekey-every", str(d["rekey"])]
        state.update(key, status="running", load_start=loadavg())
        start = time.time()
        try:
            done = subprocess.run(command, capture_output=True, text=True,
                                  timeout=max(1800, 6 * cell["est_s"]))
            if done.returncode != 0:
                raise RuntimeError(done.stderr[-400:])
            for suffix in (".jsonl", "-ledger.json", "-mem.json"):
                os.replace(partial[suffix], f"{base}{suffix}")
            row = summary_row(d["scheme"], d["n"], d["size"], d["reps"], d["warmup"],
                              f"{base}.jsonl", points)
            peak = json.load(open(f"{base}-mem.json", encoding="utf-8"))["peak_host_rss_mb"]
            write_row(out / "compute" / "summary.csv",
                      dict(group=d["group"], peak_host_rss_mb=f"{peak:.1f}", **row))
            if d["rekey"]:
                samples = handshake_samples(load_rows(f"{base}.jsonl"))
                hs = dict(group=d["group"], scheme=d["scheme"], sites=d["n"], rekey_every=d["rekey"],
                          handshakes=len(samples))
                for metric in ("bytes", "time_s", "crypto_s", "auth_s"):
                    for point, value in percentiles([s[metric] for s in samples], points).items():
                        hs[f"{metric}_p{float(point):g}"] = f"{value:.6f}"
                write_row(out / "compute" / "handshake.csv", hs)
            state.update(key, status="done", seconds=time.time() - start, load_end=loadavg())
        except Exception as exc:  # recorded; the campaign continues with other cells
            state.update(key, status="failed", seconds=time.time() - start, error=repr(exc)[:500])


def compose_cmd(out, *extra):
    return ["docker", "compose", "-p", COMPOSE_PROJECT, "-f", str(out / "compose.yaml"), *extra]


def run_net(args, spec, state, out):
    max_n = max(n for g in spec["groups"] if g["profiles"] for n in g["sites"])
    (out / "compose.yaml").write_text(yaml.safe_dump(campaign_compose(max_n, image=args.image)))
    profiles = sorted({c["detail"]["profile"] for c in state.data["cells"].values() if c["phase"] == "net"})
    for profile in profiles:
        pending = [(k, c) for k, c in state.data["cells"].items()
                   if c["phase"] == "net" and c["status"] == "pending" and c["detail"]["profile"] == profile]
        if not pending:
            continue
        env = dict(os.environ, NETEM_PROFILE=profile)
        subprocess.run(compose_cmd(out, "up", "-d"), env=env, check=True, capture_output=True)
        time.sleep(10)
        try:
            for key, cell in sorted(pending, key=lambda kc: (kc[1]["detail"]["size"], kc[1]["detail"]["n"],
                                                             kc[1]["detail"]["scheme"])):
                d = cell["detail"]
                cid = cell_id(d["scheme"], d["n"], d["size"])
                ledger = out / "compute" / f"{d['ledger_group']}__{cid}-ledger.json"
                if not ledger.exists():
                    state.update(key, status="skipped", detail=dict(d, reason="no ledger (compute failed)"))
                    continue
                result_dir = out / "net" / profile
                result_dir.mkdir(parents=True, exist_ok=True)
                result = result_dir / f"{d['group']}__{cid}.json"
                actors = [*site_ids(d["n"]), "reducer"]
                command = compose_cmd(out, "run", "--rm", "-T", "--no-deps", "controller",
                                      "python", "scripts/net_replay.py",
                                      "--ledger", f"/ledgers/{out.name}/compute/{ledger.name}",
                                      "--profile", profile, "--repetitions", str(d["reps"]),
                                      "--warmup", str(d["warmup"]),
                                      "--output", f"/ledgers/{out.name}/net/{profile}/{result.name}",
                                      "--agents", *agent_arguments(actors))
                state.update(key, status="running", load_start=loadavg())
                start = time.time()
                try:
                    for attempt in range(3):  # agents may still be starting
                        done = subprocess.run(command, env=env, capture_output=True, text=True,
                                              timeout=max(900, 4 * cell["est_s"]))
                        if done.returncode == 0:
                            break
                        result.unlink(missing_ok=True)
                        time.sleep(10)
                    if done.returncode != 0:
                        raise RuntimeError(done.stderr[-400:])
                    state.update(key, status="done", seconds=time.time() - start, load_end=loadavg())
                except Exception as exc:
                    state.update(key, status="failed", seconds=time.time() - start, error=repr(exc)[:500])
        finally:
            subprocess.run(compose_cmd(out, "down", "-t", "2"), env=env, capture_output=True)


def main():
    args = parse_args()
    spec = load_spec(args.spec)
    profiles_all = load_profiles(args.profiles_file)
    used = {p for g in spec["groups"] for p in g["profiles"]}
    if used - set(profiles_all):
        sys.exit(f"Unknown profiles: {sorted(used - set(profiles_all))}")
    plan = build_plan(spec, profiles_all)
    print_plan(plan)
    if args.plan_only:
        return
    out = Path(args.output_dir).resolve()
    if out.parent != RUNS.resolve():
        sys.exit("--output-dir must be a direct subdirectory of runs/")
    if out.exists() and not args.resume:
        sys.exit("Output directory exists; use --resume to continue it")
    out.mkdir(exist_ok=True)
    state_path = out / "state.json"
    state = State.load(state_path) if args.resume and state_path.exists() else State(state_path)
    for key, phase, est_s, reason, detail in plan:
        state.add(key, phase, est_s, detail)
        if reason and state.data["cells"][key]["status"] == "pending":
            state.data["cells"][key]["status"] = "skipped"
            state.data["cells"][key]["detail"]["reason"] = reason
    for stale in state.data["cells"].values():
        if stale["status"] == "running":  # interrupted previously
            stale["status"] = "pending"
    state.save()
    (out / "spec.yaml").write_text(Path(args.spec).read_text())
    run_compute(args, spec, state, out)
    run_net(args, spec, state, out)
    print(json.dumps(eta(state.data)["phases"]))


if __name__ == "__main__":
    main()
