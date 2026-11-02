#!/usr/bin/env python3
"""Audit saved pilot artifacts without training, GPU use, or network access.

Run from any directory: python scripts/assess_pilots.py
Writes corrected per-run tables and an audit JSON under docs/pilot_review_2026-11-02/.
The primary comparison is explicitly restricted to matched four-round configurations;
batch-size sensitivity and ten-round runs never count as seed replications.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/pilot_review_2026-11-02"
SOURCES = set()


def read_json(path):
    path = ROOT / path
    SOURCES.add(path)
    return json.loads(path.read_text())


def read_jsonl(path):
    path = ROOT / path
    SOURCES.add(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(members, others):
    """Independent pairwise probability implementation, accounting for ties."""
    others = np.sort(others)
    lower = np.searchsorted(others, members, side="left")
    upper = np.searchsorted(others, members, side="right")
    return float(np.mean((lower + upper) / (2 * len(others))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCES.add(Path(__file__).resolve())
    rescored = {r["checkpoint"]: r for r in read_jsonl("runs/scoring-v2/scoring.jsonl")}
    references = {}
    for model, job in [("2B", "null-check-base"), ("0.8B", "q08b-null-check")]:
        path = Path(read_json(f"runs/gpu-queue/done/{job}.json")["args"]["base_nll_path"])
        if path.exists():
            SOURCES.add(path)
            with np.load(path, allow_pickle=False) as data:
                references[model] = {key: data[key].copy() for key in data.files}
    training, checks = [], []
    for model, directory in [("2B", "runs/pilot"), ("0.8B", "runs/pilot-q08b")]:
        for path in sorted((ROOT / directory).glob("*/summary.json")):
            summary = read_json(path)
            config = summary["config"]
            final_path = path.parent / "final.jsonl"
            final = read_jsonl(final_path)[-1]
            score = rescored.get(final["checkpoint"], final)
            score_path = ROOT / "runs/scoring-v2/scoring.jsonl" if score is not final else final_path
            primary = (config["rounds"] == 4 and config["logical_batch"] == 32
                       and config["clamp_bound"] == .02
                       and config["scheme"] in ("N", "A", "B", "B-BGV", "C"))
            if primary:
                assert config["micro_batch"] == 4 and config["lr"] == .0002
                assert config["seed"] == 0 and config["items_per_site"] == 2210
                assert len(config["sites"]) == 3
                assert config["clip"] == (None if config["epsilon"] is None else
                                          (4.0 if model == "2B" else 7.5))
                assert score["utility"]["validation"]["n"] == 4183
                assert score["utility"]["holdout"]["n"] == 2975
            loss = score["membership"]["pooled"]["loss"]
            top = score["canaries"]["levels"]["16"]
            row = dict(model=model, run=path.parent.name, primary=primary,
                       scheme=config["scheme"], epsilon=config["epsilon"], seed=config["seed"],
                       rounds=config["rounds"], logical_batch=config["logical_batch"],
                       clip=config["clip"], clamp_bound=config["clamp_bound"],
                       secret_randomness=config.get("secret_randomness", False),
                       accuracy=score["utility"]["validation"]["accuracy"],
                       perplexity=score["utility"]["holdout"]["perplexity"],
                       holdout_n=score["utility"]["holdout"]["n"],
                       mia_loss_auc=loss["auc"],
                       mia_reference_auc=score["membership"]["pooled"]["reference"]["auc"],
                       tpr_fpr_01=loss["tpr_at_fpr_0.01"],
                       tpr_fpr_001=loss.get("tpr_at_fpr_0.001"),
                       canary16_exposure=top["mean_exposure"],
                       canary16_extracted=round(top["extraction_rate"] * top["n"]),
                       canary16_n=top["n"], train_seconds=summary["train_seconds"],
                       score_source=str(score_path.relative_to(ROOT)), score_tag=score["tag"],
                       checkpoint=score["checkpoint"])
            training.append(row)
            if not primary:
                continue
            if score is not final:
                nll_path = ROOT / "runs/scoring-v2" / (score["tag"] + "-nll.npz")
            else:
                nll_path = path.parent / (score["tag"] + "-nll.npz")
            assert nll_path.exists(), nll_path
            SOURCES.add(nll_path)
            with np.load(nll_path, allow_pickle=False) as data:
                clean = ~data["nonmember_duplicate"]
                assert (len(data["member_nll"]), int(clean.sum())) == (6000, 2975)
                members, others = -data["member_nll"], -data["nonmember_nll"][clean]
                calculated = auc(members, others)
                assert abs(calculated - row["mia_loss_auc"]) < 1e-12
                item = dict(model=model, run=row["run"], loss_auc=calculated)
                if model in references:
                    ref = references[model]
                    assert np.array_equal(data["member_ids"], ref["member_ids"])
                    assert np.array_equal(data["nonmember_ids"], ref["nonmember_ids"])
                    reference_auc = auc(ref["member_nll"] + members,
                                        ref["nonmember_nll"][clean] + others)
                    assert abs(reference_auc - row["mia_reference_auc"]) < 1e-12
                    item["reference_auc"] = reference_auc
                for fpr in [.01, .001]:
                    threshold = np.sort(others)[-int(np.floor(fpr * len(others))) - 1]
                    tpr = float(np.mean(members > threshold))
                    assert abs(tpr - loss[f"tpr_at_fpr_{fpr:g}"]) < 1e-12
                    item[f"tpr_{fpr:g}"] = tpr
                    item[f"actual_fpr_{fpr:g}"] = float(np.mean(others > threshold))
                checks.append(item)
    primary = [r for r in training if r["primary"]]
    keys = [(r["model"], r["scheme"], r["epsilon"], r["seed"]) for r in primary]
    assert len(keys) == len(set(keys)) == 17
    write_csv("training.csv", training)

    cost_path = ROOT / "runs/campaign-4/compute/summary.csv"
    SOURCES.add(cost_path)
    costs = []
    for row in csv.DictReader(cost_path.open()):
        net = read_json(f"runs/campaign-4/net/wan/main__{row['scheme']}-n3-v8409600.json")
        costs.append(dict(scheme=row["scheme"], sites=int(row["sites"]),
                          vector_size=int(row["vector_size"]), bytes_per_round=int(float(row["bytes_p50"])),
                          compute_repetitions=int(row["repetitions"]),
                          crypto_s=float(row["crypto_s_p50"]),
                          peak_host_rss_mb=float(row["peak_host_rss_mb"]),
                          wan_repetitions=len(net["round_network_s"]),
                          wan_median_s=statistics.median(net["round_network_s"])))
    write_csv("costs.csv", costs)

    sessions = []
    for campaign in (3, 5, 6):
        for path in sorted((ROOT / f"runs/campaign-{campaign}/net/wan").glob("*.json")):
            data = read_json(path)
            if data["vector_size"] in (1000000, 3407872):
                sessions.append(dict(campaign=campaign, scheme=data["scheme"],
                                     vector_size=data["vector_size"],
                                     repetitions=len(data["round_network_s"]),
                                     median_s=statistics.median(data["round_network_s"])))
    write_csv("network_sessions.csv", sessions)
    states = {}
    for campaign in (3, 4, 5, 6):
        state = read_json(f"runs/campaign-{campaign}/state.json")
        counts = {}
        for cell in state["cells"].values():
            status = cell["status"]
            counts[status] = counts.get(status, 0) + 1
        assert set(counts) <= {"done", "skipped"}
        states[str(campaign)] = counts

    manifest = read_json("docs/pilot_manifest.json")
    integrity = {"checked": len(manifest["files"]), "missing": [], "changed": []}
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.exists():
            integrity["missing"].append(entry["path"])
        elif digest(path) != entry["sha256"]:
            integrity["changed"].append(entry["path"])
    # The optional split audit uses exported public data, if still available locally.
    job = read_json("runs/gpu-queue/done/null-check-base.json")["args"]
    split_audit = {"available": False}
    paths = {key: Path(job[key + "_path"]) for key in ("pool", "holdout", "validation")}
    if all(path.exists() for path in paths.values()):
        def text_key(row):
            return "|".join(re.sub(r"\W+", " ", str(row[k]).lower()).strip()
                            for k in ("question", "opa", "opb", "opc", "opd"))
        keys = {key: [text_key(row) for row in read_jsonl(path)] for key, path in paths.items()}
        key_sets = {key: set(values) for key, values in keys.items()}
        split_audit = dict(available=True,
                           sizes={key: len(values) for key, values in keys.items()},
                           internal_duplicates={key: len(values) - len(set(values))
                                                for key, values in keys.items()},
                           overlaps={f"{a}_in_{b}": sum(k in key_sets[b] for k in keys[a])
                                     for a, b in [("holdout", "pool"), ("validation", "pool"),
                                                  ("validation", "holdout")]})
    source_hashes = {}
    for path in sorted(SOURCES):
        # Avoid embedding machine-specific external scratch paths in this public-facing artifact.
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else "exported_data/" + path.name
        source_hashes[name] = digest(path)
    audit = dict(primary_runs=len(primary), all_training_runs=len(training),
                 independent_training_seeds=sorted({r["seed"] for r in primary}),
                 campaign_status=states, historical_manifest=integrity,
                 independently_recomputed_membership=checks, split_audit=split_audit,
                 source_sha256=source_hashes)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ["primary_runs", "all_training_runs",
                                         "campaign_status", "historical_manifest", "split_audit"]}, indent=2))
    print(f"Verified {len(checks)} primary runs against per-example NLL; output: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
