"""Aggregation of the training runs (pilot or final campaign) into per-configuration summaries and the
verdicts of hypotheses H3, H4 and H5 of the protocol.

Input: run directories holding summary.json (config) and final.jsonl (last record: utility, membership,
canaries). Intervals are the percentile bootstrap over seeds of pqfl.stats (protocol: no significance
tests); with five seeds they are coarse, and with one seed no interval exists (point values only).
Margins are those of the protocol: H3 within 1 accuracy point and 2 % relative perplexity.
"""
import json
import os

import numpy as np

from pqfl.stats import bootstrap_ci, intervals_overlap

ACCURACY_MARGIN = 0.01
PERPLEXITY_MARGIN = 0.02
SCHEMES = ("N", "A", "C", "B", "B-BGV")
LEVELS = ("none", "eps8", "eps2")
ATTACKS = {"mia_auc": "membership AUC (loss)", "tpr_1pct": "TPR at FPR 1 %", "canary16": "canary exposure, level 16",
           "extraction16": "extraction rate, level 16"}


def _level(epsilon) -> str:
    return "none" if epsilon is None else f"eps{epsilon:g}"


def load_runs(root: str, rounds: int | None = 4) -> list[dict]:
    """One row per finished run under `root` (directories with summary.json and final.jsonl)."""
    rows = []
    for name in sorted(os.listdir(root)):
        directory = os.path.join(root, name)
        summary_path, final_path = os.path.join(directory, "summary.json"), os.path.join(directory, "final.jsonl")
        if not (os.path.exists(summary_path) and os.path.exists(final_path)):
            continue
        config = json.load(open(summary_path, encoding="utf-8"))["config"]
        if rounds is not None and config["rounds"] != rounds:
            continue
        final = [json.loads(line) for line in open(final_path, encoding="utf-8")][-1]
        loss = final["membership"]["pooled"]["loss"]
        top = final["canaries"]["levels"].get("16", {})
        rows.append(dict(
            run=name, scheme=config["scheme"], level=_level(config["epsilon"]), seed=config["seed"],
            accuracy=final["utility"]["validation"]["accuracy"], perplexity=final["utility"]["holdout"]["perplexity"],
            mia_auc=loss["auc"], mia_ref_auc=final["membership"]["pooled"]["reference"]["auc"],
            tpr_1pct=loss.get("tpr_at_fpr_0.01"), tpr_01pct=loss.get("tpr_at_fpr_0.001"),
            canary16=top.get("mean_exposure"), extraction16=top.get("extraction_rate")))
    return rows


def _by_seed(rows, scheme, level, metric) -> dict:
    return {r["seed"]: r[metric] for r in rows if r["scheme"] == scheme and r["level"] == level
            and r[metric] is not None}


def interval(values):
    """(mean, low, high) over seeds; (mean, None, None) with a single seed."""
    values = list(values)
    if not values:
        return None
    if len(values) < 2:
        return float(values[0]), None, None
    return bootstrap_ci(values)


def paired(rows, scheme, reference, level, metric, relative=False):
    """Per-seed differences scheme - reference (optionally relative to the reference) and their interval."""
    a, b = _by_seed(rows, scheme, level, metric), _by_seed(rows, reference, level, metric)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    diffs = [(a[s] - b[s]) / (b[s] if relative else 1.0) for s in seeds]
    return dict(n=len(seeds), interval=interval(diffs))


def summarize(rows) -> list[dict]:
    """Mean and interval of every metric for each (scheme, level) present."""
    out = []
    for scheme in SCHEMES + ("Z",):
        for level in LEVELS:
            subset = [r for r in rows if r["scheme"] == scheme and r["level"] == level]
            if not subset:
                continue
            entry = dict(scheme=scheme, level=level, n=len(subset))
            for metric in ("accuracy", "perplexity", "mia_auc", "mia_ref_auc", "tpr_1pct", "tpr_01pct",
                           "canary16", "extraction16"):
                entry[metric] = interval([r[metric] for r in subset if r[metric] is not None])
            out.append(entry)
    return out


def h3(rows) -> list[dict]:
    """Without DP, each protected scheme against N: within 1 accuracy point and 2 % relative perplexity."""
    verdicts = []
    for scheme in ("A", "C", "B", "B-BGV"):
        accuracy = paired(rows, scheme, "N", "none", "accuracy")
        perplexity = paired(rows, scheme, "N", "none", "perplexity", relative=True)
        if not accuracy or not perplexity:
            continue
        def inside(result, margin):
            low, high = result["interval"][1:]
            return None if low is None else (-margin < low and high < margin)
        flags = (inside(accuracy, ACCURACY_MARGIN), inside(perplexity, PERPLEXITY_MARGIN))
        verdicts.append(dict(scheme=scheme, n=accuracy["n"], accuracy_diff=accuracy["interval"],
                             perplexity_rel_diff=perplexity["interval"],
                             verdict="n/a (one seed)" if None in flags else
                             ("equivalent" if all(flags) else "equivalence not shown")))
    return verdicts


def h4(rows) -> list[dict]:
    """Attack success against epsilon (none > eps8 > eps2 expected), interval separation none vs eps2."""
    verdicts = []
    for scheme in SCHEMES:
        for metric in ATTACKS:
            per_level = {level: interval(_by_seed(rows, scheme, level, metric).values()) for level in LEVELS}
            present = {k: v for k, v in per_level.items() if v}
            if len(present) < 2:
                continue
            means = [present[level][0] for level in LEVELS if level in present]
            monotone = all(x >= y for x, y in zip(means, means[1:]))
            separated = None
            if "none" in present and "eps2" in present and None not in (present["none"][1], present["eps2"][1]):
                separated = not intervals_overlap(present["none"], present["eps2"])
            verdicts.append(dict(scheme=scheme, metric=metric, per_level=per_level, monotone=monotone,
                                 separated_none_vs_eps2=separated))
    return verdicts


def h5(rows) -> list[dict]:
    """At each DP level, attack success of each protected scheme against N: does the paired interval contain 0?"""
    verdicts = []
    for level in LEVELS:
        for scheme in ("A", "C", "B", "B-BGV"):
            for metric in ATTACKS:
                result = paired(rows, scheme, "N", level, metric)
                if not result:
                    continue
                low, high = result["interval"][1:]
                verdicts.append(dict(level=level, scheme=scheme, metric=metric, n=result["n"],
                                     difference=result["interval"],
                                     contains_zero=None if low is None else low <= 0 <= high))
    return verdicts


def _fmt(value, digits=4):
    if value is None:
        return "-"
    mean, low, high = value
    return f"{mean:.{digits}f}" if low is None else f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def markdown(rows) -> str:
    seeds = sorted({r["seed"] for r in rows})
    lines = [f"# Training results ({len(rows)} runs, seeds {seeds})", "",
             "Intervals: percentile bootstrap over seeds, 95 %; none with a single seed.", "",
             "## Per configuration", "",
             "| Scheme | DP | n | Accuracy | Perplexity | MIA AUC | TPR 1 % | Canary 16 (bit) | Extraction 16 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in summarize(rows):
        lines.append(f"| {e['scheme']} | {e['level']} | {e['n']} | {_fmt(e['accuracy'])} | {_fmt(e['perplexity'], 3)} | "
                     f"{_fmt(e['mia_auc'])} | {_fmt(e['tpr_1pct'])} | {_fmt(e['canary16'], 2)} | "
                     f"{_fmt(e['extraction16'], 2)} |")
    lines += ["", "## H3: utility without DP against N (margins: 1 accuracy point, 2 % relative perplexity)", "",
              "| Scheme | n | Accuracy difference | Relative perplexity difference | Verdict |", "| --- | --- | --- | --- | --- |"]
    for v in h3(rows):
        lines.append(f"| {v['scheme']} | {v['n']} | {_fmt(v['accuracy_diff'])} | {_fmt(v['perplexity_rel_diff'])} | {v['verdict']} |")
    lines += ["", "## H4: attack success against epsilon", "",
              "| Scheme | Metric | none | eps8 | eps2 | Monotone | none vs eps2 separated |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for v in h4(rows):
        cells = " | ".join(_fmt(v["per_level"][level], 3) for level in LEVELS)
        lines.append(f"| {v['scheme']} | {ATTACKS[v['metric']]} | {cells} | {v['monotone']} | {v['separated_none_vs_eps2']} |")
    lines += ["", "## H5: attack success of each scheme against N at the same DP level", "",
              "| DP | Scheme | Metric | n | Difference | Interval contains 0 |", "| --- | --- | --- | --- | --- | --- |"]
    for v in h5(rows):
        lines.append(f"| {v['level']} | {v['scheme']} | {ATTACKS[v['metric']]} | {v['n']} | {_fmt(v['difference'])} | {v['contains_zero']} |")
    return "\n".join(lines) + "\n"
