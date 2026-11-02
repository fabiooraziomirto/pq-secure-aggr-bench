import json
import os

import numpy as np
import pytest

from pqfl.training_report import h3, h4, h5, load_runs, markdown, paired, summarize


def write_run(root, scheme, epsilon, seed, accuracy, perplexity, auc, canary, rounds=4):
    directory = os.path.join(root, f"{scheme}-{epsilon}-s{seed}")
    os.makedirs(directory)
    json.dump(dict(config=dict(scheme=scheme, epsilon=epsilon, seed=seed, rounds=rounds)),
              open(os.path.join(directory, "summary.json"), "w"))
    final = dict(utility=dict(validation=dict(accuracy=accuracy), holdout=dict(perplexity=perplexity)),
                 membership=dict(pooled=dict(loss=dict(auc=auc, **{"tpr_at_fpr_0.01": auc / 10, "tpr_at_fpr_0.001": None}),
                                             reference=dict(auc=auc))),
                 canaries=dict(levels={"16": dict(mean_exposure=canary, extraction_rate=canary / 10)}))
    open(os.path.join(directory, "final.jsonl"), "w").write(json.dumps(final) + "\n")


@pytest.fixture
def runs(tmp_path):
    rng = np.random.default_rng(0)
    for seed in range(1, 6):
        noise = rng.normal(0, 0.001)
        write_run(tmp_path, "N", None, seed, 0.507 + noise, 7.71 + noise, 0.63 + noise, 7.5 + noise)
        write_run(tmp_path, "A", None, seed, 0.508 + noise, 7.71 + noise, 0.63 + noise, 7.4 + noise)
        write_run(tmp_path, "B", None, seed, 0.40 + noise, 9.0 + noise, 0.63 + noise, 7.5 + noise)
        write_run(tmp_path, "N", 8.0, seed, 0.494 + noise, 9.3 + noise, 0.55 + noise, 3.0 + noise)
        write_run(tmp_path, "N", 2.0, seed, 0.491 + noise, 9.9 + noise, 0.50 + noise, 1.5 + noise)
    write_run(tmp_path, "N", None, 9, 0.5, 7.7, 0.6, 7.0, rounds=10)       # another protocol: ignored
    return load_runs(str(tmp_path))


def test_only_runs_of_the_requested_protocol_are_loaded(runs):
    assert len(runs) == 25 and {r["seed"] for r in runs} == {1, 2, 3, 4, 5}


def test_h3_separates_an_equivalent_scheme_from_a_degraded_one(runs):
    verdicts = {v["scheme"]: v["verdict"] for v in h3(runs)}
    assert verdicts == {"A": "equivalent", "B": "equivalence not shown"}


def test_h4_detects_a_monotone_decrease_with_separated_intervals(runs):
    entry = next(v for v in h4(runs) if v["scheme"] == "N" and v["metric"] == "mia_auc")
    assert entry["monotone"] is True and entry["separated_none_vs_eps2"] is True


def test_h5_pairs_by_seed(runs):
    entry = next(v for v in h5(runs) if v["scheme"] == "A" and v["level"] == "none" and v["metric"] == "canary16")
    assert entry["n"] == 5 and entry["difference"][0] == pytest.approx(-0.1, abs=1e-9)
    assert entry["contains_zero"] is False                      # constant -0.1 offset, tiny interval


def test_a_single_seed_gives_point_values_and_no_verdict(tmp_path):
    write_run(tmp_path, "N", None, 1, 0.5, 7.7, 0.63, 7.5)
    write_run(tmp_path, "A", None, 1, 0.51, 7.7, 0.63, 7.5)
    rows = load_runs(str(tmp_path))
    assert summarize(rows)[0]["accuracy"] == (0.5, None, None)
    assert h3(rows)[0]["verdict"] == "n/a (one seed)"
    assert "Training results (2 runs" in markdown(rows)
    assert paired(rows, "A", "N", "none", "accuracy")["n"] == 1
