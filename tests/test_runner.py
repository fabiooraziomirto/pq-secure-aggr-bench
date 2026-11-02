import json

import numpy as np
import pytest
import yaml

from pqfl.analysis import per_round_samples, percentiles
from pqfl.config import load_run_config
from pqfl.runner import run_cell, site_ids

CFG = dict(run_id="t", sites=["A", "B", "C"], clip_bound=1.0, scale_bits=24, modulus_bits=40,
           ring_dim=16384, scaling_mod_size=50, dp_level="none", run_seed=3,
           rekey_every_rounds=None, auth={"method": "none_insecure"})


def cfg(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(CFG))
    return load_run_config(path)


def test_site_ids_sorted_and_validated():
    ids = site_ids(12)
    assert ids == sorted(ids) and len(set(ids)) == 12
    with pytest.raises(ValueError):
        site_ids(1)


def test_warmup_rounds_are_not_logged_and_rows_are_per_round(tmp_path):
    out = tmp_path / "cell.jsonl"
    run_cell("STUB", cfg(tmp_path), sites=site_ids(4), vector_size=100, rounds=3, warmup=2, output=out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["round"] for r in rows if r["round"] is not None} == {2, 3, 4}
    setup, samples = per_round_samples(rows)
    assert len(samples) == 3
    # STUB: 4 sites upload 800 B each and the reducer broadcasts 800 B to each.
    assert all(s["bytes"] == 4 * 800 * 2 for s in samples)
    assert all(s["cpu_total_s"] >= s["site_max_s"] for s in samples)


def test_run_cell_refuses_existing_output_and_bad_counts(tmp_path):
    out = tmp_path / "cell.jsonl"
    run_cell("STUB", cfg(tmp_path), sites=site_ids(3), vector_size=10, rounds=1, warmup=0, output=out)
    with pytest.raises(FileExistsError):
        run_cell("STUB", cfg(tmp_path), sites=site_ids(3), vector_size=10, rounds=1, warmup=0, output=out)
    with pytest.raises(ValueError):
        run_cell("STUB", cfg(tmp_path), sites=site_ids(3), vector_size=10, rounds=0, warmup=0,
                 output=tmp_path / "x.jsonl")


def test_percentiles_linear():
    assert percentiles([1.0, 2.0, 3.0, 4.0], [50, 100]) == {"50": 2.5, "100": 4.0}
