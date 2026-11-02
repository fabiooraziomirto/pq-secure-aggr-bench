"""Registration of the dropout schemes: registry, config, runner (the quorum policy is not all_or_abort)."""
import json

import numpy as np
import pytest
import yaml

from pqfl.aggregators import AGGREGATOR_REGISTRY, SchemeA, SchemeADropout, SchemeBGV, SchemeBGVThreshold
from pqfl.config import SCHEMES, load_run_config
from pqfl.runner import run_cell, site_ids

CFG = dict(run_id="t", sites=["A", "B", "C"], clip_bound=1.0, scale_bits=24, modulus_bits=40, ring_dim=16384,
           scaling_mod_size=50, dp_level="none", run_seed=3, rekey_every_rounds=None, auth={"method": "none_insecure"})


def config(tmp_path, **changes):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({**CFG, **changes}))
    return load_run_config(path)


def test_registry_and_schemes_include_the_dropout_schemes():
    assert AGGREGATOR_REGISTRY["A-dropout"] is SchemeADropout and AGGREGATOR_REGISTRY["B-BGV-threshold"] is SchemeBGVThreshold
    assert issubclass(SchemeADropout, SchemeA) and issubclass(SchemeBGVThreshold, SchemeBGV)
    assert {"A-dropout", "B-BGV-threshold"} <= set(SCHEMES)
    # Deliberately not the all_or_abort contract: they are tested in test_dropout.py instead.
    assert SchemeADropout.participation_policy == SchemeBGVThreshold.participation_policy == "quorum"
    assert SchemeA.participation_policy == "all_or_abort"


def test_dropout_threshold_config_is_validated(tmp_path):
    assert config(tmp_path).dropout_threshold is None
    assert config(tmp_path, dropout_threshold=2).dropout_threshold == 2
    for bad in (1, 0, True, "2", 2.5):
        with pytest.raises(ValueError, match="dropout_threshold"):
            config(tmp_path, dropout_threshold=bad)


@pytest.mark.requires_oqs
def test_runner_runs_a_dropout_scheme_without_a_drop(tmp_path):
    out = tmp_path / "a.jsonl"
    run_cell("A-dropout", config(tmp_path), sites=site_ids(3), vector_size=500, rounds=2, warmup=1, output=out)
    assert {json.loads(line)["scheme"] for line in out.read_text().splitlines()} == {"A-dropout"}


@pytest.mark.requires_oqs
@pytest.mark.requires_openfhe
def test_runner_builds_the_threshold_scheme_with_a_default_that_tolerates_one_dropout(tmp_path):
    cfg = config(tmp_path)
    out = tmp_path / "b.jsonl"
    run_cell("B-BGV-threshold", cfg, sites=site_ids(3), vector_size=500, rounds=2, warmup=1, output=out)
    from pqfl.runner import build_aggregator
    agg = build_aggregator("B-BGV-threshold", cfg, site_ids(4), None)
    assert agg.threshold == 3                                   # max(2, n - 1)
    assert build_aggregator("B-BGV-threshold", config(tmp_path, dropout_threshold=2), site_ids(4), None).threshold == 2
