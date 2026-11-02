"""The secure schemes on training-sized updates: tiny elements (about 1e-4) must survive quantization."""
import json

import numpy as np
import pytest

from pqfl.train.aggregation import build_training_aggregator

SITES = ["S0", "S1", "S2"]


def updates(size=3000, scale=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    return {s: rng.uniform(-scale, scale, size) for s in SITES}


def check(scheme, tmp_path, atol):
    aggregator = build_training_aggregator(scheme, SITES, clip_bound=0.05, out_dir=str(tmp_path), seed=3)
    data = updates()
    total = aggregator.aggregate_round(0, data)
    expected = sum(data.values())
    assert np.max(np.abs(total - expected)) < atol
    return aggregator


@pytest.mark.requires_oqs
@pytest.mark.parametrize("scheme,atol", [("A", 3 * 2 ** -25 + 1e-12), ("C", 3 * 2 ** -25 + 1e-12)])
def test_masking_and_sharing_preserve_small_updates(scheme, atol, tmp_path):
    aggregator = check(scheme, tmp_path, atol)
    assert aggregator.stats().bytes_sent["reducer"] > 0


@pytest.mark.requires_oqs
@pytest.mark.requires_openfhe
@pytest.mark.parametrize("scheme,atol", [("B", 1e-9), ("B-BGV", 3 * 2 ** -21 + 1e-12)])
def test_homomorphic_schemes_preserve_small_updates(scheme, atol, tmp_path):
    check(scheme, tmp_path, atol)


@pytest.mark.requires_oqs
def test_plain_reference_and_logging_and_resume_log(tmp_path):
    aggregator = check("N", tmp_path, 1e-6)                         # float32 payload: relative 6e-8
    assert aggregator.security_status == "UNPROTECTED_BASELINE"
    rows = [json.loads(l) for l in (tmp_path / "aggregation.jsonl").read_text().splitlines()]
    assert rows and all(r["scheme"] == "N" and r["schema_version"] == 2 for r in rows)
    build_training_aggregator("N", SITES, clip_bound=0.05, out_dir=str(tmp_path), seed=3)   # resumed session
    assert (tmp_path / "aggregation-1.jsonl").exists()
