import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from pqfl.aggregators.plain_fl import PlainFL
from pqfl.analysis import handshake_samples, per_round_samples
from pqfl.campaign import load_spec, repetitions_for
from pqfl.net.profiles import load_profiles

_spec = importlib.util.spec_from_file_location("rc", Path("scripts/run_campaign.py"))
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)
PROFILES = load_profiles("configs/network_profiles.yaml")


def test_reduced_spec_loads_and_plan_is_consistent():
    spec = load_spec("configs/campaign_reduced.yaml")
    cells = rc.build_plan(spec, PROFILES)
    keys = [c[0] for c in cells]
    assert len(keys) == len(set(keys))
    compute = {k for k in keys if k.startswith("compute:")}
    assert not any(k.startswith("compute:sensitivity") for k in compute)  # reuses main's ledgers
    for key, phase, est, reason, d in cells:
        if phase == "net" and not reason:
            assert f"compute:{d['ledger_group']}:{key.split(':')[3]}" in compute
    reasons = [c[3] for c in cells if c[3]]
    assert all(r for r in reasons) and any("spec" in r for r in reasons)
    heavy = [c for c in cells if c[1] == "compute" and c[4]["scheme"] == "B" and c[4]["size"] == 3407872]
    assert heavy and heavy[0][4]["reps"] < 30 and heavy[0][4]["warmup"] == 1


def test_repetitions_policy_shrinks_heavy_cells_but_never_below_minimum():
    policy = dict(repetitions=30, warmup=3, min_repetitions=3, cell_budget_s=1200, net_cap_s=3600)
    assert repetitions_for(1.0, policy) == (30, 3)
    assert repetitions_for(100.0, policy) == (12, 1)
    assert repetitions_for(5000.0, policy) == (3, 1)


def test_spec_validation(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text(yaml.safe_dump(dict(
        policy=dict(repetitions=30, warmup=3, min_repetitions=3, cell_budget_s=1, net_cap_s=1),
        percentiles=[50], groups=[dict(name="g", schemes=["A"], sites=[3], vector_sizes=[10],
                                       compute=False)])))
    with pytest.raises(ValueError, match="ledger_from"):
        load_spec(bad)


def test_plain_fl_n_costs_four_bytes_per_element():
    agg = PlainFL()
    agg.setup(["a", "b", "c"])
    out = agg.aggregate_round(0, {s: np.full(1000, 0.5) for s in "abc"})
    np.testing.assert_allclose(out, 1.5, atol=1e-6)
    assert sum(agg.stats().bytes_sent.values()) == (3 + 3) * 4000
    assert agg.security_status == "UNPROTECTED_BASELINE"


def _row(round_idx, epoch, phase, seconds, sent):
    return dict(round=round_idx, epoch=epoch, phase=phase, site="a", duration_s=seconds,
                bytes_sent={"a": sent}, bytes_received={"a": 0})


def test_handshake_samples_separate_setup_from_rekeys():
    rows = [_row(None, 0, "crypto", 1.0, 100), _row(None, 1, "crypto", 0.5, 40), _row(None, 1, "auth", 0.25, 10),
            _row(None, 2, "crypto", 0.75, 60), _row(5, 2, "compute", 0.1, 7)]
    setup, samples = per_round_samples(rows)
    assert setup == dict(bytes=100, time_s=1.0) and len(samples) == 1
    handshakes = handshake_samples(rows)
    assert [h["bytes"] for h in handshakes] == [50, 60]
    assert handshakes[0]["time_s"] == 0.75 and handshakes[0]["auth_s"] == 0.25



def test_spec_rejects_fractional_counts(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text(yaml.safe_dump(dict(
        policy=dict(repetitions=5.5, warmup=3, min_repetitions=3, cell_budget_s=1, net_cap_s=1),
        percentiles=[50], groups=[dict(name="g", schemes=["A"], sites=[3], vector_sizes=[10])])))
    with pytest.raises(ValueError, match="integer"):
        load_spec(bad)


def test_empty_stage_costs_no_network_time():
    from pqfl.campaign import ledger_net_estimate_s
    wan = PROFILES["wan"]
    stages = [[("A", "reducer", 1000)]]
    assert ledger_net_estimate_s(stages + [[]], wan) == ledger_net_estimate_s(stages, wan)


def test_matched_n_costs_five_bytes_per_element_and_sums_in_fixed_point():
    from pqfl.aggregators.plain_fl import PlainFixedPoint
    from pqfl.campaign import stage_flows
    sites = ["a", "b", "c"]
    agg = PlainFixedPoint(scale_bits=24, modulus_bits=40)
    agg.setup(sites)
    updates = {s: np.linspace(-0.02, 0.02, 1000) * (i + 1) for i, s in enumerate(sites)}
    total = agg.aggregate_round(0, updates)
    np.testing.assert_allclose(total, sum(updates.values()), atol=3 * 2 ** -24)
    assert sum(agg.stats().bytes_sent.values()) == (3 + 3) * 1000 * 5
    assert stage_flows("N-FP", 3, 1000) == stage_flows("A", 3, 1000)
    assert agg.security_status == "UNPROTECTED_BASELINE"
    with pytest.raises(ValueError):
        PlainFixedPoint(scale_bits=40, modulus_bits=40)
