import json

import pytest

from pqfl.campaign import ledger_net_estimate_s
from pqfl.dropout_report import markdown, summarize
from pqfl.net.profiles import load_profiles

PROFILES = load_profiles("configs/network_profiles.yaml")


def test_ledger_model_shares_a_sender_link_and_is_loss_limited_on_wan():
    lan = dict(PROFILES["lan"], rtt_ms=0)
    one = ledger_net_estimate_s([[("a", "r", 1_000_000_000)]], lan)
    two = ledger_net_estimate_s([[("a", "r", 1_000_000_000), ("a", "s", 1_000_000_000)]], lan)
    assert two == pytest.approx(2 * one - 0.005, rel=0.02)                # two flows share one sender link
    spread = ledger_net_estimate_s([[("a", "r", 1_000_000_000), ("b", "s", 1_000_000_000)]], lan)
    assert spread == pytest.approx(one, rel=0.02)                          # different senders do not
    wan = ledger_net_estimate_s([[("a", "r", 10_000_000)]], PROFILES["wan"])
    assert 6 < wan < 9                                                     # ~1.4 MB/s per flow, not 12.5 MB/s
    tiny = ledger_net_estimate_s([[("a", "r", 1)], [("r", "a", 1)]], PROFILES["wan"])
    assert tiny == pytest.approx(2 * (0.05 + 0.005), abs=0.005)           # one RTT per tiny stage, not six
    assert ledger_net_estimate_s([[("a", "r", 500_000)]], PROFILES["wan"]) > 5 * 0.05   # slow start for a 500 KB flow


def result(extra_bytes=656_382):
    session = lambda b, cpu: dict(bytes=b, cpu_total_s=cpu, site_max_s=0.1, reducer_s=0.1, crypto_s=cpu, compute_s=0.0)
    stage = lambda sender, size: [(sender, "reducer", size)]
    return dict(scheme="B-BGV", sites=["S0", "S1", "S2"], dropped="S0", threshold=2, vector_size=1000, sessions=3,
                setup=[dict(bytes=9_000_000, time_s=0.5)] * 3,
                rounds=dict(normal=[session(70_000_000, 1.0)] * 3,
                            dropout=[session(50_000_000 + extra_bytes, 0.9 + d) for d in (0.0, 0.02, 0.04)],
                            next=[session(50_000_000, 0.9)] * 3),
                ledgers=dict(setup=[stage("S0", 1)], normal=[stage("S0", 1)] * 4,
                             dropout=[stage("S1", 1)] * 7, next=[stage("S1", 1)] * 5))


def test_summary_isolates_the_recovery_and_marks_the_lead(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(result()))
    s = summarize(str(path), {"lan": PROFILES["lan"], "wan": PROFILES["wan"]})
    assert s["lead_dropped"] and s["recovery_extra_bytes"] == 656_382
    assert s["recovery_extra_cpu_s"] == pytest.approx(0.02) and s["stages"] == dict(setup=1, normal=4, dropout=7, next=5)
    assert s["net_wan_recovery_extra_s"] == pytest.approx(2 * (0.05 + 0.005), rel=0.01)   # two extra stages of 1 byte
    table = markdown([s], ["lan", "wan"])
    assert "| B-BGV | lead | 1000 | 3 | 9.00 | 70.00/50.66/50.00 | 4/7/5 | 656,382 |" in table
