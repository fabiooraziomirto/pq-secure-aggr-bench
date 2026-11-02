import json

import pytest

from pqfl.campaign import State, cell_id, compute_estimate_s, eta, net_rep_estimate_s, stage_egress_bytes
from pqfl.net.compose import agent_arguments, campaign_compose
from pqfl.net.profiles import load_profiles

PROFILES = load_profiles("configs/network_profiles.yaml")


def test_compose_has_one_container_per_site_plus_reducer_and_controller():
    services = campaign_compose(4)["services"]
    assert sorted(services) == ["controller", "reducer", "s0000", "s0001", "s0002", "s0003"]
    assert services["s0002"]["environment"]["ACTOR"] == "S0002"
    assert "NET_ADMIN" in services["reducer"]["cap_add"]
    assert services["controller"]["volumes"] == ["../:/ledgers"]
    assert agent_arguments(["S0000", "reducer"]) == ["S0000=s0000:9000", "reducer=reducer:9000"]


def test_estimates_match_measured_byte_counts_and_order_schemes():
    # Measured at n=10, 1e5: A 10 MB, C 55 MB per round (sum over all senders).
    n, size = 10, 100_000
    a = stage_egress_bytes("A", n, size)
    assert a == [5 * size, n * 5 * size]
    assert stage_egress_bytes("B", 3, 8192)[0] == 788_000
    wan = PROFILES["wan"]
    assert net_rep_estimate_s("A", n, size, wan) < net_rep_estimate_s("C", n, size, wan) \
        < net_rep_estimate_s("B", n, size, wan)
    assert net_rep_estimate_s("C", 5, 10_000, PROFILES["lan"]) < net_rep_estimate_s("C", 5, 10_000, wan)
    assert compute_estimate_s("B", 10, 100_000, 33) > compute_estimate_s("A", 10, 100_000, 33)


def test_bgv_estimates_use_16384_slots_and_flooded_ciphertexts():
    egress = stage_egress_bytes("B-BGV", 3, 16384)
    assert egress[0] == 1_311_945 and egress[2] == 656_560
    # Per element BGV ships fewer chunks than CKKS (16384 vs 8192 slots).
    assert stage_egress_bytes("B-BGV", 3, 1_000_000)[0] < 2 * stage_egress_bytes("B", 3, 1_000_000)[0]
    assert compute_estimate_s("B-BGV", 5, 10_000, 33) == compute_estimate_s("B", 5, 10_000, 33)


def test_state_roundtrip_and_eta_rescaling(tmp_path):
    state = State(tmp_path / "state.json")
    state.add("compute:A", "compute", 100.0)
    state.add("compute:B", "compute", 100.0)
    state.add("net:x", "net", 50.0)
    state.update("compute:A", status="done", seconds=200.0)  # twice the estimate
    reloaded = State.load(tmp_path / "state.json")
    report = eta(reloaded.data, now=reloaded.data["started"] + 10)
    assert report["phases"]["compute"]["ratio"] == pytest.approx(2.0)
    assert report["phases"]["compute"]["remaining_s"] == pytest.approx(200.0)
    assert report["phases"]["net"]["remaining_s"] == pytest.approx(50.0)  # no observations: ratio 1
    assert report["remaining_s"] == pytest.approx(250.0)
    assert not (tmp_path / "state.json.tmp").exists()
    assert cell_id("A", 3, 10) == "A-n3-v10"
    json.dumps(reloaded.data)


# Median replay time per round measured in campaign 3 on the wan profile (50 ms RTT, 100 Mbit/s, 0.1 % loss).
WAN_OBSERVED_S = {("N", 100000): 0.857, ("A", 100000): 0.838, ("C", 100000): 1.675, ("B", 100000): 20.509,
                  ("B-BGV", 100000): 18.221, ("N", 1000000): 6.079, ("A", 1000000): 7.803, ("C", 1000000): 12.526,
                  ("B", 1000000): 169.504, ("B-BGV", 1000000): 143.109, ("N", 3407872): 19.949,
                  ("A", 3407872): 24.495, ("C", 3407872): 39.767, ("B", 3407872): 546.379,
                  ("B-BGV", 3407872): 468.667, ("N", 20971520): 112.750, ("A", 20971520): 141.817,
                  ("C", 20971520): 215.119}


def test_wan_estimate_tracks_the_measured_round_times():
    for (scheme, size), observed in WAN_OBSERVED_S.items():
        predicted = net_rep_estimate_s(scheme, 3, size, PROFILES["wan"])
        assert 0.7 <= predicted / observed <= 1.6, (scheme, size, predicted, observed)


def test_loss_limits_each_flow_so_wan_time_follows_the_mathis_cap_not_the_link():
    wan = PROFILES["wan"]
    base = net_rep_estimate_s("A", 3, 3407872, wan)
    harsher = net_rep_estimate_s("A", 3, 3407872, dict(wan, loss_pct=wan["loss_pct"] * 4))
    assert 1.8 < harsher / base < 2.2                       # sqrt(4) = 2 on a loss-limited transfer
    fatter = net_rep_estimate_s("A", 3, 3407872, dict(wan, rate_mbit=10000))
    assert fatter == pytest.approx(base, rel=0.05)           # a faster link does not help
    lossless = net_rep_estimate_s("A", 3, 3407872, dict(wan, loss_pct=0))
    assert lossless < base / 2
    # lan and metro have no loss: time follows bytes over rate (metro's 1 Gbit/s is slower than lan's)
    assert net_rep_estimate_s("N", 3, 3407872, PROFILES["metro"]) > net_rep_estimate_s("N", 3, 3407872, PROFILES["lan"])


def test_stage_flows_match_the_protocol_structure():
    from pqfl.campaign import stage_flows
    assert [k for _, k in stage_flows("C", 5, 1000)] == [4, 1, 5]            # n-1 shares per site
    assert [k for _, k in stage_flows("A", 7, 1000)] == [1, 7]
    assert len(stage_flows("B", 3, 1000)) == len(stage_flows("B-BGV", 3, 1000)) == 4
    assert max(b for b, _ in stage_flows("B", 3, 1_000_000)) > 20 * max(b for b, _ in stage_flows("A", 3, 1_000_000))
