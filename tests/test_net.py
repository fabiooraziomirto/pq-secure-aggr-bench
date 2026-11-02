import asyncio
import json

import numpy as np
import pytest
import yaml

from conftest import InsecureTestAuth
from pqfl.net.profiles import load_profiles, tc_commands
from pqfl.net.replay import Agent, Controller, ledger_stages

SITES = ["A", "B", "C"]
UPDATES = {s: np.full(50, 0.1 * (i + 1)) for i, s in enumerate(SITES)}


def aggregators(fake_kem_factory):
    from pqfl.aggregators.insecure_stub import InsecureSumStub
    from pqfl.aggregators.scheme_a import SchemeA
    from pqfl.aggregators.scheme_c import SchemeC
    kw = dict(clip_bound=1., scale_bits=12, modulus_bits=32, authenticator=InsecureTestAuth())
    yield "STUB", InsecureSumStub(), [2, 0]
    yield "A", SchemeA(fake_kem_factory(1), **kw), [3, 2]
    yield "C", SchemeC(fake_kem_factory(1), **kw), [3, 3]


def test_ledger_stages_match_protocol_structure(fake_kem_factory):
    for name, agg, (setup_stage_count, round_extra) in aggregators(fake_kem_factory):
        agg.setup(SITES)
        setup = ledger_stages(agg.stats())
        assert len(setup) == (setup_stage_count if name != "STUB" else 0) or name == "STUB"
        agg.aggregate_round(0, UPDATES)
        stages = ledger_stages(agg.stats())
        assert len(stages) == {"STUB": 2, "A": 2, "C": 3}[name]
        flat = sum(len(stage) for stage in stages)
        assert flat == len(agg.stats().transfers)
        # Last stage is always the reducer broadcast.
        assert all(sender == "reducer" for sender, _, _ in stages[-1])


def test_b_stages_follow_the_key_chain_and_four_round_steps():
    pytest.importorskip("openfhe")
    from pqfl.aggregators.scheme_b import SchemeB
    agg = SchemeB(clip_bound=1., authenticator=InsecureTestAuth())
    agg.setup(SITES)
    setup = ledger_stages(agg.stats())
    assert [{(s, r) for s, r, _ in stage} for stage in setup] == [
        {("A", "B")}, {("B", "C")}, {("C", "A"), ("C", "B")}]
    agg.aggregate_round(0, UPDATES)
    assert len(ledger_stages(agg.stats())) == 4


def test_ledger_without_stages_is_rejected():
    from pqfl.logging.metrics import RoundStats, Transfer
    with pytest.raises(ValueError):
        ledger_stages(RoundStats({}, {}, [Transfer("a", "b", 1)], []))


def test_replay_over_real_sockets_moves_exactly_the_ledger_bytes():
    stages = [[("A", "reducer", 100_000), ("B", "reducer", 250_000), ("C", "reducer", 1)],
              [("reducer", "A", 7), ("reducer", "B", 70_000), ("reducer", "C", 0)]]

    async def scenario():
        agents = {name: Agent(name) for name in ("A", "B", "C", "reducer")}
        addresses = {}
        for name, agent in agents.items():
            addresses[name] = ("127.0.0.1", await agent.start("127.0.0.1", 0))
        controller = Controller(addresses)
        try:
            await controller.prepare(stages)
            times = [await controller.run_stages(stages) for _ in range(3)]
            return times, await controller.stats()
        finally:
            await controller.close()
            for agent in agents.values():
                await agent.close()

    times, stats = asyncio.run(scenario())
    assert all(len(t) == 2 and all(x > 0 for x in t) for t in times)
    total = sum(size for stage in stages for _, _, size in stage)
    assert sum(s["bytes_received"] for s in stats.values()) == 3 * total
    assert stats["reducer"]["bytes_received"] == 3 * 350_001
    assert stats["A"]["bytes_sent"] == 3 * 100_000


def test_profiles_and_tc_commands(tmp_path):
    profiles = load_profiles("configs/network_profiles.yaml")
    assert set(profiles) == {"lan", "metro", "wan", "constrained"}
    assert tc_commands(profiles["lan"], "eth0") == []
    wan = tc_commands(profiles["wan"], "eth0")
    assert "delay 25ms" in wan[1] and "loss 0.1%" in wan[1] and "rate 100mbit" in wan[2]
    assert "delay 5ms" in tc_commands(profiles["metro"], "eth0")[1]
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({"profiles": {"bad": dict(rtt_ms=-1, rate_mbit=None, loss_pct=0)}}))
    with pytest.raises(ValueError):
        load_profiles(path)
    with pytest.raises(ValueError):
        tc_commands(profiles["wan"], "eth0; rm -rf /")


def test_interface_counters_are_read_from_proc_net_dev():
    async def scenario():
        agent = Agent("A")
        port = await agent.start("127.0.0.1", 0)
        controller = Controller({"A": ("127.0.0.1", port)})
        return await controller.ifstats("lo"), await controller.ifstats("no-such-iface")

    lo, missing = asyncio.run(scenario())
    assert lo["A"]["rx_bytes"] > 0 and lo["A"]["tx_bytes"] > 0
    assert missing["A"] == {"rx_bytes": None, "tx_bytes": None}
