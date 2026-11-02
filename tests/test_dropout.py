"""Dropout recovery: A (seed reveal) and B-BGV (t-of-n key sharing)."""
import numpy as np
import pytest

from conftest import InsecureTestAuth
from pqfl.aggregators.scheme_a_dropout import InsufficientParticipantsError, SchemeADropout
from pqfl.net.replay import ledger_stages

QUANT = 3 * 0.5 * 2 ** -12 + 1e-12


def updates(sites, size=6, seed=0):
    rng = np.random.default_rng(seed)
    return {s: rng.uniform(-0.5, 0.5, size) for s in sites}


def make_a(fake_kem_factory, sites):
    agg = SchemeADropout(fake_kem_factory(3), clip_bound=1., scale_bits=12, modulus_bits=32,
                         authenticator=InsecureTestAuth())
    agg.setup(sites)
    return agg


def test_a_recovers_the_survivor_sum_and_continues_without_the_dropped_site(fake_kem_factory):
    sites = ["a", "b", "c", "d", "e"]
    agg = make_a(fake_kem_factory, sites)
    u = updates(sites)
    out = agg.aggregate_round(0, u, dropped={"c", "e"})
    np.testing.assert_allclose(out, sum(u[s] for s in "abd"), rtol=0, atol=QUANT)
    assert agg.sites == ["a", "b", "d"] and agg.actors == ["a", "b", "d", "reducer"]
    nxt = updates(agg.sites, seed=1)
    np.testing.assert_allclose(agg.aggregate_round(1, nxt), sum(nxt.values()), rtol=0, atol=QUANT)
    agg.rekey()                                                  # rekey among survivors still works
    nxt = updates(agg.sites, seed=2)
    np.testing.assert_allclose(agg.aggregate_round(2, nxt), sum(nxt.values()), rtol=0, atol=QUANT)


def test_a_recovery_traffic_and_stages(fake_kem_factory):
    agg = make_a(fake_kem_factory, ["a", "b", "c"])
    agg.aggregate_round(0, updates(["a", "b", "c"]), dropped={"b"})
    stats = agg.stats()
    stages = ledger_stages(stats)
    assert len(stages) == 4
    assert {s for s, _, _ in stages[0]} == {"a", "c"}                      # the dropped site never uploads
    assert all(r == "reducer" for _, r, _ in stages[2]) and all(size == 32 for *_, size in stages[2])
    assert all(s == "reducer" for s, _, _ in stages[1]) and {r for _, r, _ in stages[3]} == {"a", "c"}
    assert stats.bytes_sent["b"] == 0 and stats.bytes_received["b"] == 0


def test_a_quorum_and_validation_fail_before_any_traffic(fake_kem_factory):
    agg = make_a(fake_kem_factory, ["a", "b", "c"])
    before = agg.stats()
    with pytest.raises(InsufficientParticipantsError):
        agg.aggregate_round(0, updates(["a", "b", "c"]), dropped={"a", "b"})
    with pytest.raises(ValueError, match="active sites"):
        agg.aggregate_round(0, updates(["a", "b", "c"]), dropped={"zz"})
    assert agg.stats() == before and agg.used_rounds == set()
    out = agg.aggregate_round(0, updates(["a", "b", "c"]))                 # no drop: parent path
    assert out.shape == (6,)


# ---- B-BGV threshold -------------------------------------------------------------------
@pytest.fixture
def make_b(fake_kem_factory):
    pytest.importorskip("openfhe")
    from pqfl.aggregators.scheme_b_threshold import SchemeBGVThreshold

    def create(sites, threshold):
        agg = SchemeBGVThreshold(fake_kem_factory(5), clip_bound=1., threshold=threshold,
                                 authenticator=InsecureTestAuth())
        agg.setup(sites)
        return agg
    return create


def expect(u, sites, scale=2 ** 20):
    return sum(np.rint(u[s] * scale) for s in sites) / scale


@pytest.mark.requires_openfhe
@pytest.mark.parametrize("sites,threshold,dropped", [
    (["a", "b", "c"], 2, set()), (["a", "b", "c"], 2, {"c"}), (["a", "b", "c"], 2, {"a"}),
    (["a", "b", "c", "d", "e"], 3, {"b", "e"}), (["a", "b", "c", "d", "e"], 3, {"a", "d"})])
def test_b_threshold_recovers_exactly_and_persists_across_rounds(make_b, sites, threshold, dropped):
    agg = make_b(sites, threshold)
    u = updates(sites, size=20)
    survivors = [s for s in sites if s not in dropped]
    np.testing.assert_array_equal(agg.aggregate_round(0, u, dropped=dropped), expect(u, survivors))
    nxt = updates(survivors, size=20, seed=1)
    np.testing.assert_array_equal(agg.aggregate_round(1, nxt), expect(nxt, survivors))


@pytest.mark.requires_openfhe
def test_b_threshold_traffic_shapes_and_quorum(make_b):
    agg = make_b(["a", "b", "c"], 2)
    setup = agg.stats()
    share_messages = [t for t in setup.transfers if t.size == agg._share_bytes + 16]
    assert len(share_messages) == 3 * 2                                  # every site shares with each peer
    agg.aggregate_round(0, updates(["a", "b", "c"], size=20), dropped={"b"})
    recovery = agg.stats()
    sizes = [t.size for t in recovery.transfers]
    assert agg._share_bytes + 16 in sizes and recovery.bytes_sent["b"] == 0
    assert len(ledger_stages(recovery)) == 7      # upload, sum, partials, request, shares, leader partial, result
    agg.aggregate_round(1, updates(["a", "c"], size=20))
    assert agg._share_bytes + 16 not in [t.size for t in agg.stats().transfers]   # keys already recovered
    with pytest.raises(InsufficientParticipantsError):
        agg.aggregate_round(2, updates(["a", "c"], size=20), dropped={"a"})
    with pytest.raises(ValueError, match="threshold"):
        make_b(["a", "b"], 3)
