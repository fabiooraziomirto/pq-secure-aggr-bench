"""Needs torch: run in the training environment with `--noconftest`."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pqfl.train.federated import latest_checkpoint, run_federated  # noqa: E402
from pqfl.train.local import flatten, load_flat, train_local  # noqa: E402
from pqfl.train.updates import local_seed  # noqa: E402
from test_train_local import CharTokenizer, Tiny  # noqa: E402


class Sum:
    """Plain sum aggregator (what N does), with the byte counters of the real ones."""
    def __init__(self, sites):
        self.sites, self.rounds = sites, []

    def aggregate_round(self, round_idx, updates):
        self.rounds.append(round_idx)
        self._bytes = {s: u.nbytes for s, u in updates.items()}
        return sum(updates.values())

    def stats(self):
        return SimpleNamespace(bytes_sent=self._bytes)


def items(offset):
    return [dict(id=f"{offset}{i}", prompt=f"Q{offset}{i}: pick Answer:", completion=" A" if (i + offset) % 2 else " B")
            for i in range(8)]


SITES = {"a": items(0), "b": items(1)}
KW = dict(epochs=1, micro_batch=2, accumulation=2, lr=1e-2, max_len=32, eos_id=1, pad_id=0)


def go(directory, rounds, **extra):
    model = Tiny()
    out = run_federated(model, CharTokenizer(), SITES, Sum(sorted(SITES)), rounds=rounds, train_kwargs=KW, seed=3,
                        checkpoint_dir=str(directory), log=lambda m: None, **extra)
    return model, out


def test_a_round_is_the_average_of_the_local_updates(tmp_path):
    start = flatten(Tiny())
    deltas = []
    for index, site in enumerate(sorted(SITES)):
        model = Tiny()
        train_local(model, SITES[site], CharTokenizer(), seed=local_seed(3, 0, index), **KW)
        deltas.append(flatten(model) - start)
    _, out = go(tmp_path, 1)
    expected = start + (deltas[0] + deltas[1]) / 2
    assert torch.allclose(out["vector"], expected.float(), atol=1e-6)
    record = out["history"][0]
    assert record["bytes_sent_total"] == 2 * 8 * start.numel() and set(record["sites"]) == {"a", "b"}


def test_training_progresses_and_checkpoints_every_round(tmp_path):
    _, out = go(tmp_path, 4)
    first, last = out["history"][0]["sites"]["a"]["loss_first"], out["history"][-1]["sites"]["a"]["loss_last"]
    assert last < first and len(out["history"]) == 4
    assert sorted(n for n in os.listdir(tmp_path)) == [f"round-{r:04d}.pt" for r in range(4)]
    assert latest_checkpoint(str(tmp_path)).endswith("round-0003.pt")


def test_resume_after_a_stop_equals_an_uninterrupted_run(tmp_path):
    _, straight = go(tmp_path / "straight", 4)
    go(tmp_path / "split", 2)                                    # "crash" after round 1
    model, resumed = go(tmp_path / "split", 4)
    assert torch.equal(resumed["vector"], straight["vector"])
    assert [h["round"] for h in resumed["history"]] == [0, 1, 2, 3]
    aggregator = Sum(sorted(SITES))
    run_federated(Tiny(), CharTokenizer(), SITES, aggregator, rounds=4, train_kwargs=KW, seed=3,
                  checkpoint_dir=str(tmp_path / "split"), log=lambda m: None)
    assert aggregator.rounds == []                               # already complete: nothing to do


def test_clamping_bounds_the_update_and_is_logged(tmp_path):
    start = flatten(Tiny())
    _, out = go(tmp_path, 1, clamp_bound=1e-4)
    assert out["history"][0]["clamped_fraction"] > 0.3
    assert float((out["vector"] - start).abs().max()) <= 1e-4 + 1e-7


def test_mismatched_sites_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="different sites"):
        run_federated(Tiny(), CharTokenizer(), SITES, Sum(["a", "zz"]), rounds=1, train_kwargs=KW, seed=3,
                      checkpoint_dir=str(tmp_path), log=lambda m: None)


def test_dp_training_runs_inside_the_federated_loop_and_logs_clipping(tmp_path):
    from pqfl.train.dpsgd import train_local_dp
    from test_train_dpsgd import LoraLike
    kw = dict(epochs=1, micro_batch=4, logical_batch=4, lr=1e-2, clip=0.5, sigma=1.0, max_len=32, eos_id=1, pad_id=0)
    model = LoraLike()          # only bias-free Linear modules are trainable, as with LoRA
    out = run_federated(model, CharTokenizer(), SITES, Sum(sorted(SITES)), rounds=2, train_kwargs=kw, seed=3,
                        checkpoint_dir=str(tmp_path), log=lambda m: None, train_function=train_local_dp)
    site = out["history"][0]["sites"]["a"]
    assert site["clipped_fraction"] is not None and site["grad_norm_median"] > 0 and len(out["history"]) == 2


def test_a_single_site_is_centralized_training_with_one_epoch_per_round(tmp_path):
    from test_train_federated import Sum as _Sum
    only = {"Z": SITES["a"] + SITES["b"]}
    model = Tiny()
    out = run_federated(model, CharTokenizer(), only, _Sum(["Z"]), rounds=3, train_kwargs=KW, seed=3,
                        checkpoint_dir=str(tmp_path), log=lambda m: None)
    first, last = out["history"][0]["sites"]["Z"]["loss_first"], out["history"][-1]["sites"]["Z"]["loss_last"]
    assert last < first and out["history"][0]["sites"]["Z"]["steps"] == 4        # 16 items / logical batch 4



def test_resume_with_another_configuration_or_fewer_rounds_is_refused(tmp_path):
    go(tmp_path, 2, config=dict(lr=1e-2))
    with pytest.raises(ValueError, match="different configuration"):
        go(tmp_path, 3, config=dict(lr=5e-3))
    with pytest.raises(ValueError, match="already past"):
        go(tmp_path, 1, config=dict(lr=1e-2))
    _, out = go(tmp_path, 3, config=dict(lr=1e-2))
    assert [h["round"] for h in out["history"]] == [0, 1, 2]


def test_the_round_is_checkpointed_before_the_evaluation(tmp_path):
    def crashing_evaluation(model, round_idx):
        raise RuntimeError("evaluation crashed")
    with pytest.raises(RuntimeError):
        go(tmp_path, 2, evaluate=crashing_evaluation)
    assert sorted(os.listdir(tmp_path)) == ["round-0000.pt"]


def test_site_updates_are_saved_with_the_global_vector_they_start_from(tmp_path):
    start = flatten(Tiny())
    _, out = go(tmp_path / "checkpoints", 2, updates_dir=str(tmp_path / "updates"))
    assert sorted(os.listdir(tmp_path / "updates")) == ["round-0000.npz", "round-0001.npz"]
    first, second = (np.load(tmp_path / "updates" / f"round-{r:04d}.npz") for r in (0, 1))
    assert int(first["round"]) == 0 and sorted(k for k in first.files if k.startswith("update:")) == \
        ["update:a", "update:b"]
    assert np.array_equal(first["global_before"], start.numpy())
    expected = start.double().numpy() + (first["update:a"] + first["update:b"]) / 2
    np.testing.assert_allclose(second["global_before"], expected, atol=1e-6)
    final = second["global_before"] + (second["update:a"] + second["update:b"]) / 2
    np.testing.assert_allclose(out["vector"].numpy(), final, atol=1e-6)


def test_without_updates_dir_nothing_but_checkpoints_is_written(tmp_path):
    go(tmp_path, 1)
    assert os.listdir(tmp_path) == ["round-0000.pt"]
