"""DP-FedSGD with integer Skellam shares (needs torch)."""
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pqfl.train.dpsgd import clip_factors  # noqa: E402
from pqfl.train.fedsgd import run_dp_fedsgd, site_integer_sum, skellam_noise  # noqa: E402
from pqfl.train.local import flatten  # noqa: E402
from pqfl.train.data import encode_pair  # noqa: E402
from test_train_dpsgd import LoraLike, naive_per_sample_grads  # noqa: E402
from test_train_federated import Rekeying, Sum  # noqa: E402
from test_train_local import CharTokenizer  # noqa: E402


def items(offset, count=12):
    return [dict(id=f"{offset}-{i}", prompt=f"Q{offset}{i}: pick Answer:", completion=" A" if (i + offset) % 2 else " B")
            for i in range(count)]


SITES = {"a": items(0), "b": items(1), "c": items(2)}
KW = dict(sampling_rate=0.5, clip=0.5, sigma=0.05, tolerated=0, scale_bits=16, lr=5e-2, micro_batch=4, max_len=32,
          eos_id=1, pad_id=0, seed=3)


def go(directory, steps, aggregator=None, **extra):
    model = LoraLike()
    kwargs = dict(KW, **extra)
    out = run_dp_fedsgd(model, CharTokenizer(), SITES, aggregator or Sum(sorted(SITES)), steps=steps,
                        checkpoint_dir=str(directory), log=lambda m: None, **kwargs)
    return model, out


def test_site_sum_is_the_sum_of_rounded_scaled_clipped_example_gradients():
    model, scale, clip = LoraLike(), 2.0 ** 10, 0.5
    encoded = [encode_pair(CharTokenizer(), i["prompt"], i["completion"], max_len=32, eos_id=1) for i in items(0, 5)]
    total, stats = site_integer_sum(model, encoded, list(range(5)), clip=clip, scale=scale, micro_batch=2, pad_id=0)
    from pqfl.train.data import make_batch
    batch = make_batch(encoded, 0)
    ids, labels = torch.tensor(batch["input_ids"]), torch.tensor(batch["labels"])
    reference = naive_per_sample_grads(model, ids, labels)        # mask-free: equal-length prompts in items()
    factors = clip_factors(reference.pow(2).sum(dim=1), clip)
    expected = torch.round(reference * factors[:, None] * scale).to(torch.int64).sum(dim=0).numpy()
    assert total.dtype == np.int64 and np.abs(total - expected).max() <= 2     # float rounding ties only
    assert stats["examples"] == 5 and 0 <= stats["clipped_fraction"] <= 1
    from pqfl.dp import skellam_sensitivity
    dim = int(reference.shape[1])
    assert 0 < stats["max_example_l2"] <= skellam_sensitivity(clip, scale, dim)[0]
    assert stats["max_example_l2"] == pytest.approx(float(torch.round(reference * factors[:, None] * scale)
                                                         .norm(dim=1).max()), rel=1e-3)


def test_skellam_noise_is_integer_centred_and_has_the_requested_variance():
    gen = np.random.default_rng(0)
    noise = skellam_noise(400_000, 50.0, gen)
    assert noise.dtype == np.int64
    assert abs(noise.mean()) < 0.05 and noise.var() == pytest.approx(50.0, rel=0.02)
    assert np.array_equal(skellam_noise(10, 8.0, np.random.default_rng(1)), skellam_noise(10, 8.0, np.random.default_rng(1)))
    with pytest.raises(ValueError):
        skellam_noise(10, 0.0, gen)


def test_shares_follow_the_tolerated_collusion(tmp_path):
    _, zero = go(tmp_path / "t0", 1, tolerated=0)
    _, local = go(tmp_path / "t2", 1, tolerated=2)
    assert zero["share_sigma"] == pytest.approx(0.05 / math.sqrt(3))
    assert local["share_sigma"] == pytest.approx(0.05)


def test_a_step_is_the_server_update_of_the_aggregated_noisy_sum(tmp_path):
    start = flatten(LoraLike()).double().numpy()
    model, out = go(tmp_path / "c", 1, updates_dir=str(tmp_path / "u"), server_optimizer="sgd")
    saved = np.load(tmp_path / "u" / "round-0000.npz")
    released = sum(saved[f"update:{s}"] for s in SITES)
    for s in SITES:                                                   # what is sent is integer / 2^scale_bits
        assert np.array_equal(saved[f"update:{s}"] * 2 ** 16, np.rint(saved[f"update:{s}"] * 2 ** 16))
    expected_batch = KW["sampling_rate"] * sum(len(v) for v in SITES.values())
    np.testing.assert_allclose(flatten(model).double().numpy(), start - KW["lr"] * released / expected_batch,
                               atol=1e-6)
    assert out["history"][0]["epoch"] is None and set(out["history"][0]["sites"]) == set(SITES)


def test_training_progresses_with_little_noise(tmp_path):
    _, out = go(tmp_path, 12, evaluate_every=None)
    losses = [h["mean_batch_loss"] for h in out["history"]]
    assert np.mean(losses[-3:]) < np.mean(losses[:3])


def test_resume_equals_an_uninterrupted_run(tmp_path):
    straight, _ = go(tmp_path / "straight", 4)
    go(tmp_path / "split", 2)
    resumed, out = go(tmp_path / "split", 4)
    assert torch.equal(flatten(resumed), flatten(straight))
    assert [h["step"] for h in out["history"]] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="different configuration"):
        go(tmp_path / "split", 5, sigma=0.1)


def test_updates_are_saved_every_k_steps_and_rekey_happens_every_k_steps(tmp_path):
    aggregator = Rekeying(sorted(SITES))
    _, out = go(tmp_path / "c", 5, aggregator=aggregator, updates_dir=str(tmp_path / "u"), save_every=2,
                rekey_every=2)
    assert sorted(os.listdir(tmp_path / "u")) == ["round-0000.npz", "round-0002.npz", "round-0004.npz"]
    assert aggregator.rekeyed_before == [2, 4] and [h["epoch"] for h in out["history"]] == [0, 0, 1, 1, 2]


def test_secure_scheme_a_sums_the_integer_shares_exactly(tmp_path, fake_kem_factory):
    from conftest import InsecureTestAuth
    from pqfl.aggregators.scheme_a import SchemeA

    class Recording(SchemeA):
        def aggregate_round(self, round_idx, updates):
            total = super().aggregate_round(round_idx, updates)
            self.check = np.array_equal(total * 2 ** 16, sum(np.rint(u * 2 ** 16) for u in updates.values()))
            return total

    aggregator = Recording(fake_kem_factory(7), scale_bits=16, modulus_bits=48, clip_bound=1000.0,
                           authenticator=InsecureTestAuth())
    aggregator.setup(sorted(SITES))
    go(tmp_path, 1, aggregator=aggregator)
    assert aggregator.check


def test_sigma_zero_is_the_noise_free_reference_with_identical_batches(tmp_path):
    """sigma = 0 adds no noise (a non-private reference run); same public seed gives the same batches, so the
    difference to a noisy run is exactly the added noise."""
    _, free = go(tmp_path / "free", 1, sigma=0.0, updates_dir=str(tmp_path / "uf"))
    _, noisy = go(tmp_path / "noisy", 1, sigma=0.5, updates_dir=str(tmp_path / "un"))
    assert free["share_sigma"] == 0.0 and free["aggregate_sigma"] == 0.0
    a, b = np.load(tmp_path / "uf" / "round-0000.npz"), np.load(tmp_path / "un" / "round-0000.npz")
    scale, clip = 2.0 ** 16, KW["clip"]
    noises = []
    for site in SITES:
        exact, with_noise = a[f"update:{site}"], b[f"update:{site}"]
        assert np.array_equal(exact * scale, np.rint(exact * scale))            # integers on the 2^-16 grid
        noises.append(with_noise - exact)                                       # gradient units
    # the toy model has 128 coordinates: pooling the 3 sites gives 384 samples, relative error of the variance ~7 %
    assert np.concatenate(noises).var() == pytest.approx((clip * 0.5 / math.sqrt(3)) ** 2, rel=0.25)
    assert free["history"][0]["sites"]["a"]["examples"] == noisy["history"][0]["sites"]["a"]["examples"]


def test_invalid_settings_are_refused(tmp_path):
    for bad in (dict(tolerated=3), dict(sigma=-0.1), dict(sampling_rate=0.0), dict(sampling_rate=1.5),
                dict(scale_bits=0), dict(server_optimizer="lion")):
        with pytest.raises(ValueError):
            go(tmp_path, 1, **bad)
    with pytest.raises(ValueError, match="equal"):
        run_dp_fedsgd(LoraLike(), CharTokenizer(), {"a": items(0), "b": items(1, 5)}, Sum(["a", "b"]), steps=1,
                      checkpoint_dir=str(tmp_path), log=lambda m: None, **KW)
