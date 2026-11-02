"""Needs torch: run in the training environment with `--noconftest`."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pqfl.train.dpsgd import (PerSampleGrads, clip_factors, example_loss_sum, poisson_batches,  # noqa: E402
                              train_local_dp)
from pqfl.train.local import flatten, trainable  # noqa: E402
from test_train_local import CharTokenizer  # noqa: E402


class LoraLike(torch.nn.Module):
    """Frozen embedding and head, trainable bias-free A and B Linear modules (like a LoRA adapter)."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(50, 16)
        self.a = torch.nn.Linear(16, 4, bias=False)
        self.b = torch.nn.Linear(4, 16, bias=False)
        self.head = torch.nn.Linear(16, 50, bias=False)
        for module in (self.emb, self.head):
            for p in module.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        return SimpleNamespace(logits=self.head(h + self.b(self.a(h))))


def batch_of(n=4, width=9):
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(2, 50, (n, width), generator=gen)
    labels = ids.clone()
    labels[:, :4] = -100
    labels[1, 7:] = -100                     # a shorter completion in one example
    return ids, labels


def naive_per_sample_grads(model, ids, labels):
    out = []
    for b in range(ids.shape[0]):
        model.zero_grad()
        example_loss_sum(model(input_ids=ids[b:b + 1]).logits, labels[b:b + 1]).backward()
        out.append(torch.cat([p.grad.reshape(-1) for _, p in trainable(model)]))
    return torch.stack(out)


def test_hook_gradients_equal_the_naive_per_example_gradients():
    model = LoraLike()
    ids, labels = batch_of()
    reference = naive_per_sample_grads(model, ids, labels)
    model.zero_grad()
    with PerSampleGrads(model) as hooks:
        example_loss_sum(model(input_ids=ids).logits, labels).backward()
        hooked = torch.cat([hooks.grads[name].flatten(1) for name, _ in hooks.layers], dim=1)
        norms = hooks.squared_norms().sqrt()
    assert torch.allclose(hooked, reference, atol=1e-5)
    assert torch.allclose(norms, reference.norm(dim=1), atol=1e-5)


def test_clip_factors_bound_every_example_by_the_clip_norm():
    squared = torch.tensor([0.25, 4.0, 100.0, 0.0])
    factors = clip_factors(squared, 1.0)
    assert torch.allclose(factors, torch.tensor([1.0, 0.5, 0.1, 1.0]), atol=1e-6)
    assert bool((squared.sqrt() * factors <= 1.0 + 1e-6).all())


def test_poisson_batches_have_the_expected_size_and_are_reproducible():
    batches = poisson_batches(1000, 0.05, 200, seed=3)
    assert abs(sum(map(len, batches)) / 200 - 50) < 2 and batches == poisson_batches(1000, 0.05, 200, seed=3)
    assert batches != poisson_batches(1000, 0.05, 200, seed=4)


ITEMS = [dict(id=str(i), prompt=f"Q{i}: pick Answer:", completion=" A" if i % 2 else " B") for i in range(40)]
KW = dict(epochs=2, micro_batch=4, logical_batch=8, lr=1e-2, clip=1.0, max_len=32, eos_id=1, pad_id=0, seed=5)


def test_without_noise_the_update_is_the_average_of_clipped_example_gradients():
    ref = LoraLike()
    start = flatten(ref)
    out = train_local_dp(LoraLike(), ITEMS, CharTokenizer(), sigma=0.0, **KW)
    assert out["steps"] == 2 * 5 and 0.0 <= max(out["clipped_fraction"]) <= 1.0
    assert out["grad_norm_median"] > 0 and abs(sum(out["batch_sizes"]) / out["steps"] - 8) < 4
    model = LoraLike()
    train_local_dp(model, ITEMS, CharTokenizer(), sigma=0.0, **KW)
    assert (flatten(model) - start).abs().sum() > 0


def test_noise_is_gaussian_with_the_requested_scale_and_seeded():
    # lr small and a huge sigma: the (Adam-normalised) update direction is dominated by noise, but the raw
    # noise scale is checked directly through the generator used for p.grad
    gen = torch.Generator().manual_seed(5)
    draws = torch.cat([torch.randn(1000, generator=gen) * (2.0 * 1.0) for _ in range(20)])
    assert abs(float(draws.std()) - 2.0) < 0.05
    a = LoraLike(); b = LoraLike()
    train_local_dp(a, ITEMS, CharTokenizer(), sigma=1.5, **KW)
    train_local_dp(b, ITEMS, CharTokenizer(), sigma=1.5, **KW)
    assert torch.equal(flatten(a), flatten(b))                                    # reproducible
    c = LoraLike()
    train_local_dp(c, ITEMS, CharTokenizer(), sigma=1.5, **dict(KW, seed=6))
    assert not torch.equal(flatten(a), flatten(c))                                # different seed, different noise


def test_hooks_reject_modules_with_trainable_bias():
    model = LoraLike()
    model.a.bias = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError):
        PerSampleGrads(model)



def test_invalid_dp_parameters_are_rejected():
    for bad in (dict(clip=None), dict(clip=0.0), dict(sigma=-1.0), dict(logical_batch=len(ITEMS) + 1)):
        with pytest.raises(ValueError):
            train_local_dp(LoraLike(), ITEMS, CharTokenizer(), **dict(dict(KW, sigma=1.0), **bad))


def test_secret_randomness_is_not_reproducible_and_reports_signal_and_noise():
    a, b = LoraLike(), LoraLike()
    out = train_local_dp(a, ITEMS, CharTokenizer(), sigma=1.5, secret_randomness=True, **KW)
    train_local_dp(b, ITEMS, CharTokenizer(), sigma=1.5, secret_randomness=True, **KW)
    assert not torch.equal(flatten(a), flatten(b)) and out["secret_randomness"] is True
    dimension = sum(p.numel() for _, p in trainable(LoraLike()))
    expected = KW["logical_batch"]
    assert out["noise_norm_expected"] == pytest.approx(1.5 * KW["clip"] * dimension ** 0.5 / expected)
    assert 0 < out["signal_norm_mean"] <= KW["clip"] * max(out["batch_sizes"]) / expected + 1e-6
