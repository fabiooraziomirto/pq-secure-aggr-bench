"""DP-SGD for LoRA (example-level): per-example gradients by hooks, clipping, Gaussian noise.

LoRA modules are bias-free nn.Linear of rank 8, so the gradient of one example w.r.t. a module
weight, sum_t g_t x_t^T, is tiny (at most 6144 x 8 per module, about 34 MB for all parameters of
the 2B model): it is computed explicitly in the same backward pass, which makes micro-batches
larger than one possible. One example is one sequence; its loss is the MEAN token NLL over its
completion (so gradient scale does not depend on length), and the batch loss is the SUM over
examples, so the hooks see exactly the gradient of each example's own loss.

Per logical batch: sum of the clipped per-example gradients + N(0, (sigma*C)^2) noise, divided by
the expected batch size q*N (Poisson sampling, as the accountant in pqfl.dp assumes). The optimizer
step on this noised average is post-processing.

Randomness: by default the Poisson batches and the noise derive from the public run seed, which makes
experiments reproducible but is not a deployable DP release (whoever knows the seed can remove the
noise). With secret_randomness=True both come from os.urandom and are never logged.
"""
import math
import os
import random
import time

import torch

from pqfl.train.data import IGNORE, encode_pair, make_batch
from pqfl.train.local import trainable


class PerSampleGrads:
    """Forward/backward hooks on the trainable bias-free Linear modules (context manager)."""

    def __init__(self, model):
        self.layers = [(n, m) for n, m in model.named_modules()
                       if isinstance(m, torch.nn.Linear) and m.weight.requires_grad]
        weights = {id(m.weight) for _, m in self.layers}
        if weights != {id(p) for _, p in trainable(model)} or any(m.bias is not None for _, m in self.layers):
            raise ValueError("Only bias-free Linear modules may be trainable")
        self.order = [name for name, _ in trainable(model)]
        self._names = {id(m): n for n, m in self.layers}
        self._inputs, self.grads, self._handles = {}, {}, []

    def __enter__(self):
        for name, module in self.layers:
            self._handles.append(module.register_forward_hook(self._forward(name)))
            self._handles.append(module.register_full_backward_hook(self._backward(name)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles, self._inputs, self.grads = [], {}, {}

    def _forward(self, name):
        def hook(module, inputs, output):
            if module.training and torch.is_grad_enabled():
                self._inputs[name] = inputs[0].detach()
        return hook

    def _backward(self, name):
        def hook(module, grad_input, grad_output):
            x, g = self._inputs.pop(name), grad_output[0].detach()
            if g.dim() == 2:
                g, x = g.unsqueeze(1), x.unsqueeze(1)
            self.grads[name] = torch.einsum("bto,bti->boi", g.float(), x.float())     # (B, out, in)
        return hook

    def squared_norms(self) -> torch.Tensor:
        return sum(g.flatten(1).pow(2).sum(dim=1) for g in self.grads.values())

    def clipped_sum(self, factors: torch.Tensor) -> dict:
        return {name: torch.einsum("b,boi->oi", factors, g) for name, g in self.grads.items()}


def clip_factors(squared_norms: torch.Tensor, clip: float) -> torch.Tensor:
    return torch.clamp(clip / (squared_norms.sqrt() + 1e-12), max=1.0)


def example_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum over examples of each example's mean completion NLL."""
    shifted, logits = labels[:, 1:], logits[:, :-1].float()
    token_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1),
                                                   ignore_index=IGNORE, reduction="none").view(shifted.shape)
    counts = (shifted != IGNORE).sum(dim=1).clamp(min=1)
    return (token_loss.sum(dim=1) / counts).sum()


def poisson_batches(n: int, rate: float, steps: int, seed: int) -> list[list[int]]:
    """`steps` index lists, each element included independently with probability `rate`."""
    generator = random.Random(seed)
    return [[i for i in range(n) if generator.random() < rate] for _ in range(steps)]


def train_local_dp(model, items: list[dict], tokenizer, *, epochs: int, micro_batch: int, logical_batch: int,
                   lr: float, clip: float, sigma: float, max_len: int, eos_id: int, pad_id: int, seed: int,
                   weight_decay: float = 0.0, secret_randomness: bool = False) -> dict:
    """Local DP-SGD with Poisson sampling at rate q = logical_batch / len(items); epochs * round(1/q) steps."""
    if clip is None or not clip > 0 or sigma is None or not sigma >= 0:
        raise ValueError("DP-SGD needs a positive clipping norm and a nonnegative noise multiplier")
    if logical_batch < 1 or logical_batch > len(items):
        raise ValueError("logical_batch must be between one and the number of examples (rate q <= 1)")
    device = next(model.parameters()).device
    encoded = [encode_pair(tokenizer, i["prompt"], i["completion"], max_len=max_len, eos_id=eos_id) for i in items]
    n = len(encoded)
    rate = logical_batch / n
    steps = epochs * max(1, round(n / logical_batch))
    expected = rate * n
    params = trainable(model)
    optimizer = torch.optim.AdamW([p for _, p in params], lr=lr, weight_decay=weight_decay)
    if secret_randomness:
        noise_seed, batch_seed = (int.from_bytes(os.urandom(8), "big") for _ in range(2))
    else:
        noise_seed = batch_seed = seed
    noise_generator = torch.Generator().manual_seed(noise_seed)
    batches = poisson_batches(n, rate, steps, batch_seed)
    dimension = sum(m.weight.numel() for _, m in PerSampleGrads(model).layers)
    model.train()
    losses, clipped_fraction, norms, sizes, started, tokens = [], [], [], [], time.time(), 0
    signal_norms = []
    with PerSampleGrads(model) as hooks:
        for batch in batches:
            accumulated = {name: torch.zeros_like(m.weight, dtype=torch.float32) for name, m in hooks.layers}
            loss_total, over = 0.0, 0
            for i in range(0, len(batch), micro_batch):
                chunk = make_batch([encoded[j] for j in batch[i:i + micro_batch]], pad_id)
                ids = torch.tensor(chunk["input_ids"], device=device)
                mask = torch.tensor(chunk["attention_mask"], device=device)
                labels = torch.tensor(chunk["labels"], device=device)
                hooks.grads = {}
                loss = example_loss_sum(model(input_ids=ids, attention_mask=mask).logits, labels)
                loss.backward()
                squared = hooks.squared_norms()
                factors = clip_factors(squared, clip)
                for name, g in hooks.clipped_sum(factors).items():
                    accumulated[name] += g
                loss_total += float(loss.detach())
                tokens += int(mask.sum())
                over += int((squared.sqrt() > clip).sum())
                norms += squared.sqrt().tolist()
            signal_norms.append(math.sqrt(sum(float(a.pow(2).sum()) for a in accumulated.values())) / expected)
            for name, module in hooks.layers:
                p = module.weight
                noise = torch.randn(p.shape, generator=noise_generator) * (sigma * clip)
                p.grad = ((accumulated[name] + noise.to(p.device)) / expected).to(p.dtype)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sizes.append(len(batch))
            losses.append(loss_total / max(1, len(batch)))
            clipped_fraction.append(over / max(1, len(batch)))
    return dict(steps=steps, losses=losses, clipped_fraction=clipped_fraction, batch_sizes=sizes, tokens=tokens,
                grad_norm_median=float(torch.tensor(norms).median()) if norms else None, rate=rate,
                seconds=time.time() - started, secret_randomness=secret_randomness,
                signal_norm_mean=sum(signal_norms) / len(signal_norms) if signal_norms else None,
                noise_norm_expected=sigma * clip * math.sqrt(dimension) / expected)
