"""DP-FedSGD with distributed Skellam noise: one DP-SGD step per aggregation round.

Every step, each site Poisson-samples its own records at rate q, clips each example's gradient (at the
current global model) to L2 norm `clip`, scales it by 2^scale_bits and rounds it to the integer grid, sums,
and adds an integer Skellam share. The aggregator only ever sees integer vectors divided by 2^scale_bits,
which the masking schemes (scale_bits equal) carry exactly. The server divides the aggregate by the
expected batch size and takes an optimizer step (post-processing). Accounting: pqfl.dp.skellam_epsilon with
the noise multipliers of pqfl.dp.noise_shares (any n - t sites carry sigma).

Randomness: by default batches and noise derive from the public seed (reproducible, resumable, NOT a
deployable release); secret_randomness draws both from os.urandom. NumPy's Poisson sampler is a floating-point
algorithm, not a constant-time or exact sampler: a stated limitation, as for the Gaussian path in dpsgd.

The full state (global vector, optimizer state, history) is rewritten atomically after every step to
checkpoint_dir/state.pt; resuming requires the same configuration.
"""
import math
import os
import time

import numpy as np
import torch

from pqfl.dp import noise_shares, skellam_sensitivity
from pqfl.train.data import encode_pair, make_batch
from pqfl.train.dpsgd import PerSampleGrads, clip_factors, example_loss_sum
from pqfl.train.federated import _save_updates
from pqfl.train.local import flatten, load_flat, trainable

OPTIMIZERS = ("adamw", "sgd")


def skellam_noise(size: int, variance: float, generator: np.random.Generator) -> np.ndarray:
    """Integer Skellam noise with the given variance: difference of two Poisson(variance / 2)."""
    if not variance > 0:
        raise ValueError("Skellam variance must be positive")
    rate = variance / 2
    return generator.poisson(rate, size).astype(np.int64) - generator.poisson(rate, size).astype(np.int64)


def site_integer_sum(model, encoded: list[dict], batch: list[int], *, clip: float, scale: float, micro_batch: int,
                     pad_id: int) -> tuple[np.ndarray, dict]:
    """Sum over the batch of round(scale * clipped example gradient), int64, in the order of flatten(model).
    Every rounded example is checked against the L2 sensitivity the accountant assumes (skellam_sensitivity),
    so float32 arithmetic can never silently exceed it."""
    device = next(model.parameters()).device
    params = trainable(model)
    bound, _ = skellam_sensitivity(clip, scale, sum(p.numel() for _, p in params))
    total = {name: torch.zeros(p.shape, dtype=torch.int64, device=device) for name, p in params}
    loss_total, clipped, largest = 0.0, 0, 0.0
    model.train()
    with PerSampleGrads(model) as hooks:
        weight_of = {f"{name}.weight": name for name, _ in hooks.layers}
        for i in range(0, len(batch), micro_batch):
            chunk = make_batch([encoded[j] for j in batch[i:i + micro_batch]], pad_id)
            ids = torch.tensor(chunk["input_ids"], device=device)
            mask = torch.tensor(chunk["attention_mask"], device=device)
            labels = torch.tensor(chunk["labels"], device=device)
            hooks.grads = {}
            loss = example_loss_sum(model(input_ids=ids, attention_mask=mask).logits, labels)
            loss.backward()
            squared = hooks.squared_norms()
            factors = clip_factors(squared, clip) * scale
            example_squares = torch.zeros(len(squared), dtype=torch.float64, device=device)
            for name, _ in params:
                rounded = torch.round(hooks.grads[weight_of[name]] * factors[:, None, None]).to(torch.int64)
                example_squares += rounded.double().pow(2).flatten(1).sum(dim=1)
                total[name] += rounded.sum(dim=0)
            norm = float(example_squares.max().sqrt())
            if norm > bound:
                raise RuntimeError(f"A rounded example has L2 norm {norm} above the accounted sensitivity {bound}")
            largest = max(largest, norm)
            loss_total += float(loss.detach())
            clipped += int((squared.sqrt() > clip).sum())
    model.zero_grad(set_to_none=True)
    vector = torch.cat([total[name].reshape(-1) for name, _ in params]).cpu().numpy()
    return vector, dict(examples=len(batch), clipped_fraction=clipped / max(1, len(batch)),
                        mean_loss=loss_total / max(1, len(batch)), max_example_l2=largest)


def _generators(seed: int, step: int, site_index: int, secret: bool):
    if secret:
        return (np.random.default_rng(int.from_bytes(os.urandom(32), "big")) for _ in range(2))
    return (np.random.default_rng(np.random.SeedSequence([seed, step, site_index, purpose])) for purpose in (0, 1))


def _save_state(directory, step, model, optimizer, history, config):
    os.makedirs(directory, exist_ok=True)
    final = os.path.join(directory, "state.pt")
    torch.save({"step": step, "vector": flatten(model), "optimizer": optimizer.state_dict(), "history": history,
                "config": config}, final + ".tmp")
    os.replace(final + ".tmp", final)


def run_dp_fedsgd(model, tokenizer, site_items: dict, aggregator, *, steps: int, sampling_rate: float, clip: float,
                  sigma: float, tolerated: int, scale_bits: int, lr: float, micro_batch: int, max_len: int,
                  eos_id: int, pad_id: int, seed: int, checkpoint_dir: str, server_optimizer: str = "adamw",
                  weight_decay: float = 0.0, secret_randomness: bool = False, updates_dir: str | None = None,
                  save_every: int = 1, rekey_every: int | None = None, evaluate=None, evaluate_every: int | None = None,
                  log=print, config: dict | None = None) -> dict:
    """Run (or resume) `steps` DP-FedSGD steps; sigma is the noise multiplier any n - t sites carry together.
    Returns {"history", "vector", "share_sigma", "aggregate_sigma"}. With updates_dir, every save_every steps the
    per-site released vectors (integer / 2^scale_bits) are saved: what an adversary reads once the masks fall."""
    sites = sorted(site_items)
    if sigma == 0:       # non-private reference run: no noise (same batches as the noisy runs for the same seed)
        if type(tolerated) is not int or not 0 <= tolerated < len(sites):
            raise ValueError("Require 0 <= tolerated < sites")
        share_sigma = aggregate_sigma = 0.0
    else:
        share_sigma, aggregate_sigma = noise_shares(sigma, sites=len(sites), tolerated=tolerated)
    if not 0 < sampling_rate <= 1 or type(scale_bits) is not int or not 1 <= scale_bits <= 30 or not clip > 0:
        raise ValueError("Require 0 < sampling_rate <= 1, 1 <= scale_bits <= 30 and a positive clip")
    if server_optimizer not in OPTIMIZERS:
        raise ValueError(f"server_optimizer must be one of {OPTIMIZERS}")
    if len({len(site_items[s]) for s in sites}) != 1:
        raise ValueError("The accounting assumes sites of equal size")
    if rekey_every is not None and (type(rekey_every) is not int or rekey_every <= 0
                                    or not callable(getattr(aggregator, "rekey", None))):
        raise ValueError("rekey_every needs a positive integer and an aggregator with rekey()")
    if type(save_every) is not int or save_every <= 0:
        raise ValueError("save_every must be a positive integer")
    expected = getattr(aggregator, "sites", None)
    if expected is not None and sorted(expected) != sites:
        raise ValueError("The aggregator was set up with different sites")
    scale = 2.0 ** scale_bits
    share_variance = (scale * clip * share_sigma) ** 2
    expected_batch = sampling_rate * sum(len(site_items[s]) for s in sites)
    config = dict(config or {}, mechanism="dp-fedsgd-skellam", sites=sites, sampling_rate=sampling_rate, clip=clip,
                  sigma=sigma, tolerated=tolerated, scale_bits=scale_bits, lr=lr, seed=seed,
                  server_optimizer=server_optimizer, weight_decay=weight_decay, secret_randomness=secret_randomness)
    encoded = {s: [encode_pair(tokenizer, i["prompt"], i["completion"], max_len=max_len, eos_id=eos_id)
                   for i in site_items[s]] for s in sites}
    params = [p for _, p in trainable(model)]
    optimizer = (torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay) if server_optimizer == "adamw"
                 else torch.optim.SGD(params, lr=lr, weight_decay=weight_decay))
    history, start = [], 0
    state_path = os.path.join(checkpoint_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, weights_only=False)
        if state["config"] != config:
            raise ValueError(f"{state_path} was written by a different configuration; use a new checkpoint_dir")
        if state["step"] >= steps:
            raise ValueError(f"{state_path} is already past step {steps - 1}; use a new checkpoint_dir")
        load_flat(model, state["vector"])
        optimizer.load_state_dict(state["optimizer"])
        history, start = state["history"], state["step"] + 1
        log(f"resumed after step {state['step']}")
    for step in range(start, steps):
        started = time.time()
        if rekey_every and step > start and step % rekey_every == 0:
            aggregator.rekey()
        global_vector = flatten(model)
        released, record = {}, {"step": step, "epoch": getattr(aggregator, "epoch", None), "sites": {}}
        for index, site in enumerate(sites):
            batch_rng, noise_rng = _generators(seed, step, index, secret_randomness)
            batch = np.flatnonzero(batch_rng.random(len(encoded[site])) < sampling_rate).tolist()
            total, stats = site_integer_sum(model, encoded[site], batch, clip=clip, scale=scale,
                                            micro_batch=micro_batch, pad_id=pad_id)
            if share_variance > 0:
                total += skellam_noise(total.size, share_variance, noise_rng)
            released[site] = total.astype(np.float64) / scale        # exact: |total| << 2^53
            record["sites"][site] = dict(stats, max_abs=float(np.abs(released[site]).max()))
        aggregated = np.asarray(aggregator.aggregate_round(step, released), dtype=np.float64)
        if updates_dir is not None and step % save_every == 0:
            _save_updates(updates_dir, step, global_vector, released)
        gradient = torch.from_numpy(aggregated / expected_batch)
        offset = 0
        for p in params:
            p.grad = gradient[offset:offset + p.numel()].view_as(p).to(device=p.device, dtype=p.dtype)
            offset += p.numel()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        examples = sum(r["examples"] for r in record["sites"].values())
        record["mean_batch_loss"] = (sum(r["mean_loss"] * r["examples"] for r in record["sites"].values())
                                     / max(1, examples))
        record["step_seconds"] = time.time() - started
        history.append(record)
        _save_state(checkpoint_dir, step, model, optimizer, history, config)
        if evaluate is not None and evaluate_every and (step + 1) % evaluate_every == 0:
            record["evaluation"] = evaluate(model, step)
            _save_state(checkpoint_dir, step, model, optimizer, history, config)
        log(f"step {step}: loss {record['mean_batch_loss']:.4f}, {examples} examples")
    return {"history": history, "vector": flatten(model), "share_sigma": share_sigma,
            "aggregate_sigma": aggregate_sigma}
