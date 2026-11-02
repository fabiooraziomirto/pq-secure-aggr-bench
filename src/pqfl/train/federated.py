"""Federated LoRA loop: local training per site, secure (or plain) sum of the updates, average.

The aggregator is any object with aggregate_round(round_idx, {site: float64 vector}) -> summed
vector and, optionally, stats() (byte counters) and sites. The loop never imports the
aggregator package, so it also runs in the training environment. Every round writes an atomic
checkpoint (global vector, history); a restart resumes after the last one with a fresh aggregator
session, and with the same seeds the continuation is identical to an uninterrupted run. A checkpoint
records the run configuration; resuming with a different configuration, or with fewer rounds than
already done, is refused (it would silently mix trajectories or, under DP, misstate the budget). The
checkpoint is written right after aggregation, before the evaluation, so that a crash during the
evaluation never re-runs (re-releases) an aggregated round.
"""
import json
import os
import time

import numpy as np
import torch

from pqfl.train.local import flatten, load_flat, train_local
from pqfl.train.updates import apply_average, clamp_update, local_seed


def latest_checkpoint(directory):
    names = sorted(n for n in os.listdir(directory) if n.startswith("round-") and n.endswith(".pt")) \
        if os.path.isdir(directory) else []
    return os.path.join(directory, names[-1]) if names else None


def _save(directory, round_idx, vector, history, config):
    os.makedirs(directory, exist_ok=True)
    final = os.path.join(directory, f"round-{round_idx:04d}.pt")
    temporary = final + ".tmp"
    torch.save({"round": round_idx, "vector": vector, "history": history, "config": config}, temporary)
    os.replace(temporary, final)


def run_federated(model, tokenizer, site_items: dict, aggregator, *, rounds: int, train_kwargs: dict, seed: int,
                  checkpoint_dir: str, clamp_bound: float | None = None, evaluate=None, log=print,
                  train_function=train_local, config: dict | None = None) -> dict:
    """Run (or resume) `rounds` federated rounds; returns {"history": [...], "vector": final global vector}.
    `train_kwargs` go to `train_function`: train_local (epochs, micro_batch, accumulation, lr, max_len, eos_id,
    pad_id) or pqfl.train.dpsgd.train_local_dp (DP-SGD, same notion of seed)."""
    sites = sorted(site_items)
    expected = getattr(aggregator, "sites", None)
    if expected is not None and sorted(expected) != sites:
        raise ValueError("The aggregator was set up with different sites")
    history, start = [], 0
    previous = latest_checkpoint(checkpoint_dir)
    if previous:
        state = torch.load(previous)
        if state.get("config") != config:
            raise ValueError(f"{previous} was written by a different configuration; use a new checkpoint_dir")
        if state["round"] >= rounds:
            raise ValueError(f"{previous} is already past round {rounds - 1}; use a new checkpoint_dir")
        load_flat(model, state["vector"])
        history, start = state["history"], state["round"] + 1
        log(f"resumed after round {state['round']}")
    for round_idx in range(start, rounds):
        started = time.time()
        global_vector = flatten(model)
        deltas, record = {}, {"round": round_idx, "sites": {}}
        clamped = []
        for index, site in enumerate(sites):
            load_flat(model, global_vector)
            out = train_function(model, site_items[site], tokenizer, seed=local_seed(seed, round_idx, index),
                                 **train_kwargs)
            delta = (flatten(model) - global_vector).double().numpy()
            largest = float(np.max(np.abs(delta)))
            delta, fraction = clamp_update(delta, clamp_bound)
            clamped.append(fraction)
            deltas[site] = delta
            record["sites"][site] = dict(loss_first=out["losses"][0] if out["losses"] else None,
                                         loss_last=out["losses"][-1] if out["losses"] else None,
                                         steps=out["steps"], tokens=out["tokens"], seconds=out["seconds"],
                                         update_norm=float(np.linalg.norm(delta)), max_abs_update=largest,
                                         clipped_fraction=(float(np.mean(out["clipped_fraction"]))
                                                           if out.get("clipped_fraction") else None),
                                         grad_norm_median=out.get("grad_norm_median"),
                                         signal_norm_mean=out.get("signal_norm_mean"),
                                         noise_norm_expected=out.get("noise_norm_expected"))
        aggregated_at = time.time()
        total = aggregator.aggregate_round(round_idx, deltas)
        record["aggregate_seconds"] = time.time() - aggregated_at
        stats = aggregator.stats() if hasattr(aggregator, "stats") else None
        if stats is not None:
            record["bytes_sent_total"] = int(sum(stats.bytes_sent.values()))
        record["clamped_fraction"] = float(np.mean(clamped))
        new_vector = apply_average(global_vector.double().numpy(), np.asarray(total, dtype=np.float64), len(sites))
        load_flat(model, torch.from_numpy(new_vector).float())
        record["round_seconds"] = time.time() - started
        history.append(record)
        vector = flatten(model)
        _save(checkpoint_dir, round_idx, vector, history, config)
        if evaluate is not None:
            record["evaluation"] = evaluate(model, round_idx)
            _save(checkpoint_dir, round_idx, vector, history, config)
        log(json.dumps({k: v for k, v in record.items() if k != "sites"}))
    return {"history": history, "vector": flatten(model)}
