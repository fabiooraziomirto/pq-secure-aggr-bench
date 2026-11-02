"""Jobs the GPU worker can run; each returns a JSON-serializable dict."""


def gpu_probe(allocate_gb: float = 0.0) -> dict:
    """Memory state of this process; optionally allocate to check the reserved cache is reused."""
    import torch
    before = torch.cuda.memory_reserved()
    block = torch.empty(int(allocate_gb * 2 ** 30), dtype=torch.uint8, device="cuda") if allocate_gb else None
    after = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    out = dict(reserved_before_gb=before / 2 ** 30, reserved_after_gb=after / 2 ** 30,
               allocated_gb=torch.cuda.memory_allocated() / 2 ** 30, device_free_gb=free / 2 ** 30,
               device_total_gb=total / 2 ** 30, reused_cache=block is not None and after == before)
    del block
    return out


def _reset_peak_memory():
    """Peak-memory statistics are per process: reset them at the start of every job of the worker."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def sleep_job(seconds: float = 0.0, fail: bool = False) -> dict:
    import time
    time.sleep(seconds)
    if fail:
        raise RuntimeError("requested failure")
    return {"slept": seconds}


def smoke_train(model_path: str, rows_path: str, manifest_path: str, site: int = 0, n_items: int = 16,
                micro_batch: int = 1, accumulation: int = 4, epochs: int = 1, lr: float = 1e-4,
                max_len: int = 512, seed: int = 0, rank: int = 8, alpha: int = 16) -> dict:
    """A few LoRA steps on natural pool examples of one site: checks loading, loss, memory and speed."""
    import json
    import time

    import torch

    from pqfl.train.data import site_examples
    from pqfl.train.local import count_trainable, flatten, load_model, train_local
    rows = {r["id"]: r for r in map(json.loads, open(rows_path, encoding="utf-8"))}
    pool = json.load(open(manifest_path, encoding="utf-8"))["pool"][site][:n_items]
    items = site_examples(site, pool, rows, None)
    torch.cuda.reset_peak_memory_stats()
    loaded = time.time()
    model, tokenizer = load_model(model_path, rank=rank, alpha=alpha, seed=seed)
    load_seconds = time.time() - loaded
    before = flatten(model)
    out = train_local(model, items, tokenizer, epochs=epochs, micro_batch=micro_batch, accumulation=accumulation,
                      lr=lr, max_len=max_len, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, seed=seed)
    after = flatten(model)
    return dict(trainable=count_trainable(model), examples=len(items), steps=out["steps"], losses=out["losses"],
                tokens=out["tokens"], train_seconds=out["seconds"], load_seconds=load_seconds,
                step_seconds=out["step_seconds"], step_tokens=out["step_tokens"],
                tokens_per_s=out["tokens"] / out["seconds"], update_norm=float((after - before).norm()),
                finite=bool(torch.isfinite(after).all()), peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30)


def smoke_federated(model_path: str, rows_path: str, manifest_path: str, checkpoint_dir: str, n_items: int = 8,
                    rounds: int = 2, stop_after: int | None = None, micro_batch: int = 4, lr: float = 1e-4,
                    max_len: int = 512, seed: int = 0) -> dict:
    """Federated loop on the real model with three sites and a plain sum; `stop_after` runs only that many
    rounds (then call again with the full count to test the resume)."""
    import json
    import time
    from types import SimpleNamespace

    import torch

    from pqfl.train.data import site_examples
    from pqfl.train.federated import run_federated
    from pqfl.train.local import count_trainable, load_model

    class PlainSum:
        sites = None

        def aggregate_round(self, round_idx, updates):
            self.bytes = {s: u.nbytes for s, u in updates.items()}
            return sum(updates.values())

        def stats(self):
            return SimpleNamespace(bytes_sent=self.bytes)

    rows = {r["id"]: r for r in map(json.loads, open(rows_path, encoding="utf-8"))}
    pools = json.load(open(manifest_path, encoding="utf-8"))["pool"]
    site_items = {f"S{i}": site_examples(i, pools[i][:n_items], rows, None) for i in range(3)}
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=seed)
    started = time.time()
    out = run_federated(model, tokenizer, site_items, PlainSum(), rounds=stop_after or rounds, seed=seed,
                        checkpoint_dir=checkpoint_dir, log=print,
                        train_kwargs=dict(epochs=1, micro_batch=micro_batch, accumulation=1, lr=lr, max_len=max_len,
                                          eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id))
    return dict(trainable=count_trainable(model), rounds_done=len(out["history"]), seconds=time.time() - started,
                round_seconds=[h["round_seconds"] for h in out["history"]],
                bytes_sent_total=[h["bytes_sent_total"] for h in out["history"]],
                clamped=[h["clamped_fraction"] for h in out["history"]],
                update_norms=[{s: v["update_norm"] for s, v in h["sites"].items()} for h in out["history"]],
                finite=bool(torch.isfinite(out["vector"]).all()), peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30)


def eval_job(model_path: str, validation_path: str, holdout_path: str, n_validation: int = 500,
             n_holdout: int = 300, batch_size: int = 8, max_len: int = 512, checkpoint: str | None = None,
             output: str | None = None, tag: str = "eval") -> dict:
    """Evaluate the base model (or a federated checkpoint) on a validation subset and on holdout rows."""
    import json
    import time

    import torch

    from pqfl.train.evaluate import append_jsonl, evaluate_model
    from pqfl.train.local import load_flat, load_model
    _reset_peak_memory()
    load = lambda p, n: [json.loads(l) for l in open(p, encoding="utf-8")][:n]
    validation, holdout = load(validation_path, n_validation), load(holdout_path, n_holdout)
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=0, gradient_checkpointing=False)
    if checkpoint:
        load_flat(model, torch.load(checkpoint)["vector"])
    started = time.time()
    result = evaluate_model(model, tokenizer, validation_rows=validation, holdout_rows=holdout, batch_size=batch_size,
                            max_len=max_len, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id)
    result.update(tag=tag, seconds=time.time() - started, checkpoint=checkpoint,
                  peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    if output:
        append_jsonl(output, result)
    return result


def _rows(path):
    import json
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _mean_nll(model, tokenizer, rows, batch_size, max_len):
    """Per-row mean completion NLL (the membership-inference score), in the order of `rows`."""
    import numpy as np

    from pqfl.train.data import completion_text, prompt_text
    from pqfl.train.evaluate import example_nll
    items = [dict(prompt=prompt_text(r), completion=completion_text(r)) for r in rows]
    nll = example_nll(model, tokenizer, items, batch_size=batch_size, max_len=max_len, eos_id=tokenizer.eos_token_id,
                      pad_id=tokenizer.pad_token_id)
    return np.array([s / c for s, c in nll])


def base_nll_job(model_path: str, pool_path: str, holdout_path: str, output: str, batch_size: int = 8,
                 max_len: int = 512) -> dict:
    """Per-example NLL of the untrained base model on members (pool) and non-members (holdout): the
    reference of the calibrated membership-inference attack (computed once, reused by every run)."""
    import numpy as np

    from pqfl.train.local import load_model
    pool, holdout = _rows(pool_path), _rows(holdout_path)
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=0, gradient_checkpointing=False)
    members, others = _mean_nll(model, tokenizer, pool, batch_size, max_len), _mean_nll(model, tokenizer, holdout, batch_size, max_len)
    np.savez(output, member_ids=np.array([r["id"] for r in pool]), member_nll=members,
             nonmember_ids=np.array([r["id"] for r in holdout]), nonmember_nll=others)
    return dict(members=len(pool), nonmembers=len(holdout), member_mean=float(members.mean()),
                nonmember_mean=float(others.mean()), output=output)


def final_scoring(model_path: str, pool_path: str, holdout_path: str, validation_path: str, canaries_path: str,
                  manifest_path: str, base_nll_path: str, checkpoint: str | None = None, output: str | None = None,
                  tag: str = "final", batch_size: int = 8, max_len: int = 512, n_validation: int | None = None) -> dict:
    """Utility, canary exposure/extraction and membership inference of a model (base if no checkpoint).

    Non-members whose text duplicates a member (MedMCQA repeats questions under different ids) are
    excluded from the membership metrics; the per-example NLLs are saved next to `output` so that
    the attacks can be recomputed without the GPU."""
    import json
    import os
    import time

    import numpy as np
    import torch

    from pqfl.attacks.membership import evaluate as mia_evaluate
    from pqfl.train.data import duplicates_of
    from pqfl.train.evaluate import append_jsonl, evaluate_model
    from pqfl.train.local import load_flat, load_model
    from pqfl.train.scoring import canary_report, canary_scores
    _reset_peak_memory()
    started = time.time()
    pool, holdout, validation = _rows(pool_path), _rows(holdout_path), _rows(validation_path)[:n_validation]
    clean = ~np.array(duplicates_of(holdout, pool))
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=0, gradient_checkpointing=False)
    if checkpoint:
        load_flat(model, torch.load(checkpoint)["vector"])
    result = dict(tag=tag, checkpoint=checkpoint)
    # held-out perplexity without the non-members that duplicate a member (the variant with them is kept)
    result["utility"] = evaluate_model(model, tokenizer, validation_rows=validation,
                                       holdout_rows=[r for r, keep in zip(holdout, clean) if keep],
                                       batch_size=batch_size, max_len=max_len, eos_id=tokenizer.eos_token_id,
                                       pad_id=tokenizer.pad_token_id)
    members, others = _mean_nll(model, tokenizer, pool, batch_size, max_len), _mean_nll(model, tokenizer, holdout, batch_size, max_len)
    base = np.load(base_nll_path)
    by_id = lambda ids, values: dict(zip(ids.tolist(), values))
    base_members, base_others = by_id(base["member_ids"], base["member_nll"]), by_id(base["nonmember_ids"], base["nonmember_nll"])
    reference_m = np.array([base_members[r["id"]] for r in pool])
    reference_o = np.array([base_others[r["id"]] for r in holdout])
    result["membership"] = dict(
        pooled=mia_evaluate(members, others[clean], reference_member_nll=reference_m,
                            reference_nonmember_nll=reference_o[clean]),
        pooled_with_duplicates=mia_evaluate(members, others, reference_member_nll=reference_m,
                                            reference_nonmember_nll=reference_o),
        excluded_duplicate_nonmembers=int((~clean).sum()))
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    for site in range(len(manifest["pool"])):
        in_site = lambda rows, ids: np.array([r["id"] in ids for r in rows])
        keep_m = in_site(pool, set(manifest["pool"][site]))
        keep_o = in_site(holdout, set(manifest["holdout"][site])) & clean
        result["membership"][f"site{site}"] = mia_evaluate(members[keep_m], others[keep_o], fprs=(0.01,),
                                                           reference_member_nll=reference_m[keep_m],
                                                           reference_nonmember_nll=reference_o[keep_o])
    canaries = json.load(open(canaries_path, encoding="utf-8"))
    result["canaries"] = canary_report(canary_scores(model, tokenizer, canaries, pad_id=tokenizer.pad_token_id))
    result["seconds"] = time.time() - started
    result["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    if output:
        append_jsonl(output, result)
        np.savez(os.path.join(os.path.dirname(output) or ".", f"{tag}-nll.npz"),
                 member_ids=np.array([r["id"] for r in pool]), member_nll=members,
                 nonmember_ids=np.array([r["id"] for r in holdout]), nonmember_nll=others,
                 nonmember_duplicate=~clean)
    return result


def lr_probe(model_path: str, rows_path: str, manifest_path: str, canaries_path: str, validation_path: str,
             holdout_path: str, lr: float, site: int = 0, micro_batch: int = 4, accumulation: int = 8,
             max_len: int = 512, seed: int = 0, n_validation: int = 500, n_holdout: int = 300) -> dict:
    """One local epoch on one site (natural pool plus its canary copies) at a given learning rate, then a quick
    evaluation: how fast the loss falls and what it does to utility (pilot tuning of the learning rate)."""
    import json
    import time

    import torch

    from pqfl.train.data import site_examples
    from pqfl.train.evaluate import evaluate_model
    from pqfl.train.local import load_model, train_local
    rows = {r["id"]: r for r in _rows(rows_path)}
    pool = json.load(open(manifest_path, encoding="utf-8"))["pool"][site]
    items = site_examples(site, pool, rows, json.load(open(canaries_path, encoding="utf-8")))
    _reset_peak_memory()
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=seed)
    started = time.time()
    out = train_local(model, items, tokenizer, epochs=1, micro_batch=micro_batch, accumulation=accumulation, lr=lr,
                      max_len=max_len, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, seed=seed)
    model.gradient_checkpointing_disable()
    evaluation = evaluate_model(model, tokenizer, validation_rows=_rows(validation_path)[:n_validation],
                                holdout_rows=_rows(holdout_path)[:n_holdout], batch_size=8, max_len=max_len,
                                eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id)
    losses = out["losses"]
    return dict(lr=lr, items=len(items), steps=out["steps"], train_seconds=out["seconds"],
                loss_first5=sum(losses[:5]) / 5, loss_last5=sum(losses[-5:]) / 5, finite=all(l == l for l in losses),
                accuracy=evaluation["validation"]["accuracy"], holdout_nll=evaluation["holdout"]["mean_nll"],
                holdout_perplexity=evaluation["holdout"]["perplexity"], total_seconds=time.time() - started,
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30)


class _SingleSite:
    """Aggregator of the centralized reference Z: one site, the 'sum' is its own update."""
    sites = ["Z"]

    def aggregate_round(self, round_idx, updates):
        self._bytes = {"Z": 0}
        return updates["Z"]

    def stats(self):
        from types import SimpleNamespace
        return SimpleNamespace(bytes_sent=self._bytes)


def pilot_run(model_path: str, rows_path: str, validation_path: str, holdout_path: str, manifest_path: str,
              canaries_path: str, base_nll_path: str, out_dir: str, scheme: str = "N", epsilon: float | None = None,
              delta: float = 1e-5, rounds: int = 10, lr: float = 2e-4, seed: int = 0, micro_batch: int = 4,
              logical_batch: int = 32, clip: float | None = None, clamp_bound: float | None = None,
              eval_validation: int = 500, eval_holdout: int = 300, max_len: int = 512,
              oqs_path: str | None = None, secret_randomness: bool = False, save_updates: bool = False) -> dict:
    """One full configuration: three sites, `rounds` federated rounds with the given aggregation scheme and
    (optionally) example-level DP-SGD at the given epsilon, per-round quick evaluation, final scoring.
    secret_randomness draws the DP noise and batches from os.urandom (not reproducible, see dpsgd).
    save_updates writes the per-site updates of every round to out_dir/updates (sensitive, see run_federated)."""
    if epsilon is not None and (clip is None or not clip > 0):
        raise ValueError("A DP run (epsilon set) needs a positive clipping norm `clip`")
    if epsilon is not None and scheme == "Z":
        raise ValueError("Z is the non-private centralized reference; its DP accounting is not implemented")
    if logical_batch % micro_batch:
        raise ValueError("logical_batch must be a multiple of micro_batch")
    if epsilon is None and (clip is not None or secret_randomness):
        raise ValueError("clip and secret_randomness apply only to DP runs (epsilon set)")
    import json
    import os
    import time

    import torch

    from pqfl.dp import calibrate_sigma
    from pqfl.train.data import duplicates_of, site_examples
    from pqfl.train.dpsgd import train_local_dp
    from pqfl.train.evaluate import append_jsonl, evaluate_model
    from pqfl.train.federated import latest_checkpoint, run_federated
    from pqfl.train.local import load_model, train_local
    _reset_peak_memory()
    os.makedirs(out_dir, exist_ok=True)
    rows = {r["id"]: r for r in _rows(rows_path)}
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    canaries = json.load(open(canaries_path, encoding="utf-8"))
    sites = [f"S{i}" for i in range(len(manifest["pool"]))]
    site_items = {s: site_examples(i, manifest["pool"][i], rows, canaries) for i, s in enumerate(sites)}
    items_per_site = len(site_items[sites[0]])
    if epsilon is not None and len({len(v) for v in site_items.values()}) != 1:
        raise ValueError("The DP calibration assumes sites of equal size")
    if scheme == "Z":      # centralized reference: the union of the data of the three sites, one 'site'
        site_items = {"Z": [i for s in sites for i in site_items[s]]}
        sites = ["Z"]
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=seed)
    common = dict(epochs=1, micro_batch=micro_batch, lr=lr, max_len=max_len, eos_id=tokenizer.eos_token_id,
                  pad_id=tokenizer.pad_token_id)
    config = dict(scheme=scheme, epsilon=epsilon, delta=delta, rounds=rounds, lr=lr, seed=seed, micro_batch=micro_batch,
                  logical_batch=logical_batch, clip=clip, clamp_bound=clamp_bound, items_per_site=items_per_site,
                  sites=sites, secret_randomness=secret_randomness)
    if epsilon is None:
        train_function, kwargs = train_local, dict(common, accumulation=logical_batch // micro_batch)
    else:
        rate = logical_batch / items_per_site
        steps = rounds * max(1, round(items_per_site / logical_batch))
        sigma = calibrate_sigma(epsilon, delta, rate, steps)
        config.update(sigma=sigma, sampling_rate=rate, total_steps=steps)
        train_function, kwargs = train_local_dp, dict(common, logical_batch=logical_batch, clip=clip, sigma=sigma,
                                                      secret_randomness=secret_randomness)
    if scheme == "Z":
        aggregator = _SingleSite()
    elif scheme == "N" and clamp_bound is None:
        from pqfl.aggregators.plain_fl import PlainFL
        aggregator = PlainFL()
        aggregator.setup(sites)
    else:
        # Secure schemes (and N when a clamp is given) are built like in the cost campaign; they need a
        # per-element limit B_inf = clamp_bound, and liboqs for the KEM and ML-DSA.
        if clamp_bound is None:
            raise ValueError("clamp_bound (B_inf) is required for the secure schemes")
        if oqs_path:
            os.environ["OQS_INSTALL_PATH"] = oqs_path
        from pqfl.train.aggregation import build_training_aggregator
        aggregator = build_training_aggregator(scheme, sites, clip_bound=clamp_bound, out_dir=out_dir, seed=seed,
                                               dp_level="none" if epsilon is None else f"eps{epsilon:g}",
                                               run_id=os.path.basename(out_dir.rstrip("/")))
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    validation, holdout = _rows(validation_path)[:eval_validation], _rows(holdout_path)
    holdout = [r for r, dup in zip(holdout, duplicates_of(holdout, list(rows.values()))) if not dup][:eval_holdout]

    def quick_evaluation(model_, round_idx):
        model_.gradient_checkpointing_disable()
        result = evaluate_model(model_, tokenizer, validation_rows=validation, holdout_rows=holdout, batch_size=8,
                                max_len=max_len, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id)
        model_.gradient_checkpointing_enable()
        record = dict(round=round_idx, accuracy=result["validation"]["accuracy"],
                      holdout_perplexity=result["holdout"]["perplexity"])
        append_jsonl(os.path.join(out_dir, "rounds.jsonl"), record)
        return record

    started = time.time()
    out = run_federated(model, tokenizer, site_items, aggregator, rounds=rounds, train_kwargs=kwargs, seed=seed,
                        checkpoint_dir=os.path.join(out_dir, "checkpoints"), clamp_bound=clamp_bound,
                        evaluate=quick_evaluation, log=print, train_function=train_function, config=config,
                        updates_dir=os.path.join(out_dir, "updates") if save_updates else None)
    train_seconds = time.time() - started
    train_peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    del model
    final = final_scoring(model_path, rows_path, holdout_path, validation_path, canaries_path, manifest_path,
                          base_nll_path, checkpoint=latest_checkpoint(os.path.join(out_dir, "checkpoints")),
                          output=os.path.join(out_dir, "final.jsonl"), tag=os.path.basename(out_dir.rstrip("/")))
    summary = dict(config=config, train_seconds=train_seconds, train_peak_gpu_gb=train_peak_gb,
                   scoring_peak_gpu_gb=final["peak_gpu_gb"], rounds=len(out["history"]),
                   accuracy=final["utility"]["validation"]["accuracy"], perplexity=final["utility"]["holdout"]["perplexity"],
                   mia_loss_auc=final["membership"]["pooled"]["loss"]["auc"],
                   mia_reference_auc=final["membership"]["pooled"]["reference"]["auc"], canaries=final["canaries"])
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)
    return summary


def dp_smoke(model_path: str, rows_path: str, manifest_path: str, n_items: int = 64, logical_batch: int = 16,
             micro_batch: int = 4, clip: float = 1.0, sigma: float = 0.65, lr: float = 2e-4, max_len: int = 512,
             seed: int = 0) -> dict:
    """DP-SGD on the real model: (1) the hook per-example gradient norms equal the norms of separate backward
    passes (with gradient checkpointing on); (2) a few DP steps run, with memory and speed."""
    import json

    import torch

    from pqfl.train.data import encode_pair, make_batch, site_examples
    from pqfl.train.dpsgd import PerSampleGrads, example_loss_sum, train_local_dp
    from pqfl.train.local import load_model, trainable
    rows = {r["id"]: r for r in _rows(rows_path)}
    pool = json.load(open(manifest_path, encoding="utf-8"))["pool"][0][:n_items]
    items = site_examples(0, pool, rows, None)
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=seed)
    # make the adapter non-zero so that both LoRA matrices receive gradient
    with torch.no_grad():
        for _, p in trainable(model):
            p.add_(torch.randn_like(p) * 0.01)
    device = next(model.parameters()).device
    encoded = [encode_pair(tokenizer, i["prompt"], i["completion"], max_len=max_len, eos_id=tokenizer.eos_token_id)
               for i in items[:3]]
    batch = make_batch(encoded, tokenizer.pad_token_id)
    ids, mask, labels = (torch.tensor(batch[k], device=device) for k in ("input_ids", "attention_mask", "labels"))
    model.train()
    with PerSampleGrads(model) as hooks:
        example_loss_sum(model(input_ids=ids, attention_mask=mask).logits, labels).backward()
        hooked = hooks.squared_norms().sqrt().cpu()
    naive = []
    for b in range(len(encoded)):
        model.zero_grad(set_to_none=True)
        e = make_batch([encoded[b]], tokenizer.pad_token_id)
        i2, m2, l2 = (torch.tensor(e[k], device=device) for k in ("input_ids", "attention_mask", "labels"))
        example_loss_sum(model(input_ids=i2, attention_mask=m2).logits, l2).backward()
        naive.append(sum(float(p.grad.float().pow(2).sum()) for _, p in trainable(model)) ** 0.5)
    model.zero_grad(set_to_none=True)
    naive = torch.tensor(naive)
    torch.cuda.reset_peak_memory_stats()
    out = train_local_dp(model, items, tokenizer, epochs=1, micro_batch=micro_batch, logical_batch=logical_batch, lr=lr,
                         clip=clip, sigma=sigma, max_len=max_len, eos_id=tokenizer.eos_token_id,
                         pad_id=tokenizer.pad_token_id, seed=seed)
    return dict(hook_norms=hooked.tolist(), naive_norms=naive.tolist(),
                max_relative_error=float(((hooked - naive).abs() / naive).max()), steps=out["steps"],
                losses=out["losses"], clipped_fraction=out["clipped_fraction"], batch_sizes=out["batch_sizes"],
                grad_norm_median=out["grad_norm_median"], tokens=out["tokens"], seconds=out["seconds"],
                tokens_per_s=out["tokens"] / out["seconds"], peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30)


def inversion_probe(model_path: str, rows_path: str, manifest_path: str, checkpoint: str | None = None,
                    n_examples: int = 2, completion_tokens: int = 16, steps: int = 20, lr: float = 0.1,
                    temperature: float = 1.0, max_prompt_tokens: int = 40, seed: int = 0,
                    reference_kernels: bool = True) -> dict:
    """Scenario R0 on the real model: the attacker knows the prompt and sees the LoRA gradient of one example
    (checkpoint state, so that both LoRA matrices carry gradient); it recovers `completion_tokens` tokens.
    With a small `steps` it only measures feasibility and cost per iteration."""
    import json

    from pqfl.attacks.gradient_inversion import restore_kernels, use_reference_kernels
    rows = {r["id"]: r for r in _rows(rows_path)}
    pool = json.load(open(manifest_path, encoding="utf-8"))["pool"][0]
    _reset_peak_memory()
    saved = use_reference_kernels() if reference_kernels else {}         # double backward needs torch-only kernels
    try:
        return _inversion_probe(model_path, rows, pool, checkpoint, n_examples, completion_tokens, steps, lr,
                                temperature, max_prompt_tokens, seed, reference_kernels, sorted(saved))
    finally:
        restore_kernels(saved)


def _inversion_probe(model_path, rows, pool, checkpoint, n_examples, completion_tokens, steps, lr, temperature,
                     max_prompt_tokens, seed, reference_kernels, replaced) -> dict:
    import time

    import torch

    from pqfl.attacks.gradient_inversion import flat_gradient, invert
    from pqfl.attacks.inversion import bag_recall, exceeds_null, null_scores, rouge_l_f1
    from pqfl.train.data import completion_text, prompt_text
    from pqfl.train.dpsgd import example_loss_sum
    from pqfl.train.local import load_flat, load_model
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=seed, gradient_checkpointing=False,
                                  attn_implementation="eager" if reference_kernels else None)
    if checkpoint:
        load_flat(model, torch.load(checkpoint)["vector"])
    embedding = model.get_input_embeddings().weight.detach()
    corpus = [t for pid in pool[:300] for t in tokenizer.encode(completion_text(rows[pid]))]
    device = next(model.parameters()).device
    results = []
    for pid in pool:
        row = rows[pid]
        prompt = list(tokenizer.encode(prompt_text(row)))
        completion = list(tokenizer.encode(completion_text(row)))[:completion_tokens]
        if len(prompt) > max_prompt_tokens or len(completion) < completion_tokens:
            continue
        ids = torch.tensor([prompt + completion], device=device)
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        model.train()
        target = flat_gradient(model, ids, labels, example_loss_sum)
        started = time.time()
        try:
            out = invert(model, embedding, target, prompt, len(completion), steps=steps, lr=lr,
                         temperature=temperature, seed=seed)
        except torch.cuda.OutOfMemoryError:
            raise                                               # a resource failure, not an attack outcome
        except (RuntimeError, NotImplementedError) as exc:      # e.g. a fused kernel without double backward
            return dict(status="attack_not_runnable", error=f"{type(exc).__name__}: {exc}"[:400],
                        examples_done=len(results), results=results)
        seconds = time.time() - started
        null = null_scores(completion, bag_recall, corpus, length=len(completion), draws=300, seed=seed)
        recall = bag_recall(completion, out["ids"])
        results.append(dict(id=pid, prompt_tokens=len(prompt), completion_tokens=len(completion), steps=steps,
                            seconds=seconds, seconds_per_step=seconds / steps, cosine_distance=out["distance"],
                            bag_recall=recall, rouge_l=rouge_l_f1(completion, out["ids"]),
                            null_p95=float(sorted(null)[int(0.95 * len(null))]), above_null=exceeds_null(recall, null),
                            peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
        if len(results) >= n_examples:
            break
    return dict(status="ok", examples_done=len(results), results=results, checkpoint=checkpoint,
                reference_kernels=replaced)


def inversion_selfcheck(model_path: str, rows_path: str, manifest_path: str, checkpoint: str | None = None,
                        n_examples: int = 3, completion_tokens: int = 16, max_prompt_tokens: int = 40,
                        seed: int = 0, reference_kernels: bool = True) -> dict:
    """Is the objective of the inversion attack right on the real model? For each example the cosine between the
    observed gradient and the gradient reproduced through the soft-embedding path with (a) the TRUE tokens as exact
    one-hot distributions (must be close to 1, up to bf16 noise), (b) one wrong token, (c) fully random tokens.
    A high (a) and low (c) means the objective is informative and only the optimization is hard."""
    import json

    from pqfl.attacks.gradient_inversion import restore_kernels, use_reference_kernels
    rows = {r["id"]: r for r in _rows(rows_path)}
    pool = json.load(open(manifest_path, encoding="utf-8"))["pool"][0]
    _reset_peak_memory()
    saved = use_reference_kernels() if reference_kernels else {}
    try:
        return _inversion_selfcheck(model_path, rows, pool, checkpoint, n_examples, completion_tokens,
                                    max_prompt_tokens, seed, reference_kernels)
    finally:
        restore_kernels(saved)


def _inversion_selfcheck(model_path, rows, pool, checkpoint, n_examples, completion_tokens, max_prompt_tokens, seed,
                         reference_kernels) -> dict:
    import random

    import torch
    import torch.nn.functional as F

    from pqfl.attacks.gradient_inversion import flat_gradient, soft_loss
    from pqfl.train.data import completion_text, prompt_text
    from pqfl.train.dpsgd import example_loss_sum
    from pqfl.train.local import load_flat, load_model, trainable
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=seed, gradient_checkpointing=False,
                                  attn_implementation="eager" if reference_kernels else None)
    if checkpoint:
        load_flat(model, torch.load(checkpoint)["vector"])
    embedding = model.get_input_embeddings().weight.detach()
    vocab, device = embedding.shape[0], next(model.parameters()).device
    params = [p for _, p in trainable(model)]
    rng = random.Random(seed)
    model.train()

    def soft_gradient(prompt, completion):
        probs = torch.cat([F.one_hot(torch.tensor(prompt, device=device), vocab).float(),
                           F.one_hot(torch.tensor(completion, device=device), vocab).float()], dim=0)
        embeddings = (probs @ embedding.float()).unsqueeze(0).to(embedding.dtype)
        loss = soft_loss(model(inputs_embeds=embeddings).logits, probs, len(prompt))
        return torch.cat([g.reshape(-1).float() for g in torch.autograd.grad(loss, params)]).detach()

    out = []
    for pid in pool:
        row = rows[pid]
        prompt = list(tokenizer.encode(prompt_text(row)))
        completion = list(tokenizer.encode(completion_text(row)))[:completion_tokens]
        if len(prompt) > max_prompt_tokens or len(completion) < completion_tokens:
            continue
        ids = torch.tensor([prompt + completion], device=device)
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        target = flat_gradient(model, ids, labels, example_loss_sum)
        cosine = lambda tokens: float(F.cosine_similarity(soft_gradient(prompt, tokens), target, dim=0))
        wrong_one = list(completion)
        wrong_one[rng.randrange(len(completion))] = rng.randrange(vocab)
        out.append(dict(id=pid, true_tokens=cosine(completion), one_wrong_token=cosine(wrong_one),
                        random_tokens=cosine([rng.randrange(vocab) for _ in completion]),
                        target_norm=float(target.norm())))
        if len(out) >= n_examples:
            break
    return dict(status="ok", checkpoint=checkpoint, examples=out, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30)


def inversion_coordinate(model_path: str, rows_path: str, manifest_path: str, checkpoint: str | None = None,
                         n_examples: int = 3, completion_tokens: int = 16, max_prompt_tokens: int = 40,
                         top_k: int = 64, passes: int = 3, seed: int = 0, skip: int = 0) -> dict:
    """Scenario R0 with the coordinate-search attack (pqfl.attacks.gradient_inversion.coordinate_search) on the real
    model, fused kernels (no double backward). Reports, per example, the recall of the model's own greedy
    continuation (no gradient used: the baseline) and of the attack result, against a random-text null.
    `skip` skips the first examples that fit (to evaluate on examples not used for tuning)."""
    import json
    import time

    import torch

    from pqfl.attacks.gradient_inversion import coordinate_search, flat_gradient
    from pqfl.attacks.inversion import bag_recall, exceeds_null, null_scores, rouge_l_f1
    from pqfl.train.data import completion_text, prompt_text
    from pqfl.train.dpsgd import example_loss_sum
    from pqfl.train.local import load_flat, load_model
    rows = {r["id"]: r for r in _rows(rows_path)}
    pool = json.load(open(manifest_path, encoding="utf-8"))["pool"][0]
    _reset_peak_memory()
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=seed, gradient_checkpointing=False)
    if checkpoint:
        load_flat(model, torch.load(checkpoint)["vector"])
    corpus = [t for pid in pool[:300] for t in tokenizer.encode(completion_text(rows[pid]))]
    device = next(model.parameters()).device
    results, seen = [], 0
    for pid in pool:
        row = rows[pid]
        prompt = list(tokenizer.encode(prompt_text(row)))
        completion = list(tokenizer.encode(completion_text(row)))[:completion_tokens]
        if len(prompt) > max_prompt_tokens or len(completion) < completion_tokens:
            continue
        seen += 1
        if seen <= skip:
            continue
        ids = torch.tensor([prompt + completion], device=device)
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        model.train()
        target = flat_gradient(model, ids, labels, example_loss_sum)
        started = time.time()
        out = coordinate_search(model, target, prompt, len(completion), example_loss_sum, top_k=top_k, passes=passes)
        seconds = time.time() - started
        null = null_scores(completion, bag_recall, corpus, length=len(completion), draws=300, seed=seed)
        recall, baseline = bag_recall(completion, out["ids"]), bag_recall(completion, out["initial_ids"])
        results.append(dict(id=pid, prompt_tokens=len(prompt), completion_tokens=len(completion), top_k=top_k,
                            passes=passes, seconds=seconds, evaluations=out["evaluations"],
                            cosine_final=out["cosine"], trace=out["trace"],
                            baseline_recall=baseline, bag_recall=recall,
                            exact_positions=sum(a == b for a, b in zip(out["ids"], completion)) / len(completion),
                            baseline_exact_positions=sum(a == b for a, b in zip(out["initial_ids"], completion))
                            / len(completion),
                            rouge_l=rouge_l_f1(completion, out["ids"]),
                            null_p95=float(sorted(null)[int(0.95 * len(null))]),
                            above_null=exceeds_null(recall, null),
                            recovered=tokenizer.decode(out["ids"]), truth=tokenizer.decode(completion)))
        if len(results) >= n_examples:
            break
    return dict(status="ok", checkpoint=checkpoint, examples_done=len(results), results=results,
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30)


def calibrate_bounds(model_path: str, rows_path: str, manifest_path: str, canaries_path: str,
                     checkpoint: str | None = None, site: int = 0, n_norm_examples: int = 256, micro_batch: int = 4,
                     logical_batch: int = 32, lr: float = 2e-4, max_len: int = 512, seed: int = 0) -> dict:
    """At a given global state: (1) the distribution of per-example gradient norms (for the DP clipping norm C)
    and (2) the quantiles of |element| of one site's local update (for the per-element limit B_inf)."""
    import json

    import numpy as np
    import torch

    from pqfl.train.data import encode_pair, make_batch, site_examples
    from pqfl.train.dpsgd import PerSampleGrads, example_loss_sum
    from pqfl.train.local import flatten, load_flat, load_model, train_local
    rows = {r["id"]: r for r in _rows(rows_path)}
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    canaries = json.load(open(canaries_path, encoding="utf-8"))
    items = site_examples(site, manifest["pool"][site], rows, canaries)
    model, tokenizer = load_model(model_path, rank=8, alpha=16, seed=seed)
    if checkpoint:
        load_flat(model, torch.load(checkpoint)["vector"])
    start = flatten(model)
    device = next(model.parameters()).device
    natural = [i for i in items if i["id"] and not i["id"].startswith("canary-")][:n_norm_examples]
    encoded = [encode_pair(tokenizer, i["prompt"], i["completion"], max_len=max_len, eos_id=tokenizer.eos_token_id)
               for i in natural]
    norms = []
    model.train()
    with PerSampleGrads(model) as hooks:
        for i in range(0, len(encoded), micro_batch):
            batch = make_batch(encoded[i:i + micro_batch], tokenizer.pad_token_id)
            ids, mask, labels = (torch.tensor(batch[k], device=device) for k in ("input_ids", "attention_mask", "labels"))
            hooks.grads = {}
            example_loss_sum(model(input_ids=ids, attention_mask=mask).logits, labels).backward()
            norms += hooks.squared_norms().sqrt().tolist()
            model.zero_grad(set_to_none=True)
    train_local(model, items, tokenizer, epochs=1, micro_batch=micro_batch, accumulation=logical_batch // micro_batch,
                lr=lr, max_len=max_len, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, seed=seed)
    delta = (flatten(model) - start).abs().numpy()
    quantile = lambda values, qs: {str(q): float(np.percentile(values, q)) for q in qs}
    return dict(checkpoint=checkpoint, site=site, examples_for_norms=len(norms),
                example_grad_norm=quantile(norms, (5, 25, 50, 75, 95, 99)),
                update_abs_quantiles=quantile(delta, (50, 90, 99, 99.9, 99.99)), update_max=float(delta.max()),
                update_norm=float(np.linalg.norm(delta)))
