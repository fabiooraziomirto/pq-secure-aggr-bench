#!/usr/bin/env python3
"""Opt-in go/no-go pilot for the base model on the GPU. Never run by tests or installation.

Loads the model in bf16, counts exact LoRA parameters for candidate target sets from the
real module list, then times a few LoRA training steps on a synthetic batch with gradient
checkpointing, recording peak GPU memory and tokens/s. Needs torch, transformers and peft
in a separate environment. Synthetic token ids only: no dataset, no clinical data.
"""
import argparse
import importlib.util
import json
import time

TARGET_SETS = {
    "attn-qv": ("q_proj", "v_proj"),
    "attn-qkvo": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local path or Hugging Face id")
    parser.add_argument("--output", required=True, help="New JSON file; never overwritten")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--target", choices=[*TARGET_SETS, "all-linear-text"], required=True,
                        help="LoRA target set used for the timed steps")
    parser.add_argument("--max-peak-gb", type=float, required=True, help="Go threshold")
    parser.add_argument("--min-tokens-per-s", type=float, required=True, help="Go threshold")
    parser.add_argument("--memory-cap-gb", type=float, help="Cap this process's GPU memory so that a "
                        "shared GPU's other users are never starved; exceeding it fails this process only")
    return parser.parse_args()


def linear_modules(model):
    """(name, in, out) of every Linear in the text part (vision tower and lm_head excluded)."""
    import torch
    found = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "visual" not in name and "vision" not in name \
                and not name.endswith("lm_head"):
            found.append((name, module.in_features, module.out_features))
    return found


def lora_counts(modules, rank):
    suffix = lambda name: name.rsplit(".", 1)[-1]
    counts = {}
    for label, suffixes in TARGET_SETS.items():
        picked = [m for m in modules if suffix(m[0]) in suffixes]
        counts[label] = dict(modules=len(picked), lora_params=sum(rank * (i + o) for _, i, o in picked))
    counts["all-linear-text"] = dict(modules=len(modules),
                                     lora_params=sum(rank * (i + o) for _, i, o in modules))
    by_suffix = {}
    for name, i, o in modules:
        entry = by_suffix.setdefault(suffix(name), dict(count=0, shapes=set()))
        entry["count"] += 1
        entry["shapes"].add((i, o))
    summary = {key: dict(count=v["count"], shapes=sorted(v["shapes"])) for key, v in by_suffix.items()}
    return counts, summary


def main():
    args = parse_args()
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible")
    free_before, total = torch.cuda.mem_get_info()
    if args.memory_cap_gb:
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_cap_gb * 2 ** 30 / total))
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0})
    modules = linear_modules(model)
    counts, summary = lora_counts(modules, args.rank)
    targets = (list(TARGET_SETS[args.target]) if args.target in TARGET_SETS
               else sorted({name.rsplit(".", 1)[-1] for name, _, _ in modules}))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.manual_seed(0)
    ids = torch.randint(1000, 100000, (args.batch_size, args.seq_len), device="cuda")  # one fixed batch
    losses = []

    def step():
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss))

    error = None
    tokens_per_s = None
    try:
        for _ in range(args.warmup_steps):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        tokens_per_s = args.batch_size * args.seq_len * args.steps / (time.perf_counter() - start)
    except torch.OutOfMemoryError as exc:
        error = "CUDA out of memory: " + str(exc)[:300]
    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    verdict = dict(peak_within_limit=error is None and peak_gb <= args.max_peak_gb,
                   speed_ok=tokens_per_s is not None and tokens_per_s >= args.min_tokens_per_s,
                   loss_decreased=len(losses) > 1 and losses[-1] < losses[0])
    result = dict(model=args.model, model_class=type(model.base_model.model).__name__,
                  torch=torch.__version__, transformers=transformers.__version__,
                  gpu=torch.cuda.get_device_name(0), gpu_total_gb=total / 2 ** 30,
                  gpu_free_before_gb=free_before / 2 ** 30,
                  kernels={name: importlib.util.find_spec(name) is not None
                           for name in ("fla", "causal_conv1d", "flash_attn")},
                  rank=args.rank, seq_len=args.seq_len, batch_size=args.batch_size, target=args.target,
                  trainable_params=trainable, lora_counts_by_target=counts, linear_modules_by_suffix=summary,
                  timed_steps=args.steps, tokens_per_s=tokens_per_s, peak_gpu_gb=peak_gb,
                  memory_cap_gb=args.memory_cap_gb, error=error,
                  losses=losses, verdict=verdict, go=error is None and all(verdict.values()))
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, default=list)
    print(json.dumps(dict(go=result["go"], verdict=verdict, tokens_per_s=tokens_per_s, peak_gpu_gb=peak_gb,
                          trainable_params=trainable)))


if __name__ == "__main__":
    main()
