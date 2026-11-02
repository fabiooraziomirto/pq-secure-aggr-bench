"""Local LoRA training of one site (torch).

The loss is computed here, not by the model: cross-entropy over the completion tokens only
(prompt labels are -100), summed and divided by the number of completion tokens of the whole
logical batch, so accumulation over micro-batches equals one big batch. The optimizer is fresh
at every call (FedAvg with local AdamW, state reset each round). The LoRA vector exchanged with
the aggregators is the float32 concatenation of the trainable parameters in a fixed order.
"""
import random
import time

import torch
import torch.nn.functional as F

from pqfl.train.data import IGNORE, encode_pair, make_batch


def load_model(path: str, *, rank: int, alpha: int, seed: int, gradient_checkpointing: bool = True,
               attn_implementation: str | None = None):
    """bf16 base model with LoRA on every text Linear module (lm_head excluded) and its tokenizer.
    attn_implementation (e.g. "eager") is only for the gradient-inversion probe, which needs double backward."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    extra = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0}, **extra)
    suffixes = sorted({name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head")})
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    torch.manual_seed(seed)                      # LoRA A init; B starts at zero
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, target_modules=suffixes,
                                             task_type="CAUSAL_LM"))
    return model, tokenizer


def trainable(model):
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def count_trainable(model) -> int:
    return sum(p.numel() for _, p in trainable(model))


def flatten(model) -> torch.Tensor:
    """Float32 CPU vector of the trainable parameters in module order."""
    return torch.cat([p.detach().float().cpu().reshape(-1) for _, p in trainable(model)])


def load_flat(model, vector: torch.Tensor) -> None:
    if vector.numel() != count_trainable(model):
        raise ValueError("Vector length does not match the trainable parameters")
    offset = 0
    with torch.no_grad():
        for _, p in trainable(model):
            p.copy_(vector[offset:offset + p.numel()].view_as(p).to(device=p.device, dtype=p.dtype))
            offset += p.numel()


def completion_loss(logits: torch.Tensor, labels: torch.Tensor):
    """(summed cross-entropy, number of scored tokens) of next-token prediction on the completion."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    total = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1),
                            ignore_index=IGNORE, reduction="sum")
    return total, int((shift_labels != IGNORE).sum())


def train_local(model, items: list[dict], tokenizer, *, epochs: int, micro_batch: int, accumulation: int,
                lr: float, max_len: int, eos_id: int, pad_id: int, seed: int, weight_decay: float = 0.0) -> dict:
    """Train on `items` (id, prompt, completion). One optimizer step per logical batch of
    micro_batch * accumulation examples; order shuffled per epoch from `seed`."""
    device = next(model.parameters()).device
    encoded = [encode_pair(tokenizer, i["prompt"], i["completion"], max_len=max_len, eos_id=eos_id) for i in items]
    params = [p for _, p in trainable(model)]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    model.train()
    losses, tokens, started = [], 0, time.time()
    step_seconds, step_tokens = [], []
    cuda = device.type == "cuda"
    logical = micro_batch * accumulation
    for _ in range(epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for start in range(0, len(order), logical):
            group = order[start:start + logical]
            scored = sum(sum(1 for x in encoded[j]["labels"][1:] if x != IGNORE) for j in group)
            if scored == 0:
                continue
            batch_loss, step_start, tokens_before = 0.0, time.time(), tokens
            for i in range(0, len(group), micro_batch):
                batch = make_batch([encoded[j] for j in group[i:i + micro_batch]], pad_id)
                ids = torch.tensor(batch["input_ids"], device=device)
                mask = torch.tensor(batch["attention_mask"], device=device)
                labels = torch.tensor(batch["labels"], device=device)
                output = model(input_ids=ids, attention_mask=mask)
                total, _ = completion_loss(output.logits if hasattr(output, "logits") else output, labels)
                (total / scored).backward()
                batch_loss += float(total.detach())
                tokens += int(mask.sum())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.synchronize()
            step_seconds.append(time.time() - step_start)
            step_tokens.append(tokens - tokens_before)
            losses.append(batch_loss / scored)
    return dict(steps=len(losses), losses=losses, tokens=tokens, seconds=time.time() - started,
                step_seconds=step_seconds, step_tokens=step_tokens)
