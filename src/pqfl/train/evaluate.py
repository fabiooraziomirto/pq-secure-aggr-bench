"""Evaluation: option accuracy from letter logits, per-example completion NLL and perplexity.

Accuracy compares the logits of the four option-letter tokens at the position of the `Answer:`
cue (right padding is safe: the models are causal, so we read the last real position). The
completion NLL uses the training mask, so it is the loss the model is trained on and the score
reused by the membership-inference attack.
"""
import json
import math

import torch
import torch.nn.functional as F

from pqfl.train.data import IGNORE, completion_text, encode_pair, make_batch, prompt_text

LETTERS = (" A", " B", " C", " D")


def single_token_ids(tokenizer, texts=LETTERS) -> list[int]:
    ids = []
    for text in texts:
        encoded = list(tokenizer.encode(text))
        if len(encoded) != 1:
            raise ValueError(f"{text!r} is not a single token")
        ids.append(encoded[0])
    return ids


@torch.no_grad()
def option_predictions(model, tokenizer, rows: list[dict], letter_ids: list[int], *, batch_size: int,
                       max_len: int, pad_id: int) -> list[int]:
    """Index (0-3) of the option with the highest letter logit, for each row."""
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    for start in range(0, len(rows), batch_size):
        encoded = [list(tokenizer.encode(prompt_text(r)))[-max_len:] for r in rows[start:start + batch_size]]
        width = max(len(e) for e in encoded)
        ids = torch.tensor([e + [pad_id] * (width - len(e)) for e in encoded], device=device)
        mask = torch.tensor([[1] * len(e) + [0] * (width - len(e)) for e in encoded], device=device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        last = torch.tensor([len(e) - 1 for e in encoded], device=device)
        picked = logits[torch.arange(len(encoded), device=device), last][:, letter_ids].float()
        predictions += picked.argmax(dim=1).tolist()
    return predictions


def accuracy_report(rows: list[dict], predictions: list[int]) -> dict:
    """Overall accuracy and accuracy per subject (the validation mix differs from the training mix)."""
    correct = [int(p == int(r["cop"])) for r, p in zip(rows, predictions)]
    by_subject = {}
    for row, hit in zip(rows, correct):
        entry = by_subject.setdefault(row.get("subject_name") or "Unknown", [0, 0])
        entry[0] += hit
        entry[1] += 1
    return dict(n=len(rows), accuracy=sum(correct) / len(correct),
                by_subject={k: dict(n=n, accuracy=h / n) for k, (h, n) in sorted(by_subject.items())},
                macro_accuracy=sum(h / n for h, n in by_subject.values()) / len(by_subject))


@torch.no_grad()
def example_nll(model, tokenizer, items: list[dict], *, batch_size: int, max_len: int, eos_id: int,
                pad_id: int) -> list[tuple[float, int]]:
    """(summed NLL, number of scored tokens) of each item's completion given its prompt."""
    device = next(model.parameters()).device
    model.eval()
    encoded = [encode_pair(tokenizer, i["prompt"], i["completion"], max_len=max_len, eos_id=eos_id) for i in items]
    out = []
    for start in range(0, len(encoded), batch_size):
        batch = make_batch(encoded[start:start + batch_size], pad_id)
        ids = torch.tensor(batch["input_ids"], device=device)
        mask = torch.tensor(batch["attention_mask"], device=device)
        labels = torch.tensor(batch["labels"], device=device)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        shifted = labels[:, 1:]
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1),
                                 ignore_index=IGNORE, reduction="none").view(shifted.shape)
        counts = (shifted != IGNORE).sum(dim=1)
        out += [(float(s), int(c)) for s, c in zip(losses.sum(dim=1), counts)]
    return out


def perplexity(per_example: list[tuple[float, int]]) -> dict:
    total, tokens = sum(s for s, _ in per_example), sum(c for _, c in per_example)
    return dict(n=len(per_example), tokens=tokens, mean_nll=total / tokens, perplexity=math.exp(total / tokens))


def evaluate_model(model, tokenizer, *, validation_rows: list[dict], holdout_rows: list[dict], batch_size: int,
                   max_len: int, eos_id: int, pad_id: int) -> dict:
    letter_ids = single_token_ids(tokenizer)
    predictions = option_predictions(model, tokenizer, validation_rows, letter_ids, batch_size=batch_size,
                                     max_len=max_len, pad_id=pad_id)
    items = [dict(prompt=prompt_text(r), completion=completion_text(r)) for r in holdout_rows]
    nll = example_nll(model, tokenizer, items, batch_size=batch_size, max_len=max_len, eos_id=eos_id, pad_id=pad_id)
    return dict(validation=accuracy_report(validation_rows, predictions), holdout=perplexity(nll))


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
