"""Canary exposure and extraction, and membership-inference scores, for a trained model (torch).

Exposure follows pqfl.data.canaries: the log-probability of the true code against the codes of
the reference set, same prefix. Extraction is exact: greedy decoding from the prefix must produce
the code tokens. All canary prompts have the same token length (fixed digits, one token per
digit), which lets the greedy step run as one batch.
"""
import numpy as np
import torch

from pqfl.data.canaries import completion, exposure, prompt


@torch.no_grad()
def sequence_logprobs(model, prefix: list[int], continuations: list[list[int]], *, batch_size: int,
                      pad_id: int) -> list[float]:
    """Sum of token log-probabilities of each continuation given the shared prefix."""
    if len({len(c) for c in continuations}) != 1:
        raise ValueError("Continuations must have the same length")
    device = next(model.parameters()).device
    model.eval()
    length, results = len(continuations[0]), []
    positions = torch.arange(len(prefix) - 1, len(prefix) - 1 + length, device=device)
    for start in range(0, len(continuations), batch_size):
        chunk = continuations[start:start + batch_size]
        ids = torch.tensor([prefix + c for c in chunk], device=device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        selected = torch.log_softmax(logits[:, positions].float(), dim=-1)           # (B, length, V)
        targets = torch.tensor(chunk, device=device)
        results += selected.gather(2, targets.unsqueeze(-1)).squeeze(-1).sum(dim=1).tolist()
    return results


@torch.no_grad()
def greedy_continuations(model, prefixes: list[list[int]], steps: int) -> list[list[int]]:
    """Greedy decoding of `steps` tokens for prefixes of equal length (no cache; the prompts are short)."""
    if len({len(p) for p in prefixes}) != 1:
        raise ValueError("Prefixes must have the same length")
    device = next(model.parameters()).device
    model.eval()
    ids = torch.tensor(prefixes, device=device)
    for _ in range(steps):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, -1]
        ids = torch.cat([ids, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return ids[:, len(prefixes[0]):].tolist()


def canary_scores(model, tokenizer, manifest: dict, *, batch_size: int = 64, pad_id: int = 0) -> list[dict]:
    """Per canary: site, repetitions, exposure (bits), rank, extracted (exact greedy match)."""
    canaries = manifest["canaries"]
    prefixes = [list(tokenizer.encode(prompt(c))) for c in canaries]
    true_ids = [list(tokenizer.encode(completion(c["secret"]))) for c in canaries]
    steps = len(true_ids[0])
    generated = greedy_continuations(model, prefixes, steps) if len({len(p) for p in prefixes}) == 1 else None
    out = []
    for index, canary in enumerate(canaries):
        references = [list(tokenizer.encode(completion(r))) for r in canary["references"]]
        logprobs = sequence_logprobs(model, prefixes[index], [true_ids[index]] + references, batch_size=batch_size,
                                     pad_id=pad_id)
        rank, bits = exposure(logprobs[0], logprobs[1:])
        out.append(dict(site=canary["site"], repetitions=canary["repetitions"], rank=rank, exposure=bits,
                        extracted=bool(generated is not None and generated[index] == true_ids[index])))
    return out


def canary_report(scores: list[dict]) -> dict:
    """Per repetition level: counts, mean exposure, extraction rate; null threshold from level 0."""
    levels = {}
    for s in scores:
        levels.setdefault(s["repetitions"], []).append(s)
    report = {str(level): dict(n=len(items), mean_exposure=float(np.mean([i["exposure"] for i in items])),
                               extraction_rate=float(np.mean([i["extracted"] for i in items])))
              for level, items in sorted(levels.items())}
    null = [s["exposure"] for s in scores if s["repetitions"] == 0]
    return dict(levels=report, null_p95_exposure=float(np.percentile(null, 95)) if null else None)
