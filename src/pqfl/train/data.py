"""MedMCQA examples, loss masks and per-site datasets (canaries included).

Pure Python: the tokenizer is any object with `encode(text) -> list[int]` (no special
tokens added), so the logic is testable without the model stack. Prompts and canaries share
the style `Question: ... Answer:`; the loss covers only the completion (answer letter and
explanation, or the canary code), never the prompt.
"""
import re

LETTERS = "ABCD"
IGNORE = -100


def prompt_text(row: dict) -> str:
    options = "\n".join(f"{letter}. {row['op' + key]}" for letter, key in zip(LETTERS, "abcd"))
    return f"Question: {str(row['question']).strip()}\n{options}\nAnswer:"


def completion_text(row: dict, *, explanation: bool = True) -> str:
    text = f" {LETTERS[int(row['cop'])]}"
    exp = row.get("exp")
    if explanation and isinstance(exp, str) and exp.strip():   # NaN and None are skipped
        text += f"\nExplanation: {exp.strip()}"
    return text


def encode_pair(tokenizer, prompt: str, completion: str, *, max_len: int, eos_id: int,
                min_completion: int = 2) -> dict:
    """input_ids and labels with the prompt masked.

    Long examples lose the end of the completion first. If the prompt alone leaves less than
    `min_completion` tokens, the prompt keeps its tail, so the `Answer:` cue is never cut.
    """
    prompt_ids = list(tokenizer.encode(prompt))
    completion_ids = list(tokenizer.encode(completion)) + [eos_id]
    room = max_len - min(len(completion_ids), min_completion)
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]
    completion_ids = completion_ids[:max_len - len(prompt_ids)]
    return dict(input_ids=prompt_ids + completion_ids,
                labels=[IGNORE] * len(prompt_ids) + completion_ids,
                prompt_length=len(prompt_ids))


def make_batch(examples: list[dict], pad_id: int) -> dict:
    """Right-padded lists; labels padded with the ignore index, attention mask included."""
    width = max(len(e["input_ids"]) for e in examples)
    pad = lambda seq, value: list(seq) + [value] * (width - len(seq))
    return dict(input_ids=[pad(e["input_ids"], pad_id) for e in examples],
                labels=[pad(e["labels"], IGNORE) for e in examples],
                attention_mask=[pad([1] * len(e["input_ids"]), 0) for e in examples])


def site_examples(site: int, pool_ids: list[str], rows_by_id: dict, canaries: dict | None,
                  *, explanation: bool = True) -> list[dict]:
    """Natural pool examples followed by the canary copies of this site (level r = r copies).
    Each item: id, kind, prompt, completion. Order is deterministic; shuffle at training time."""
    items = [dict(id=rid, kind="natural", prompt=prompt_text(rows_by_id[rid]),
                  completion=completion_text(rows_by_id[rid], explanation=explanation))
             for rid in pool_ids]
    if canaries is not None:
        from pqfl.data.canaries import completion, prompt
        for canary in canaries["canaries"]:
            if canary["site"] == site:
                items += [dict(id=f"canary-{canary['case_id']}", kind="canary", prompt=prompt(canary),
                               completion=completion(canary["secret"]))] * canary["repetitions"]
    return items


def text_key(row: dict) -> str:
    """Normalized question and options: MedMCQA repeats some questions under different ids."""
    clean = lambda value: re.sub(r"\W+", " ", str(value).lower()).strip()
    return "|".join([clean(row["question"])] + [clean(row["op" + key]) for key in "abcd"])


def duplicates_of(rows: list[dict], reference_rows: list[dict]) -> list[bool]:
    """For each row, whether its text_key also occurs among `reference_rows` (e.g. a non-member that is
    textually a member: it must not count as a non-member in membership inference)."""
    keys = {text_key(r) for r in reference_rows}
    return [text_key(r) in keys for r in rows]
