import math

import pytest

from pqfl.data.canaries import generate_canaries
from pqfl.train.data import (IGNORE, completion_text, encode_pair, make_batch, prompt_text,
                             site_examples)


class CharTokenizer:
    """Tokenizer stand-in: one id per character."""
    def encode(self, text):
        return [ord(c) for c in text]


ROW = dict(question="Which nerve?", opa="Ulnar", opb="Radial", opc="Median", opd="Axillary", cop=2,
           exp="The median nerve passes through the carpal tunnel.")


def test_prompt_and_completion_formatting():
    prompt = prompt_text(ROW)
    assert prompt == "Question: Which nerve?\nA. Ulnar\nB. Radial\nC. Median\nD. Axillary\nAnswer:"
    assert completion_text(ROW) == " C\nExplanation: The median nerve passes through the carpal tunnel."
    assert completion_text(ROW, explanation=False) == " C"
    for missing in (None, float("nan"), "   "):
        assert completion_text(dict(ROW, exp=missing)) == " C"


def test_loss_mask_covers_only_the_completion_and_eos():
    out = encode_pair(CharTokenizer(), "Q:", " C", max_len=50, eos_id=1)
    assert out["input_ids"] == [ord("Q"), ord(":"), ord(" "), ord("C"), 1]
    assert out["labels"] == [IGNORE, IGNORE, ord(" "), ord("C"), 1] and out["prompt_length"] == 2


def test_truncation_cuts_the_completion_first_and_never_the_answer_cue():
    out = encode_pair(CharTokenizer(), "Q:", " C" + "x" * 100, max_len=10, eos_id=1)
    assert len(out["input_ids"]) == 10 and out["input_ids"][:2] == [ord("Q"), ord(":")]
    long_prompt = "a" * 100 + "Answer:"
    out = encode_pair(CharTokenizer(), long_prompt, " C", max_len=12, eos_id=1)
    assert len(out["input_ids"]) <= 12
    assert out["input_ids"][:out["prompt_length"]][-7:] == [ord(c) for c in "Answer:"]
    assert sum(1 for x in out["labels"] if x != IGNORE) >= 2        # the answer is still trained on


def test_batching_pads_with_ignore_labels_and_masks_padding():
    a = encode_pair(CharTokenizer(), "Q:", " A", max_len=20, eos_id=1)
    b = encode_pair(CharTokenizer(), "Question:", " B", max_len=20, eos_id=1)
    batch = make_batch([a, b], pad_id=0)
    assert len({len(x) for x in batch["input_ids"]}) == 1
    assert batch["labels"][0][-1] == IGNORE and batch["attention_mask"][0][-1] == 0
    assert batch["attention_mask"][1][-1] == 1 and batch["input_ids"][0][-1] == 0


def test_site_dataset_has_natural_examples_then_canary_copies():
    canaries = generate_canaries(split_seed=1, sites=3, levels={0: 2, 1: 2, 4: 2}, case_digits=6,
                                 secret_digits=8, references=16)
    rows = {f"r{i}": dict(ROW, question=f"Q{i}") for i in range(5)}
    items = site_examples(1, list(rows), rows, canaries)
    assert [i["kind"] for i in items[:5]] == ["natural"] * 5
    copies = [i for i in items if i["kind"] == "canary"]
    assert len(copies) == 2 * (0 + 1 + 4)                              # level 0 is never inserted
    assert {i["id"] for i in copies} == {f"canary-{c['case_id']}" for c in canaries["canaries"]
                                         if c["site"] == 1 and c["repetitions"] > 0}
    assert all(i["prompt"].startswith("Question:") and i["completion"].strip().isdigit() for i in copies)
    assert items == site_examples(1, list(rows), rows, canaries)         # deterministic
    assert len(site_examples(0, list(rows), rows, None)) == 5



def test_duplicates_are_found_by_normalized_text_not_by_id():
    from pqfl.train.data import duplicates_of, text_key
    same = dict(ROW, id="x", question="  which NERVE? ")
    other = dict(ROW, id="y", opa="Femoral")
    assert text_key(same) == text_key(ROW)
    assert duplicates_of([same, other], [dict(ROW, id="z")]) == [True, False]
