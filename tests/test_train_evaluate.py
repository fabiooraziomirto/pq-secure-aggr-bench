"""Needs torch: run in the training environment with `--noconftest`."""
import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pqfl.train.evaluate import (accuracy_report, append_jsonl, example_nll, option_predictions, perplexity,  # noqa: E402
                                 single_token_ids)

VOCAB = 128


class Tok:
    """One token per character; ids are the character codes."""
    def encode(self, text):
        return [ord(c) for c in text]


class NextIsPlusOne(torch.nn.Module):
    """Predicts that the token after id t is t + 1 (a strong, known signal)."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros(*input_ids.shape, VOCAB) + self.anchor
        logits.scatter_(2, ((input_ids + 1) % VOCAB).unsqueeze(-1), 8.0)
        return SimpleNamespace(logits=logits)


def rows(n):
    return [dict(question="q" * (3 + i % 7), opa="x", opb="y", opc="z", opd="w", cop=i % 4, exp=None,
                 subject_name="S1" if i % 2 else "S2") for i in range(n)]


# prompt_text ends with ":" ; the model predicts ord(":") + 1, so option B's token is set to that id
LETTER_IDS = [ord("A"), ord(":") + 1, ord("C"), ord("D")]


def test_single_token_ids_rejects_multi_token_letters():
    assert single_token_ids(Tok(), ["A", "B"]) == [65, 66]
    with pytest.raises(ValueError, match="single token"):
        single_token_ids(Tok(), [" A"])


def test_predictions_read_the_last_real_position_even_with_padding():
    data = rows(20)                                           # prompts of different lengths: right padding
    predictions = option_predictions(NextIsPlusOne(), Tok(), data, LETTER_IDS, batch_size=6, max_len=512, pad_id=0)
    assert predictions == [1] * 20
    long = [dict(data[0], question="q" * 600)]
    assert option_predictions(NextIsPlusOne(), Tok(), long, LETTER_IDS, batch_size=1, max_len=64, pad_id=0) == [1]


def test_accuracy_report_overall_and_per_subject():
    data = rows(8)
    report = accuracy_report(data, [1] * 8)
    assert report["n"] == 8 and report["accuracy"] == pytest.approx(2 / 8)         # cop == 1 in 2 of 8 rows
    assert set(report["by_subject"]) == {"S1", "S2"}
    assert report["macro_accuracy"] == pytest.approx(sum(v["accuracy"] for v in report["by_subject"].values()) / 2)


def test_example_nll_ignores_padding_and_matches_the_closed_form():
    items = [dict(prompt="ab:", completion=" c"), dict(prompt="a longer prompt:", completion=" cd")]
    batched = example_nll(NextIsPlusOne(), Tok(), items, batch_size=2, max_len=64, eos_id=1, pad_id=0)
    single = [example_nll(NextIsPlusOne(), Tok(), [i], batch_size=1, max_len=64, eos_id=1, pad_id=0)[0] for i in items]
    for (a, ca), (b, cb) in zip(batched, single):
        assert ca == cb and a == pytest.approx(b, rel=1e-5)
    assert [c for _, c in batched] == [3, 4]                    # completion tokens plus EOS
    # every scored token has probability 1/(e^8 + 127) of the wrong ids: NLL is large and equals this closed form
    wrong = -math.log(1 / (math.exp(8.0) + VOCAB - 1))
    assert batched[0][0] > 0
    summary = perplexity(batched)
    assert summary["n"] == 2 and summary["tokens"] == 7 and summary["perplexity"] == pytest.approx(math.exp(summary["mean_nll"]))
    assert summary["mean_nll"] <= wrong + 1e-6


def test_jsonl_append_writes_one_record_per_line(tmp_path):
    path = tmp_path / "e.jsonl"
    append_jsonl(str(path), {"round": 0, "x": 1.5})
    append_jsonl(str(path), {"round": 1})
    assert [json.loads(l)["round"] for l in path.read_text().splitlines()] == [0, 1]
