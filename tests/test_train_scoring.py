"""Needs torch: run in the training environment with `--noconftest`."""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pqfl.data.canaries import TEMPLATE  # noqa: E402
from pqfl.train.scoring import canary_report, canary_scores, greedy_continuations, sequence_logprobs  # noqa: E402

VOCAB = 128


class Tok:
    def encode(self, text):
        return [ord(c) for c in text]


class Bigram(torch.nn.Module):
    """logits at a position depend only on the current token (a bigram table)."""
    def __init__(self, successors):
        super().__init__()
        table = torch.zeros(VOCAB, VOCAB)
        for current, nxt in successors.items():
            table[ord(current), ord(nxt)] = 12.0
        self.table = torch.nn.Parameter(table, requires_grad=False)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.table[input_ids])


def chain(prompt_last, secret):
    """successor map: prompt_last -> ' ' -> secret[0] -> secret[1] ..."""
    text = prompt_last + " " + secret
    return {a: b for a, b in zip(text, text[1:])}


def manifest(case_id, secret, references, repetitions=4):
    return dict(canaries=[dict(site=0, repetitions=repetitions, index=0, case_id=case_id, secret=secret,
                               references=references)])


def test_sequence_logprobs_match_the_closed_form_and_are_batch_independent():
    model = Bigram({":": " "})
    prefix = [ord("a"), ord(":")]
    good, bad = [ord(" ")], [ord("x")]
    lp = sequence_logprobs(model, prefix, [good, bad], batch_size=1, pad_id=0)
    high = -math.log(1 + (VOCAB - 1) * math.exp(-12.0))
    assert lp[0] == pytest.approx(high, abs=1e-4) and lp[1] < lp[0] - 10
    assert sequence_logprobs(model, prefix, [good, bad], batch_size=2, pad_id=0) == pytest.approx(lp, abs=1e-5)
    with pytest.raises(ValueError):
        sequence_logprobs(model, prefix, [good, [1, 2]], batch_size=2, pad_id=0)


def test_memorized_canary_has_maximal_exposure_and_is_extracted():
    secret, refs = "12345678", ["87654321", "11111111", "99999999", "13572468"]
    model = Bigram(chain(TEMPLATE[-1], secret))                    # the model "knows" the code after ':'
    scores = canary_scores(model, Tok(), manifest("123456", secret, refs), batch_size=3)
    assert scores[0]["rank"] == 1 and scores[0]["exposure"] == pytest.approx(math.log2(len(refs) + 1))
    assert scores[0]["extracted"] is True


def test_unknown_canary_is_neither_exposed_nor_extracted():
    secret, refs = "12345678", ["87654321", "11111111", "99999999", "13572468"]
    model = Bigram(chain(TEMPLATE[-1], refs[0]))                   # it knows another code
    s = canary_scores(model, Tok(), manifest("123456", secret, refs), batch_size=4)[0]
    assert s["rank"] == len(refs) + 1 and s["exposure"] == pytest.approx(0.0) and s["extracted"] is False


def test_greedy_continuations_follow_the_bigram_chain():
    model = Bigram({"a": "b", "b": "c", "c": "d"})
    assert greedy_continuations(model, [[ord("a")], [ord("b")]], 2) == [[ord("b"), ord("c")], [ord("c"), ord("d")]]
    with pytest.raises(ValueError):
        greedy_continuations(model, [[1], [1, 2]], 1)


def test_report_groups_by_level_and_takes_the_null_percentile():
    scores = [dict(repetitions=0, exposure=e, extracted=False) for e in (0.0, 1.0, 2.0, 3.0)] + \
             [dict(repetitions=16, exposure=8.0, extracted=True)] * 2
    report = canary_report(scores)
    assert report["levels"]["16"] == dict(n=2, mean_exposure=8.0, extraction_rate=1.0)
    assert report["levels"]["0"]["mean_exposure"] == pytest.approx(1.5)
    assert report["null_p95_exposure"] == pytest.approx(2.85)
