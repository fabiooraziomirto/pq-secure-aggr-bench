"""Metrics for gradient-inversion text reconstruction, judged against a random-text null.

Sequences are lists of token ids. A reconstruction only counts as a success when it beats
what random text drawn from the corpus unigram distribution would score against the same
reference, so frequent tokens cannot fake success. The attack itself (gradient matching on
the GPU) is not implemented here.
"""
import math
from collections import Counter

import numpy as np


def bag_recall(reference, recovered) -> float:
    """Fraction of reference tokens present in the reconstruction (multiset overlap)."""
    if not reference:
        raise ValueError("Empty reference")
    overlap = sum((Counter(reference) & Counter(recovered)).values())
    return overlap / len(reference)


def rouge_n_f1(reference, recovered, n: int = 2) -> float:
    """F1 of n-gram overlap (multiset)."""
    grams = lambda seq: Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    ref, rec = grams(reference), grams(recovered)
    overlap = sum((ref & rec).values())
    if not ref or not rec or overlap == 0:
        return 0.0
    precision, recall = overlap / sum(rec.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def lcs_length(first, second) -> int:
    previous = [0] * (len(second) + 1)
    for a in first:
        current = [0]
        for j, b in enumerate(second, start=1):
            current.append(previous[j - 1] + 1 if a == b else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference, recovered) -> float:
    lcs = lcs_length(reference, recovered)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(recovered), lcs / len(reference)
    return 2 * precision * recall / (precision + recall)


def null_scores(reference, metric, corpus_tokens, *, length: int, draws: int, seed: int = 0):
    """Metric between the reference and `draws` random sequences of `length` tokens sampled
    from the empirical unigram distribution of `corpus_tokens`."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(corpus_tokens)
    return [metric(reference, rng.choice(pool, size=length, replace=True).tolist()) for _ in range(draws)]


def exceeds_null(score: float, null, quantile: float = 0.95) -> bool:
    """Strictly above the given quantile of the null scores."""
    return score > float(np.quantile(np.asarray(null, dtype=float), quantile))


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a success rate; sound for the small samples used here."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Invalid counts")
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return centre - half, centre + half


def positive_control_passes(successes: int, trials: int, minimum_lower_bound: float = 0.5) -> bool:
    """The control holds if the Wilson lower bound of the success rate exceeds the threshold."""
    return wilson_interval(successes, trials)[0] > minimum_lower_bound
