import numpy as np
import pytest

from pqfl.attacks.inversion import (bag_recall, exceeds_null, lcs_length, null_scores,
                                    positive_control_passes, rouge_l_f1, rouge_n_f1, wilson_interval)

REF = [5, 9, 9, 12, 7, 3, 8]


def test_overlap_metrics_on_known_cases():
    assert bag_recall(REF, REF) == 1.0 and bag_recall(REF, [99, 98]) == 0.0
    assert bag_recall(REF, [9]) == pytest.approx(1 / 7)             # multiset: one 9 of two
    assert bag_recall(REF, [9, 9]) == pytest.approx(2 / 7)
    assert rouge_n_f1(REF, REF, 2) == 1.0 and rouge_n_f1(REF, [3, 8, 5], 2) == pytest.approx(2 * 0.5 * (1 / 6) / (0.5 + 1 / 6))
    assert lcs_length([1, 2, 3, 4], [2, 4]) == 2 and lcs_length([1], [2]) == 0
    assert rouge_l_f1(REF, REF) == 1.0 and rouge_l_f1(REF, [99]) == 0.0
    assert rouge_n_f1([1], [1], 2) == 0.0                            # too short for bigrams
    with pytest.raises(ValueError):
        bag_recall([], [1])


def test_random_text_does_not_beat_its_own_null_but_a_reconstruction_does():
    rng = np.random.default_rng(0)
    corpus = rng.integers(0, 50, 5000).tolist()                     # small vocabulary: easy overlap
    reference = rng.integers(0, 50, 40).tolist()
    null = null_scores(reference, bag_recall, corpus, length=40, draws=300, seed=1)
    assert 0.3 < np.mean(null) < 0.9                                 # chance is high with common tokens
    random_text = rng.integers(0, 50, 40).tolist()
    assert not exceeds_null(bag_recall(reference, random_text), null, 0.999)
    near_perfect = reference[:-2] + [3, 4]
    assert exceeds_null(bag_recall(reference, near_perfect), null)
    assert null == null_scores(reference, bag_recall, corpus, length=40, draws=300, seed=1)


def test_wilson_interval_and_positive_control_rule():
    low, high = wilson_interval(15, 20)
    assert low == pytest.approx(0.531, abs=0.005) and high == pytest.approx(0.888, abs=0.005)
    assert positive_control_passes(15, 20) and not positive_control_passes(14, 20)
    assert wilson_interval(0, 20)[0] == pytest.approx(0.0, abs=1e-9)
    assert not positive_control_passes(10, 20)
    for bad in ((3, 0), (5, 4), (-1, 4)):
        with pytest.raises(ValueError):
            wilson_interval(*bad)
