import numpy as np
import pytest

from pqfl.attacks.membership import evaluate, loss_scores, reference_scores, roc_auc, tpr_at_fpr
from pqfl.stats import bootstrap_ci, intervals_overlap, paired_difference_ci


def test_auc_known_cases_and_ties():
    assert roc_auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert roc_auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert roc_auc([1, 1, 1], [1, 1, 1]) == 0.5               # all tied
    assert roc_auc([2, 2], [1, 3]) == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    assert abs(roc_auc(rng.normal(size=4000), rng.normal(size=4000)) - 0.5) < 0.03
    with pytest.raises(ValueError):
        roc_auc([], [1.0])


def test_tpr_at_fpr_budget_and_resolution():
    nonmembers = np.arange(1000, dtype=float)                  # 0..999
    members = np.arange(995, 1995, dtype=float)
    # 1 % of 1000 = 10 false positives allowed: threshold is the 11th largest (989).
    assert tpr_at_fpr(members, nonmembers, 0.01) == (members > 989).mean()
    assert tpr_at_fpr(members, nonmembers, 0.001) == (members > 998).mean()   # one false positive
    with pytest.raises(ValueError, match="resolution"):
        tpr_at_fpr(members, nonmembers[:100], 0.001)
    ties = np.zeros(1000)
    assert tpr_at_fpr(np.zeros(10), ties, 0.01) == 0.0          # ties never counted as flagged


def test_reference_calibration_beats_plain_loss_when_difficulty_varies():
    rng = np.random.default_rng(1)
    n = 3000
    difficulty_m, difficulty_n = rng.normal(2.0, 1.0, n), rng.normal(2.0, 1.0, n)
    member_nll = difficulty_m - 0.4 + rng.normal(0, 0.1, n)    # training lowers a member's loss by 0.4
    nonmember_nll = difficulty_n + rng.normal(0, 0.1, n)
    plain = evaluate(member_nll, nonmember_nll)
    calibrated = evaluate(member_nll, nonmember_nll, reference_member_nll=difficulty_m,
                          reference_nonmember_nll=difficulty_n)
    assert plain["loss"]["auc"] < 0.65 < calibrated["reference"]["auc"]
    assert calibrated["reference"]["tpr_at_fpr_0.01"] > plain["loss"]["tpr_at_fpr_0.01"]
    assert set(calibrated) == {"loss", "reference"} and "reference" not in plain


def test_null_check_gives_chance_level_metrics():
    rng = np.random.default_rng(2)
    same = evaluate(rng.normal(2, 1, 6000), rng.normal(2, 1, 3000))
    assert abs(same["loss"]["auc"] - 0.5) < 0.03
    assert same["loss"]["tpr_at_fpr_0.01"] < 0.04 and same["loss"]["tpr_at_fpr_0.001"] < 0.01
    assert evaluate(rng.normal(size=100), rng.normal(size=100))["loss"]["tpr_at_fpr_0.001"] is None
    assert np.array_equal(loss_scores([1.0, 2.0]), [-1.0, -2.0])
    assert np.array_equal(reference_scores([1.0], [3.0]), [2.0])


def test_bootstrap_intervals_and_overlap_rule():
    ci = bootstrap_ci([0.50, 0.52, 0.49, 0.51, 0.50], seed=1)
    assert ci[1] <= ci[0] <= ci[2] and ci == bootstrap_ci([0.50, 0.52, 0.49, 0.51, 0.50], seed=1)
    low, high = bootstrap_ci([1.0, 1.1, 0.9, 1.0, 1.05], seed=2), bootstrap_ci([3.0, 3.1, 2.9, 3.0, 3.05], seed=2)
    assert not intervals_overlap(low, high) and intervals_overlap(low, low)
    diff = paired_difference_ci([3, 3, 3, 3, 3.0], [1, 1, 1, 1, 1.0], seed=3)
    assert diff == pytest.approx((2.0, 2.0, 2.0))
    with pytest.raises(ValueError):
        bootstrap_ci([1.0])
    with pytest.raises(ValueError):
        paired_difference_ci([1, 2], [1])



def test_fpr_budget_is_robust_to_floating_point_and_only_resolution_is_tolerated():
    others = np.arange(100, dtype=float)
    members = others + 0.5
    # 0.29 * 100 = 28.999999999999996 in floating point: 29 false positives must still be allowed
    assert tpr_at_fpr(members, others, 0.29) == pytest.approx(np.mean(members > others[-30]))
    with pytest.raises(ValueError, match="resolution"):   # BelowResolution (by name: test_worker re-imports)
        tpr_at_fpr(members, others, 0.001)
    assert evaluate(members, others, fprs=(0.001,))["loss"]["tpr_at_fpr_0.001"] is None
    with pytest.raises(ValueError):
        evaluate([], others)
