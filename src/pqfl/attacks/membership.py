"""Membership inference scores and metrics (loss-based, with optional reference calibration).

Inputs are per-example mean negative log-likelihoods over the tokens trained on, computed by
the target model and, for calibrated attacks, by a reference model that did not train on the
member or non-member examples. A higher score means "more likely a member".
"""
import numpy as np


class BelowResolution(ValueError):
    """The requested FPR allows no false positive with this many non-members."""


def loss_scores(target_nll):
    """Plain loss attack: members tend to have lower loss."""
    return -np.asarray(target_nll, dtype=float)


def reference_scores(target_nll, reference_nll):
    """Reference-calibrated attack: how much lower the target's loss is than the reference's.
    Removes per-example difficulty, which the reference model shares with the target."""
    return np.asarray(reference_nll, dtype=float) - np.asarray(target_nll, dtype=float)


def roc_auc(member_scores, nonmember_scores) -> float:
    """Area under the ROC curve via the rank statistic; ties count one half."""
    members = np.asarray(member_scores, dtype=float)
    others = np.asarray(nonmember_scores, dtype=float)
    if len(members) == 0 or len(others) == 0:
        raise ValueError("Need members and non-members")
    scores = np.concatenate([members, others])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):  # average ranks over ties
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    u = ranks[:len(members)].sum() - len(members) * (len(members) + 1) / 2
    return float(u / (len(members) * len(others)))


def tpr_at_fpr(member_scores, nonmember_scores, fpr: float) -> float:
    """True-positive rate when at most floor(fpr * n) non-members may be flagged.

    The threshold is the smallest score keeping the false positives within that budget;
    ties at the threshold are excluded (conservative). Resolution is 1/n_nonmembers:
    with 1000 non-members, FPR 0.1 % allows a single false positive.
    """
    members = np.asarray(member_scores, dtype=float)
    others = np.sort(np.asarray(nonmember_scores, dtype=float))
    if not 0 < fpr < 1 or len(members) == 0 or len(others) == 0:
        raise ValueError("Invalid inputs")
    allowed = int(np.floor(fpr * len(others) + 1e-9))      # tolerance: 0.29 * 100 is 28.999999999999996
    if allowed == 0:
        raise BelowResolution(f"FPR {fpr} is below the resolution 1/{len(others)} of this sample")
    threshold = others[-allowed - 1]  # flag strictly above the (allowed+1)-th largest
    return float((members > threshold).mean())


def evaluate(member_nll, nonmember_nll, *, reference_member_nll=None, reference_nonmember_nll=None,
             fprs=(0.01, 0.001)) -> dict:
    """AUC and TPR at each FPR for the plain attack and, when given, the calibrated one."""
    result = {}
    attacks = {"loss": (loss_scores(member_nll), loss_scores(nonmember_nll))}
    if reference_member_nll is not None and reference_nonmember_nll is not None:
        attacks["reference"] = (reference_scores(member_nll, reference_member_nll),
                                reference_scores(nonmember_nll, reference_nonmember_nll))
    for name, (members, others) in attacks.items():
        result[name] = {"auc": roc_auc(members, others)}
        for fpr in fprs:
            try:
                result[name][f"tpr_at_fpr_{fpr:g}"] = tpr_at_fpr(members, others, fpr)
            except BelowResolution:
                result[name][f"tpr_at_fpr_{fpr:g}"] = None
    return result
