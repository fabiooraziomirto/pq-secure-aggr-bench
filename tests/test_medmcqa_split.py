import pytest

from pqfl.data.medmcqa import UNKNOWN, build_manifest, partition_subjects

COUNTS = {"Medicine": 90, "Surgery": 80, "Pathology": 70, "Anatomy": 60, "Dental": 50, "Skin": 10,
          UNKNOWN: 30}


def records(counts=COUNTS):
    return [(f"{subject}-{i}", subject) for subject, count in counts.items() for i in range(count)]


def test_partition_is_balanced_disjoint_deterministic_and_excludes_unknown():
    groups = partition_subjects(COUNTS, 3)
    flat = [s for group in groups for s in group]
    assert sorted(flat) == sorted(s for s in COUNTS if s != UNKNOWN) and len(flat) == len(set(flat))
    sizes = [sum(COUNTS[s] for s in group) for group in groups]
    assert max(sizes) - min(sizes) <= 40
    assert groups == partition_subjects(dict(reversed(list(COUNTS.items()))), 3)


def test_manifest_slices_are_disjoint_in_domain_and_reproducible():
    recs = [r for r in records() if r[1] != UNKNOWN]
    kwargs = dict(sites=3, pool=20, holdout=10, reference=5, split_seed=7)
    manifest = build_manifest(recs, ["val-1", "val-2"], **kwargs)
    assert manifest == build_manifest(recs, ["val-1", "val-2"], **kwargs)
    subject = dict(recs)
    seen = set()
    for site, subjects in enumerate(manifest["subjects_by_site"]):
        for part, size in (("pool", 20), ("holdout", 10), ("reference", 5)):
            ids = manifest[part][site]
            assert len(ids) == size and not seen & set(ids)
            assert all(subject[i] in subjects for i in ids)  # same topics for members and non-members
            seen |= set(ids)
    other = build_manifest(recs, ["val-1"], **dict(kwargs, split_seed=8))
    assert other["pool"] != manifest["pool"]


def test_manifest_rejects_overlap_duplicates_and_small_sites():
    recs = [r for r in records() if r[1] != UNKNOWN]
    with pytest.raises(ValueError, match="overlap"):
        build_manifest(recs, [recs[0][0]], sites=3, pool=5, holdout=5, reference=5, split_seed=1)
    with pytest.raises(ValueError, match="Duplicate"):
        build_manifest(recs + [recs[0]], ["v"], sites=3, pool=5, holdout=5, reference=5, split_seed=1)
    with pytest.raises(ValueError, match="needs"):
        build_manifest(recs, ["v"], sites=3, pool=500, holdout=5, reference=5, split_seed=1)
