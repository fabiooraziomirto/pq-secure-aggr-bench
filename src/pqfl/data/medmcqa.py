"""Deterministic MedMCQA split across three sites by subject (non-IID), plus held-out slices.

Works on plain (id, subject) records so it needs no dataframe library. Ids are the public
dataset ids; no question text is stored. The split seed is fixed across training seeds, so
all configurations see the same data (protocol: same data, only the listed factors change).
"""
import hashlib
import json

import numpy as np

UNKNOWN = "Unknown"


def partition_subjects(counts: dict[str, int], sites: int) -> list[list[str]]:
    """Greedy balanced partition: largest subject first into the smallest site.

    Deterministic (ties by name). Each site gets a disjoint set of subjects, so sites
    differ in topic but have similar sizes. `Unknown` is never assigned.
    """
    if sites < 2:
        raise ValueError("Require at least two sites")
    buckets = [[0, []] for _ in range(sites)]
    for subject, count in sorted(((s, c) for s, c in counts.items() if s != UNKNOWN),
                                 key=lambda item: (-item[1], item[0])):
        target = min(buckets, key=lambda bucket: (bucket[0], len(bucket[1])))
        target[0] += count
        target[1].append(subject)
    return [sorted(subjects) for _, subjects in buckets]


def build_manifest(records: list[tuple[str, str]], validation_ids: list[str], *, sites: int,
                   pool: int, holdout: int, reference: int, split_seed: int) -> dict:
    """Per site: `pool` members, `holdout` non-members (same subjects), `reference` examples
    for reference models. All slices are disjoint; validation ids must not appear in train."""
    ids = [record[0] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate example ids in the training records")
    if set(ids) & set(validation_ids):
        raise ValueError("Validation ids overlap the training records")
    counts: dict[str, int] = {}
    for _, subject in records:
        counts[subject] = counts.get(subject, 0) + 1
    groups = partition_subjects(counts, sites)
    manifest = dict(split_seed=split_seed, parameters=dict(sites=sites, pool=pool, holdout=holdout,
                                                           reference=reference),
                    subjects_by_site=groups, train_examples_by_site=[], pool=[], holdout=[],
                    reference=[], validation_ids=sorted(validation_ids))
    for index, subjects in enumerate(groups):
        eligible = sorted(rid for rid, subject in records if subject in subjects)
        need = pool + holdout + reference
        if len(eligible) < need:
            raise ValueError(f"Site {index} has {len(eligible)} examples, needs {need}")
        order = np.random.default_rng([split_seed, index]).permutation(len(eligible))
        shuffled = [eligible[i] for i in order]
        manifest["train_examples_by_site"].append(len(eligible))
        manifest["pool"].append(sorted(shuffled[:pool]))
        manifest["holdout"].append(sorted(shuffled[pool:pool + holdout]))
        manifest["reference"].append(sorted(shuffled[pool + holdout:need]))
    return manifest


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_manifest(manifest: dict, path) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
