import math

import pytest

from pqfl.data.canaries import (completion, exposure, extracted, generate_canaries, prompt,
                                training_examples)

KW = dict(split_seed=2026, sites=3, levels={0: 4, 1: 4, 4: 4, 16: 4}, case_digits=6,
          secret_digits=8, references=32)


def test_generation_is_reproducible_unique_and_disjoint_from_references():
    manifest = generate_canaries(**KW)
    assert manifest == generate_canaries(**KW)
    canaries = manifest["canaries"]
    assert len(canaries) == 3 * 16
    assert len({c["case_id"] for c in canaries}) == len(canaries)  # prefixes identify canaries
    for c in canaries:
        assert len(c["secret"]) == 8 and c["secret"] not in c["references"]
        assert len(set(c["references"])) == 32 and all(len(r) == 8 for r in c["references"])
    assert manifest != generate_canaries(**dict(KW, split_seed=1))


def test_insertion_counts_follow_the_repetition_levels_per_site():
    manifest = generate_canaries(**KW)
    for site in range(3):
        examples = training_examples(manifest, site)
        assert len(examples) == 4 * (0 + 1 + 4 + 16)   # level 0 is never inserted
        never = [c for c in manifest["canaries"] if c["site"] == site and c["repetitions"] == 0]
        assert never and all((prompt(c), completion(c["secret"])) not in examples for c in never)
    one = next(c for c in manifest["canaries"] if c["repetitions"] == 4)
    assert training_examples(manifest, one["site"]).count((prompt(one), completion(one["secret"]))) == 4
    assert prompt(one).startswith("Question:") and one["case_id"] in prompt(one)


def test_exposure_extremes_and_conservative_ties():
    refs = [-10.0] * 255
    rank, bits = exposure(-1.0, refs)
    assert rank == 1 and bits == pytest.approx(math.log2(256))
    rank, bits = exposure(-20.0, refs)
    assert rank == 256 and bits == pytest.approx(0.0)
    assert exposure(-10.0, refs)[0] == 256     # a tie ranks behind every equal reference
    assert exposure(-5.0, [-1.0, -2.0, -9.0])[0] == 3


def test_extraction_is_exact_prefix_match_after_spaces():
    assert extracted(" 12345678 and more", "12345678") and extracted("12345678", "12345678")
    assert not extracted(" 12345679", "12345678") and not extracted(" 1234", "12345678")


def test_parameter_validation():
    for bad in (dict(references=0), dict(secret_digits=5), dict(case_digits=4),
                dict(secret_digits=6, references=10 ** 5)):
        with pytest.raises(ValueError):
            generate_canaries(**dict(KW, **bad))
