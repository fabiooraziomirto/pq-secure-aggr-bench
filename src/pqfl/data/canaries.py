"""Synthetic canaries for the extraction experiment (Secret Sharer style, adapted).

A canary is an invented question with a random case id (the prefix) and a random digit
code (the secret). Canaries are inserted into a site's training data `repetitions` times;
level 0 canaries are never inserted and calibrate the null. For each canary the model
scores the true code against `references` random codes sharing the same prefix; the rank
gives the exposure in bits, and greedy decoding from the prefix gives exact extraction.
No clinical content: everything is random digits behind a fixed question template.
"""
import math

import numpy as np

TEMPLATE = "Question: What is the confidential reference code of synthetic case {case_id}?\nAnswer:"


def _digits(rng, length: int) -> str:
    return "".join(str(d) for d in rng.integers(0, 10, size=length))


def generate_canaries(*, split_seed: int, sites: int, levels: dict[int, int], case_digits: int,
                      secret_digits: int, references: int) -> dict:
    """levels maps repetitions -> canaries per site (0 repetitions = never inserted)."""
    if sites < 1 or references < 1 or case_digits < 6 or secret_digits < 6:
        raise ValueError("Invalid canary parameters")
    if 10 ** secret_digits < 100 * references:
        raise ValueError("Secret space too small for the reference set")
    canaries, used_cases = [], set()
    for site in range(sites):
        for repetitions in sorted(levels):
            for index in range(levels[repetitions]):
                rng = np.random.default_rng([split_seed, 7919, site, repetitions, index])
                case_id = _digits(rng, case_digits)
                while case_id in used_cases:
                    case_id = _digits(rng, case_digits)
                used_cases.add(case_id)
                secret = _digits(rng, secret_digits)
                alternatives = set()
                while len(alternatives) < references:
                    candidate = _digits(rng, secret_digits)
                    if candidate != secret:
                        alternatives.add(candidate)
                canaries.append(dict(site=site, repetitions=repetitions, index=index,
                                     case_id=case_id, secret=secret,
                                     references=sorted(alternatives)))
    return dict(split_seed=split_seed, template=TEMPLATE, case_digits=case_digits,
                secret_digits=secret_digits, references=references,
                levels={str(k): v for k, v in sorted(levels.items())}, canaries=canaries)


def prompt(canary: dict) -> str:
    return TEMPLATE.format(case_id=canary["case_id"])


def completion(secret: str) -> str:
    return f" {secret}"


def training_examples(manifest: dict, site: int) -> list[tuple[str, str]]:
    """(prompt, completion) pairs to add to a site's pool: `repetitions` copies each."""
    examples = []
    for canary in manifest["canaries"]:
        if canary["site"] == site:
            examples += [(prompt(canary), completion(canary["secret"]))] * canary["repetitions"]
    return examples


def exposure(true_logprob: float, reference_logprobs: list[float]) -> tuple[int, float]:
    """(rank, exposure in bits). Rank 1 = the true code beats every reference; ties count
    against the canary (conservative). Maximum exposure is log2(M + 1)."""
    rank = 1 + sum(1 for value in reference_logprobs if value >= true_logprob)
    return rank, math.log2(len(reference_logprobs) + 1) - math.log2(rank)


def extracted(generated: str, secret: str) -> bool:
    """Exact extraction: the greedy continuation starts with the secret (after spaces)."""
    return generated.lstrip().startswith(secret)
