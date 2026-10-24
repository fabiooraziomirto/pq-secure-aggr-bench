"""INSECURE test double, deliberately outside the installable package."""
import random
import pytest


class FakeKEM:
    """INSECURE: public key equals secret key; shared secret is sent in clear.

    Deterministic random.Random is for reproducible unit tests ONLY.
    Never import this module in application code.
    """

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.key_material = []
        self.generated_material = []

    def keygen(self):
        key = self.rng.randbytes(32)
        self.key_material.append(key)
        self.generated_material.append(key)
        return key, key

    def encaps(self, public):
        shared = self.rng.randbytes(32)
        self.generated_material.append(shared)
        return shared, shared

    def decaps(self, secret, ciphertext):
        return ciphertext


@pytest.fixture
def fake_kem_factory():
    return FakeKEM
