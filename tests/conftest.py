"""INSECURE test double, deliberately outside the installable package."""
import random
import pytest
from pqfl.auth import AuthProof


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


def pytest_addoption(parser):
    parser.addoption("--fail-on-oqs-skip", action="store_true",
                     help="Fail the session if any requires_oqs test is skipped")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.get_closest_marker("requires_oqs"):
        item.config._pqfl_real_skipped = True


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--fail-on-oqs-skip"):
        selected = any(item.get_closest_marker("requires_oqs") for item in session.items)
        if not selected or getattr(session.config, "_pqfl_real_skipped", False):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


class InsecureTestAuth:
    """INSECURE authentication bypass, confined to non-authentication unit tests."""

    def issue(self, site_id, epoch, public_key):
        return AuthProof()

    def verify(self, verifier, site_id, epoch, public_key, proof):
        pass
