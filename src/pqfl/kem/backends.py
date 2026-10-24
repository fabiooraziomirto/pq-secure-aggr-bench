"""Lazy, capability-based liboqs adapter; never install or download libraries."""
import ctypes
import ctypes.util
import importlib
import importlib.util
import os
from pathlib import Path
import warnings


class BackendUnavailable(RuntimeError):
    """A real KEM prerequisite is missing, not a failed cryptographic operation."""


_INSTRUCTIONS = (
    "On the cluster, install a reviewed liboqs and its official liboqs-python "
    "binding, configure the native loader (LD_LIBRARY_PATH or cluster modules), "
    "then run scripts/check_env.sh and pytest -q -rs tests/test_kem_real.py. "
    "No automatic installation or FakeKEM fallback is performed."
)


def load_oqs():
    """Check native prerequisites before importing a binding that may auto-install."""
    if importlib.util.find_spec("oqs") is None:
        raise BackendUnavailable("Python binding 'oqs' missing. " + _INSTRUCTIONS)
    libraries = [ctypes.util.find_library(name) for name in ("oqs", "liboqs")]
    # The binding supports OQS_INSTALL_PATH; require a loadable library there.
    root = os.environ.get("OQS_INSTALL_PATH", str(Path.home() / "_oqs"))
    candidates = ([str(Path(root) / directory / filename)
                   for directory in ("lib", "lib64")
                   for filename in ("liboqs.so", "liboqs.dylib")])
    candidates.extend(library for library in libraries if library)
    loaded = False
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate)
        except OSError:
            continue
        loaded = True
        break
    if not loaded:
        raise BackendUnavailable("Native liboqs not discoverable/loadable. " + _INSTRUCTIONS)
    try:
        module = importlib.import_module("oqs")
    except (ImportError, OSError, RuntimeError) as exc:
        raise BackendUnavailable("Cannot initialize liboqs-python. " + _INSTRUCTIONS) from exc
    if not all(hasattr(module, name) for name in ("KeyEncapsulation", "get_enabled_kem_mechanisms")):
        raise BackendUnavailable("'oqs' is not the supported liboqs-python API. " + _INSTRUCTIONS)
    return module


class LibOQSKEM:
    """ML-KEM-768, with explicitly opted-in, differently labelled legacy Kyber768."""

    def __init__(self, *, allow_legacy: bool = False):
        self._oqs = load_oqs()
        enabled = set(self._oqs.get_enabled_kem_mechanisms())
        if "ML-KEM-768" in enabled:
            self.mechanism = "ML-KEM-768"
        elif "Kyber768" in enabled and allow_legacy:
            self.mechanism = "Kyber768"
            warnings.warn("Kyber768 is legacy and is NOT standardized ML-KEM-768",
                          RuntimeWarning, stacklevel=2)
        elif "Kyber768" in enabled:
            raise BackendUnavailable("Only legacy Kyber768 is enabled; it is not ML-KEM-768. "
                                     "For explicit legacy-only validation use allow_legacy=True. "
                                     + _INSTRUCTIONS)
        else:
            raise BackendUnavailable("Neither ML-KEM-768 nor Kyber768 is enabled. " + _INSTRUCTIONS)
        with self._oqs.KeyEncapsulation(self.mechanism) as kem:
            self.secret_key_bytes = kem.details["length_secret_key"]
            expected = {"length_public_key": 1184, "length_ciphertext": 1088,
                        "length_shared_secret": 32}
            if any(kem.details[key] != size for key, size in expected.items()):
                raise RuntimeError("Backend dimensions do not match the selected ML-KEM-768/Kyber768 parameter set")
        self.library_versions = {
            "kem_mechanism": self.mechanism,
            "liboqs": self._oqs.oqs_version(),
            "liboqs-python": self._oqs.oqs_python_version(),
        }

    def keygen(self) -> tuple[bytes, bytes]:
        with self._oqs.KeyEncapsulation(self.mechanism) as kem:
            public = bytes(kem.generate_keypair())
            secret = bytes(kem.export_secret_key())
        return public, secret

    def encaps(self, public: bytes) -> tuple[bytes, bytes]:
        if len(public) != 1184:
            raise ValueError("Expected a 1184-byte public key")
        with self._oqs.KeyEncapsulation(self.mechanism) as kem:
            ciphertext, shared = kem.encap_secret(public)
        return bytes(ciphertext), bytes(shared)

    def decaps(self, secret: bytes, ciphertext: bytes) -> bytes:
        if len(secret) != self.secret_key_bytes:
            raise ValueError("Invalid secret key length")
        if len(ciphertext) != 1088:
            raise ValueError("Expected a 1088-byte ciphertext")
        with self._oqs.KeyEncapsulation(self.mechanism, secret_key=secret) as kem:
            return bytes(kem.decap_secret(ciphertext))


class OpenSSLKEM:
    """TODO: supported OpenSSL ML-KEM-768 binding; no fallback."""

    def __init__(self):
        raise NotImplementedError("ML-KEM-768 OpenSSL backend pending")
