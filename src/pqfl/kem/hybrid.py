"""Experimental combiner, pending expert review; NOT an authenticated channel."""
import struct

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from pqfl.kem.base import KEM


_PREFIX = b"pqfl/hybrid-kem/v1/"


def _pack(tag: bytes, *fields: bytes) -> bytes:
    return _PREFIX + tag + b"".join(struct.pack(">I", len(item)) + item for item in fields)


def _unpack(data: bytes, tag: bytes, count: int) -> tuple[bytes, ...]:
    prefix = _PREFIX + tag
    if not isinstance(data, bytes) or not data.startswith(prefix):
        raise ValueError("Invalid hybrid encoding or version")
    offset, fields = len(prefix), []
    for _ in range(count):
        if offset + 4 > len(data):
            raise ValueError("Truncated hybrid encoding")
        size = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if offset + size > len(data):
            raise ValueError("Truncated hybrid field")
        fields.append(data[offset:offset + size])
        offset += size
    if offset != len(data):
        raise ValueError("Trailing hybrid data")
    return tuple(fields)


def _public(key) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


class HybridKEM:
    """Recipient X25519 key + ephemeral sender key + PQ KEM, then HKDF.

    Explicit KEM injection supports tests only when the injected KEM is fake.
    Default construction always requests a real standardized ML-KEM backend.
    """

    def __init__(self, pq_kem: KEM | None = None):
        if pq_kem is None:
            from pqfl.kem.backends import LibOQSKEM
            pq_kem = LibOQSKEM()
        self.pq_kem = pq_kem
        self._algorithm = getattr(pq_kem, "mechanism", "explicit-injected-kem").encode("ascii")
        self.mechanism = "X25519+" + self._algorithm.decode("ascii")
        self.library_versions = dict(getattr(pq_kem, "library_versions", {}),
                                     hybrid_combiner="pqfl/hybrid-kem/v1")

    def keygen(self) -> tuple[bytes, bytes]:
        private = X25519PrivateKey.generate()
        pq_public, pq_secret = self.pq_kem.keygen()
        public = _pack(b"pk", self._algorithm, _public(private.public_key()), pq_public)
        raw = private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                    serialization.NoEncryption())
        secret = _pack(b"sk", raw, pq_secret, public)
        return public, secret

    def encaps(self, public: bytes) -> tuple[bytes, bytes]:
        algorithm, x_public, pq_public = _unpack(public, b"pk", 3)
        if algorithm != self._algorithm:
            raise ValueError("Hybrid mechanism mismatch")
        ephemeral = X25519PrivateKey.generate()
        x_shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(x_public))
        pq_ciphertext, pq_shared = self.pq_kem.encaps(pq_public)
        ciphertext = _pack(b"ct", _public(ephemeral.public_key()), pq_ciphertext)
        return ciphertext, self._combine(x_shared, pq_shared, public, ciphertext)

    def decaps(self, secret: bytes, ciphertext: bytes) -> bytes:
        x_secret, pq_secret, public = _unpack(secret, b"sk", 3)
        algorithm, x_public, _ = _unpack(public, b"pk", 3)
        ephemeral_public, pq_ciphertext = _unpack(ciphertext, b"ct", 2)
        if algorithm != self._algorithm:
            raise ValueError("Hybrid mechanism mismatch")
        private = X25519PrivateKey.from_private_bytes(x_secret)
        if _public(private.public_key()) != x_public:
            raise ValueError("Hybrid public/secret key mismatch")
        x_shared = private.exchange(X25519PublicKey.from_public_bytes(ephemeral_public))
        pq_shared = self.pq_kem.decaps(pq_secret, pq_ciphertext)
        return self._combine(x_shared, pq_shared, public, ciphertext)

    @staticmethod
    def _combine(x_shared: bytes, pq_shared: bytes, public: bytes, ciphertext: bytes) -> bytes:
        if len(x_shared) != 32 or len(pq_shared) != 32:
            raise ValueError("Both component shared secrets must have 32 bytes")
        transcript = _pack(b"combiner", public, ciphertext)
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=transcript).derive(x_shared + pq_shared)
