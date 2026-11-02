"""Classical X25519 KEM (DHKEM-style): NOT quantum-safe, by design.

The quantum-vulnerable baseline of the harvest-now-decrypt-later experiments: a recorded transcript
yields every shared secret once X25519 is broken. Never use it where post-quantum protection is claimed;
runs that select it must be labelled as such.
"""
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from pqfl.kem.hybrid import _pack, _public, _unpack


class ClassicalX25519KEM:
    """Recipient X25519 key + ephemeral sender key, then HKDF over the shared secret and the transcript."""

    mechanism = "X25519 (classical, not quantum-safe)"
    quantum_safe = False
    library_versions = {"classical_kem": "pqfl/x25519-kem/v1"}

    def keygen(self) -> tuple[bytes, bytes]:
        private = X25519PrivateKey.generate()
        public = _pack(b"x25519-pk", _public(private.public_key()))
        raw = private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                    serialization.NoEncryption())
        return public, _pack(b"x25519-sk", raw, public)

    def encaps(self, public: bytes) -> tuple[bytes, bytes]:
        (x_public,) = _unpack(public, b"x25519-pk", 1)
        ephemeral = X25519PrivateKey.generate()
        shared = ephemeral.exchange(_peer(x_public))
        ciphertext = _pack(b"x25519-ct", _public(ephemeral.public_key()))
        return ciphertext, _derive(shared, public, ciphertext)

    def decaps(self, secret: bytes, ciphertext: bytes) -> bytes:
        raw, public = _unpack(secret, b"x25519-sk", 2)
        (x_public,) = _unpack(public, b"x25519-pk", 1)
        (ephemeral_public,) = _unpack(ciphertext, b"x25519-ct", 1)
        if len(raw) != 32:
            raise ValueError("Invalid X25519 secret key")
        private = X25519PrivateKey.from_private_bytes(raw)
        if _public(private.public_key()) != x_public:
            raise ValueError("X25519 public/secret key mismatch")
        return _derive(private.exchange(_peer(ephemeral_public)), public, ciphertext)


def _peer(raw: bytes) -> X25519PublicKey:
    if len(raw) != 32:
        raise ValueError("Invalid X25519 public key")
    return X25519PublicKey.from_public_bytes(raw)


def _derive(shared: bytes, public: bytes, ciphertext: bytes) -> bytes:
    transcript = _pack(b"x25519-kdf", public, ciphertext)
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=transcript).derive(shared)
