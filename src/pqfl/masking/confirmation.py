"""Domain-separated confirmation using library HKDF, SHA-256 and HMAC."""
import hashlib
import json

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def confirmation_key(seed: bytes, epoch: int, left: str, right: str,
                     public: bytes, ciphertext: bytes) -> bytes:
    transcript_hash = hashlib.sha256(public + ciphertext).hexdigest()
    info = json.dumps(["pqfl/scheme-a/confirm/v1", epoch, left, right, transcript_hash],
                      ensure_ascii=True, separators=(",", ":")).encode()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(seed)


def confirmation_tag(key: bytes, role: str) -> bytes:
    if role not in {"left", "right"}:
        raise ValueError("Unknown key-confirmation role")
    signer = hmac.HMAC(key, hashes.SHA256())
    signer.update(role.encode("ascii"))
    return signer.finalize()
