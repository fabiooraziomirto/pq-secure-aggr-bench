"""Additive secret sharing over Z_(2^modulus_bits) with sealed pairwise delivery."""
import json
import os

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from pqfl.masking.core import Tensor, validate_parameters

TAG_BYTES = 16


def split_shares(quantized: Tensor, count: int, modulus_bits: int) -> list[Tensor]:
    """Split into count shares: count-1 uniform from the OS CSPRNG, the last closes the sum."""
    validate_parameters(0, modulus_bits)
    if type(count) is not int or count < 2:
        raise ValueError("Require at least two shares")
    modulus = 2 ** modulus_bits
    size = int(np.prod(quantized.shape))
    shares = []
    for _ in range(count - 1):
        words = np.frombuffer(os.urandom(size * 8), dtype="<u8") & np.uint64(modulus - 1)
        shares.append(words.astype(np.int64).reshape(quantized.shape))
    last = np.asarray(quantized, dtype=np.int64)
    for share in shares:
        last = (last - share) % modulus
    return [*shares, last]


def _share_key(seed: bytes, epoch: int, round_idx: int, sender: str, receiver: str) -> bytes:
    info = json.dumps(["pqfl/scheme-c/share/v1", epoch, round_idx, sender, receiver],
                      ensure_ascii=True, separators=(",", ":")).encode()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(seed)


def seal_share(seed: bytes, epoch: int, round_idx: int, sender: str, receiver: str,
               payload: bytes) -> bytes:
    # A fixed nonce is safe only because each epoch/round/direction has its own key.
    key = _share_key(seed, epoch, round_idx, sender, receiver)
    return ChaCha20Poly1305(key).encrypt(bytes(12), payload, None)


def open_share(seed: bytes, epoch: int, round_idx: int, sender: str, receiver: str,
               sealed: bytes) -> bytes:
    key = _share_key(seed, epoch, round_idx, sender, receiver)
    return ChaCha20Poly1305(key).decrypt(bytes(12), sealed, None)
