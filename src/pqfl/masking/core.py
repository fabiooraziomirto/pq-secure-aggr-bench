import json
import math

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

Tensor = np.ndarray


def validate_parameters(scale_bits: int, modulus_bits: int) -> None:
    if type(scale_bits) is not int or type(modulus_bits) is not int:
        raise ValueError("Bit widths must be integers")
    if not 0 <= scale_bits < modulus_bits <= 62:
        raise ValueError("Require 0 <= scale_bits < modulus_bits <= 62")


def quantize(update: Tensor, scale_bits: int, modulus_bits: int) -> Tensor:
    validate_parameters(scale_bits, modulus_bits)
    values = np.asarray(update, dtype=np.float64)
    scaled = np.rint(values * (2 ** scale_bits))
    limit = 2 ** (modulus_bits - 1)
    if not np.all(np.isfinite(scaled)) or np.any(scaled >= limit) or np.any(scaled < -limit):
        raise ValueError("Update exceeds signed fixed-point range")
    return np.remainder(scaled.astype(np.int64), 2 ** modulus_bits)


def dequantize(values: Tensor, scale_bits: int, modulus_bits: int) -> Tensor:
    validate_parameters(scale_bits, modulus_bits)
    modulus = 2 ** modulus_bits
    values = np.remainder(values, modulus)
    signed = np.where(values >= modulus // 2, values - modulus, values)
    return signed.astype(np.float64) / (2 ** scale_bits)


def pairwise_mask(seed: bytes, round_idx: int, left: str, right: str,
                  shape: tuple[int, ...], modulus_bits: int, *, epoch: int) -> Tensor:
    validate_parameters(0, modulus_bits)
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Require a nonnegative epoch")
    if type(round_idx) is not int or round_idx < 0 or not left < right:
        raise ValueError("Require nonnegative round and canonical distinct site IDs")
    info = json.dumps(["pqfl/scheme-a/mask/v2", epoch, round_idx, left, right],
                      ensure_ascii=True, separators=(",", ":")).encode()
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(seed)
    # A fixed nonce is safe only because each epoch/pair/round has a distinct key.
    stream = Cipher(algorithms.ChaCha20(key, bytes(16)), mode=None).encryptor()
    data = stream.update(bytes(math.prod(shape) * 8)) + stream.finalize()
    words = np.frombuffer(data, dtype="<u8") & np.uint64((1 << modulus_bits) - 1)
    return words.astype(np.int64).reshape(shape)


def recover_masks(*args, **kwargs):
    """TODO: dropout recovery with secret sharing of pairwise seeds."""
    raise NotImplementedError("Dropout recovery is outside the current protocol")
