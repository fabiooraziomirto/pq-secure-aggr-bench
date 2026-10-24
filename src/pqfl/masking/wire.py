"""Canonical protocol payloads; shape and bit width are agreed out of band."""
import math

import numpy as np

from pqfl.masking.core import Tensor, validate_parameters


def serialize_masked(tensor: Tensor, modulus_bits: int) -> bytes:
    validate_parameters(0, modulus_bits)
    values = np.asarray(tensor)
    if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= 2 ** modulus_bits):
        raise ValueError("Expected canonical unsigned modular integers")
    width = (modulus_bits + 7) // 8
    words = values.astype("<u8").ravel(order="C")
    return words.view(np.uint8).reshape(-1, 8)[:, :width].tobytes()


def deserialize_masked(data: bytes, shape: tuple[int, ...], modulus_bits: int) -> Tensor:
    validate_parameters(0, modulus_bits)
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Invalid tensor shape")
    width = (modulus_bits + 7) // 8
    count = math.prod(shape)
    if len(data) != count * width:
        raise ValueError("Unexpected masked payload length")
    words = np.zeros((count, 8), dtype=np.uint8)
    words[:, :width] = np.frombuffer(data, dtype=np.uint8).reshape(count, width)
    values = words.reshape(-1).view("<u8")
    if np.any(values >= 2 ** modulus_bits):
        raise ValueError("Noncanonical high bits in masked payload")
    return values.astype(np.int64).reshape(shape)
