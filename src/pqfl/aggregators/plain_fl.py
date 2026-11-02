from pqfl.aggregators.insecure_stub import InsecureSumStub
from pqfl.masking.core import dequantize, quantize, validate_parameters
from pqfl.masking.wire import deserialize_masked, serialize_masked


class PlainFL(InsecureSumStub):
    """N: unprotected FL reference for cost. Star topology, float32 payloads (4 bytes
    per element, what an fp32 adapter delta costs). No cryptography: never a result
    about privacy. Star ledger identical to the stub, so costs are directly comparable.
    """

    security_status = 'UNPROTECTED_BASELINE'
    dtype = '<f4'


class PlainFixedPoint(PlainFL):
    """N-FP: unprotected star with the SAME encoding as A and C (signed fixed point, scale_bits fractional
    bits, modulus_bits-bit words: 5 bytes per element at 40 bits), no masks, no sharing, no cryptography.
    The representation-matched baseline: its byte and time costs differ from A's only by the protection."""

    def __init__(self, *, scale_bits: int = 24, modulus_bits: int = 40, writer=None):
        validate_parameters(scale_bits, modulus_bits)
        super().__init__(writer=writer)
        self.scale_bits, self.modulus_bits = scale_bits, modulus_bits

    def _pack(self, values):
        return serialize_masked(quantize(values, self.scale_bits, self.modulus_bits), self.modulus_bits)

    def _unpack(self, message, shape):
        return dequantize(deserialize_masked(message, shape, self.modulus_bits), self.scale_bits, self.modulus_bits)
