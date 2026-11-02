"""Lazy OpenFHE threshold-CKKS adapter; never installs or downloads libraries.

Security caveat: partial decryptions are NOT noise-flooded. OpenFHE's
NOISE_FLOODING_DECRYPT was tried and showed no added error in multiparty
decryption (error fell to machine epsilon), so no flooding protection is claimed.
"""
import importlib
import importlib.util
import struct

from pqfl.kem.backends import BackendUnavailable

_INSTRUCTIONS = ("Install the official OpenFHE Python wheel (pip install openfhe==1.6.1.0.22.4) "
                 "in a separate environment. No automatic installation is performed.")


def load_openfhe():
    if importlib.util.find_spec("openfhe") is None:
        raise BackendUnavailable("Python package 'openfhe' missing. " + _INSTRUCTIONS)
    return importlib.import_module("openfhe")


def pack(items: list[bytes]) -> bytes:
    """Length-prefixed list of serialized objects (counted as payload bytes)."""
    return struct.pack("<I", len(items)) + b"".join(struct.pack("<I", len(i)) + i for i in items)


def unpack(data: bytes) -> list[bytes]:
    (count,) = struct.unpack_from("<I", data, 0)
    offset, items = 4, []
    for _ in range(count):
        (size,) = struct.unpack_from("<I", data, offset)
        offset += 4
        items.append(data[offset:offset + size])
        offset += size
    if offset != len(data):
        raise ValueError("Malformed packed payload")
    return items


class ThresholdCKKS:
    """Public parameters (ring dimension, scaling size) are agreed out of band."""

    def __init__(self, *, ring_dim: int = 1 << 14, scaling_mod_size: int = 50):
        self.of = load_openfhe()
        of = self.of
        params = of.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(1)
        params.SetScalingModSize(scaling_mod_size)
        params.SetBatchSize(ring_dim // 2)
        params.SetSecurityLevel(of.SecurityLevel.HEStd_128_classic)
        params.SetRingDim(ring_dim)
        self.cc = of.GenCryptoContext(params)
        for feature in (of.PKESchemeFeature.PKE, of.PKESchemeFeature.KEYSWITCH,
                        of.PKESchemeFeature.LEVELEDSHE, of.PKESchemeFeature.ADVANCEDSHE,
                        of.PKESchemeFeature.MULTIPARTY):
            self.cc.Enable(feature)
        self.slots = ring_dim // 2
        self.ring_dim, self.scaling_mod_size = ring_dim, scaling_mod_size

    def keygen_first(self):
        return self.cc.KeyGen()

    def keygen_join(self, public_bytes: bytes):
        return self.cc.MultipartyKeyGen(self.load_public(public_bytes))

    def serialize_public(self, keypair) -> bytes:
        return self.of.Serialize(keypair.publicKey, self.of.BINARY)

    def load_public(self, data: bytes):
        return self.of.DeserializePublicKeyString(data, self.of.BINARY)

    def encrypt(self, public_bytes: bytes, values) -> list[bytes]:
        public = self.load_public(public_bytes)
        flat = [float(v) for v in values]
        return [self.of.Serialize(self.cc.Encrypt(public, self.cc.MakeCKKSPackedPlaintext(
            flat[start:start + self.slots])), self.of.BINARY)
            for start in range(0, len(flat), self.slots)]

    def _load(self, data: bytes):
        return self.of.DeserializeCiphertextString(data, self.of.BINARY)

    def add(self, per_site: list[list[bytes]]) -> list[bytes]:
        chunks = len(per_site[0])
        if any(len(site) != chunks for site in per_site):
            raise ValueError("Mismatched ciphertext counts")
        out = []
        for index in range(chunks):
            total = self._load(per_site[0][index])
            for site in per_site[1:]:
                total = self.cc.EvalAdd(total, self._load(site[index]))
            out.append(self.of.Serialize(total, self.of.BINARY))
        return out

    def partial_decrypt(self, keypair, sums: list[bytes], *, lead: bool) -> list[bytes]:
        method = self.cc.MultipartyDecryptLead if lead else self.cc.MultipartyDecryptMain
        return [self.of.Serialize(method([self._load(s)], keypair.secretKey)[0], self.of.BINARY)
                for s in sums]

    def fuse(self, partials: list[list[bytes]], length: int) -> list[float]:
        """partials[0] must be the lead's contribution."""
        values: list[float] = []
        for index in range(len(partials[0])):
            plain = self.cc.MultipartyDecryptFusion([self._load(p[index]) for p in partials])
            take = min(self.slots, length - len(values))
            plain.SetLength(take)
            values.extend(plain.GetRealPackedValue())
        return values
