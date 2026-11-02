"""Lazy OpenFHE threshold-BGV adapter: exact integer sums, multiparty noise flooding.

NOISE_FLOODING_MULTIPARTY needs the number of parties at context creation, so a
context is built per session. In flooding mode the ciphertext modulus grows
(measured: 787 KB -> 1.31 MB per ciphertext at ring 2^14), which CKKS did not show.
Semi-honest, n-of-n; this does not prove the partial decryptions are simulatable.
"""
from pqfl.he.ckks import load_openfhe


def _is_prime(n: int) -> bool:
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n < 2:
        return False
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for a in small:  # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def batching_prime(ring_dim: int, bits: int) -> int:
    """Smallest prime above 2^bits congruent to 1 mod 2*ring_dim (enables slot packing)."""
    m = 2 * ring_dim
    k = (1 << bits) // m + 1
    while not _is_prime(k * m + 1):
        k += 1
    return k * m + 1


class ThresholdBGV:
    def __init__(self, *, parties: int, ring_dim: int = 1 << 14, plaintext_bits: int = 31,
                 flooding: bool = True):
        if type(parties) is not int or parties < 2:
            raise ValueError("Require at least two parties")
        self.of = load_openfhe()
        of = self.of
        self.plaintext_modulus = batching_prime(ring_dim, plaintext_bits)
        params = of.CCParamsBGVRNS()
        params.SetPlaintextModulus(self.plaintext_modulus)
        params.SetMultiplicativeDepth(1)
        params.SetRingDim(ring_dim)
        params.SetSecurityLevel(of.SecurityLevel.HEStd_128_classic)
        params.SetMultipartyMode(of.MultipartyMode.NOISE_FLOODING_MULTIPARTY if flooding
                                 else of.MultipartyMode.FIXED_NOISE_MULTIPARTY)
        params.SetThresholdNumOfParties(parties)
        self.cc = of.GenCryptoContext(params)
        for feature in (of.PKESchemeFeature.PKE, of.PKESchemeFeature.KEYSWITCH,
                        of.PKESchemeFeature.LEVELEDSHE, of.PKESchemeFeature.ADVANCEDSHE,
                        of.PKESchemeFeature.MULTIPARTY):
            self.cc.Enable(feature)
        self.slots = ring_dim
        self.ring_dim, self.parties, self.flooding = ring_dim, parties, flooding

    def keygen_first(self):
        return self.cc.KeyGen()

    def keygen_join(self, public_bytes: bytes):
        return self.cc.MultipartyKeyGen(self.load_public(public_bytes))

    def serialize_public(self, keypair) -> bytes:
        return self.of.Serialize(keypair.publicKey, self.of.BINARY)

    def load_public(self, data: bytes):
        return self.of.DeserializePublicKeyString(data, self.of.BINARY)

    def _load(self, data: bytes):
        return self.of.DeserializeCiphertextString(data, self.of.BINARY)

    def encrypt(self, public_bytes: bytes, values: list[int]) -> list[bytes]:
        public = self.load_public(public_bytes)
        return [self.of.Serialize(self.cc.Encrypt(public, self.cc.MakePackedPlaintext(
            [int(v) for v in values[start:start + self.slots]])), self.of.BINARY)
            for start in range(0, len(values), self.slots)]

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

    def fuse(self, partials: list[list[bytes]], length: int) -> list[int]:
        """partials[0] must be the lead's contribution; returns centered integers."""
        values: list[int] = []
        for index in range(len(partials[0])):
            plain = self.cc.MultipartyDecryptFusion([self._load(p[index]) for p in partials])
            plain.SetLength(min(self.slots, length - len(values)))
            values.extend(plain.GetPackedValue())
        return values
