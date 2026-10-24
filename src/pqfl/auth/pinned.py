import hashlib
import hmac

from pqfl.auth.base import AuthenticationError, AuthProof


class PinnedFingerprintAuth:
    """Per-verifier registries: verifier -> (epoch, owner) -> SHA-256 hex.

    Empty configuration rejects every received key. No trust-on-first-use.
    """

    def __init__(self, registries=None):
        self.registries = {verifier: dict(pins) for verifier, pins in (registries or {}).items()}
        for pins in self.registries.values():
            for fingerprint in pins.values():
                if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                    raise ValueError('Expected a SHA-256 hex fingerprint')
                try:
                    if len(bytes.fromhex(fingerprint)) != 32:
                        raise ValueError("Invalid SHA-256 length")
                except ValueError:
                    raise ValueError('Invalid SHA-256 fingerprint') from None

    @classmethod
    def from_config(cls, config):
        # YAML registry: verifier -> epoch -> owner -> public fingerprint.
        return cls({verifier: {(int(epoch), owner): fingerprint.lower()
                               for epoch, owners in epochs.items()
                               for owner, fingerprint in owners.items()}
                    for verifier, epochs in config.items()})

    def issue(self, site_id: str, epoch: int, public_key: bytes) -> AuthProof:
        return AuthProof()

    def verify(self, verifier: str, site_id: str, epoch: int,
               public_key: bytes, proof: AuthProof) -> None:
        expected = self.registries.get(verifier, {}).get((epoch, site_id))
        actual = hashlib.sha256(public_key).hexdigest()
        if (proof.identity_public or proof.signature or expected is None
                or not hmac.compare_digest(expected.lower(), actual)):
            raise AuthenticationError('Pinned public-key fingerprint mismatch or missing pin')
