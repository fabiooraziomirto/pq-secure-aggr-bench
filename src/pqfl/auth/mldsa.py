"""Standard ML-DSA-65 through the official liboqs Python binding."""
import hmac
import json

from pqfl.auth.base import AuthenticationError, AuthProof
from pqfl.kem.backends import BackendUnavailable, load_oqs


def authentication_message(site_id: str, epoch: int, public_key: bytes) -> bytes:
    if type(epoch) is not int or epoch < 0:
        raise ValueError('Invalid authentication epoch')
    return json.dumps(['pqfl/auth/v1', epoch, site_id, public_key.hex()],
                      ensure_ascii=True, separators=(',', ':')).encode()


class MLDSAAuth:
    """Long-term identities and per-verifier pinned identity public keys.

    Secret identity keys are injected from secure storage, never configuration
    values committed to the repository. No identity key is generated implicitly.
    """

    def __init__(self, identity_public_keys, secret_keys, trusted_identity_keys):
        self._oqs = load_oqs()
        self.mechanism = 'ML-DSA-65'
        if self.mechanism not in self._oqs.get_enabled_sig_mechanisms():
            raise BackendUnavailable('ML-DSA-65 is not enabled in native liboqs')
        self.public_keys = dict(identity_public_keys)
        self._secret_keys = dict(secret_keys)
        self.trusted = {site: dict(keys) for site, keys in trusted_identity_keys.items()}
        with self._oqs.Signature(self.mechanism) as signer:
            self.public_key_bytes = signer.details['length_public_key']
            self.secret_key_bytes = signer.details['length_secret_key']
            self.signature_bytes = signer.details['length_signature']
        if any(len(key) != self.public_key_bytes for key in self.public_keys.values()):
            raise ValueError('Invalid identity public-key length')
        if any(len(key) != self.secret_key_bytes for key in self._secret_keys.values()):
            raise ValueError('Invalid identity secret-key length')

    def issue(self, site_id: str, epoch: int, public_key: bytes) -> AuthProof:
        if site_id not in self.public_keys or site_id not in self._secret_keys:
            raise AuthenticationError('Missing local signing identity')
        with self._oqs.Signature(self.mechanism, secret_key=self._secret_keys[site_id]) as signer:
            signature = bytes(signer.sign(authentication_message(site_id, epoch, public_key)))
        if len(signature) != self.signature_bytes:
            raise AuthenticationError('Unexpected identity signature length')
        return AuthProof(self.public_keys[site_id], signature)

    def verify(self, verifier: str, site_id: str, epoch: int,
               public_key: bytes, proof: AuthProof) -> None:
        pinned = self.trusted.get(verifier, {}).get(site_id)
        if pinned is None or not hmac.compare_digest(pinned, proof.identity_public):
            raise AuthenticationError('Untrusted identity public key')
        if len(proof.signature) != self.signature_bytes:
            raise AuthenticationError('Invalid identity signature length')
        with self._oqs.Signature(self.mechanism) as verifier_backend:
            valid = verifier_backend.verify(authentication_message(site_id, epoch, public_key),
                                            proof.signature, pinned)
        if not valid:
            raise AuthenticationError('Identity signature verification failed')
