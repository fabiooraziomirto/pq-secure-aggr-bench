"""Authentication boundary: trust stores are provisioned out of band."""
from dataclasses import dataclass
from typing import Protocol


class AuthenticationError(RuntimeError):
    """Public-key authentication failed, without secret details in messages."""


@dataclass(frozen=True)
class AuthProof:
    identity_public: bytes = b''
    signature: bytes = b''


class Authenticator(Protocol):
    def issue(self, site_id: str, epoch: int, public_key: bytes) -> AuthProof: ...

    def verify(self, verifier: str, site_id: str, epoch: int,
               public_key: bytes, proof: AuthProof) -> None: ...
