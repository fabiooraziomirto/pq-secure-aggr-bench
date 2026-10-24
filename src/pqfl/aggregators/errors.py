class KeyConfirmationError(RuntimeError):
    """Peer key confirmation failed; messages never include secret material."""


class AggregateOverflowError(ValueError):
    """The simulator's signed integer aggregate is outside the selected range."""


class MissingParticipantError(ValueError):
    """all_or_abort: a missing or extra participant prevents any result."""
