"""Explicit request semantics, independent verification, and compilation."""

from .music import (
    HOLD, MASK, PAD, REST, VOCAB_SIZE, CountRule, MusicSpec,
    VerificationResult, note_token, token_pitch, verify_music,
)

__all__ = [
    "REST", "HOLD", "MASK", "PAD", "VOCAB_SIZE", "CountRule", "MusicSpec",
    "VerificationResult", "note_token", "token_pitch", "verify_music",
]
