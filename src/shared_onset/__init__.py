"""Exact shared-onset inference, full-target sampling, and public music semantics."""
from .contracts import BatchSample, Budget
from .errors import BudgetExceeded, InvalidSpecification, UnsupportedSpec, ZeroMass
from .music import CountRule, MusicSpec, note_token, verify_music
from .onset_reset import OnsetResetMusicInference

__all__ = [
    "BatchSample", "Budget", "BudgetExceeded", "CountRule",
    "InvalidSpecification", "MusicSpec", "OnsetResetMusicInference",
    "UnsupportedSpec", "ZeroMass", "note_token", "verify_music",
]

from tri.sampling.onset_rejection import OnsetRejectionSampler, OnsetRejectionDraw
from tri.sampling.onset_rejection_adaptive import AdaptiveOnsetRejectionSampler
from tri.inference.product_multi import MultiSpanProductChainMusicInference
__all__ += ["OnsetRejectionSampler", "OnsetRejectionDraw", "AdaptiveOnsetRejectionSampler", "MultiSpanProductChainMusicInference"]
