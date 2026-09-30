"""Reference reveal schedule and a full-support, state-only mixture.

The guide selects positions before seeing sampled values. Probabilities refer
to the complete mixture, including sets reachable from both components.
"""
from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np

from tri.errors import InvalidSpecification


@dataclass(frozen=True)
class RevealDraw:
    positions: tuple[int, ...]
    log_reference: float
    log_proposal: float


def _positions(values: Sequence[int], label: str) -> tuple[int, ...]:
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) or i < 0 for i in values):
        raise InvalidSpecification(f"{label} must contain nonnegative integer positions")
    result = tuple(sorted(int(i) for i in values))
    if len(result) != len(set(result)):
        raise InvalidSpecification(f"{label} contains duplicate positions")
    return result


class RevealSchedule:
    """Linear-noise ancestral schedule, with optional deterministic guide.

    Default b_t=1/(T-t). Empty reveals advance model time. Completed states
    are absorbing; the final round reveals all remaining positions.
    """

    def __init__(self, steps: int, epsilon: float = 1.0,
                 rates: Sequence[float] | None = None):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise InvalidSpecification("steps must be a positive integer")
        if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise InvalidSpecification("epsilon must satisfy 0 < epsilon <= 1")
        self.steps, self.epsilon = steps, float(epsilon)
        self.rates = tuple(float(v) for v in rates) if rates is not None else tuple(1 / (steps - t) for t in range(steps))
        if len(self.rates) != steps or any(not math.isfinite(b) or not 0 < b < 1 for b in self.rates[:-1]) or self.rates[-1] != 1:
            raise InvalidSpecification("rates must be in (0,1), with final rate exactly 1")

    def log_probs(self, step: int, missing: Sequence[int], positions: Sequence[int],
                  guide: Sequence[int] | None = None) -> tuple[float, float]:
        if not isinstance(step, int) or not 0 <= step < self.steps:
            raise InvalidSpecification("step outside schedule")
        remaining, selected = _positions(missing, "missing"), _positions(positions, "selected")
        if not set(selected).issubset(remaining):
            raise InvalidSpecification("selected positions must be missing")
        guided = _positions(guide if guide is not None else remaining[:1], "guide")
        if not set(guided).issubset(remaining):
            raise InvalidSpecification("guide positions must be missing")
        if not remaining or step == self.steps - 1:
            value = 0.0 if selected == remaining else -math.inf
            return value, value
        b = self.rates[step]
        log_ref = len(selected) * math.log(b) + (len(remaining) - len(selected)) * math.log1p(-b)
        log_prop = math.log(self.epsilon) + log_ref
        if selected == guided and self.epsilon < 1:
            log_prop = float(np.logaddexp(log_prop, math.log1p(-self.epsilon)))
        return log_ref, log_prop

    def sample(self, step: int, missing: Sequence[int], rng: np.random.Generator,
               guide: Sequence[int] | None = None) -> RevealDraw:
        remaining = _positions(missing, "missing")
        guided = _positions(guide if guide is not None else remaining[:1], "guide")
        # Validate before indexing rates or advancing RNG.
        self.log_probs(step, remaining, (), guided)
        if not remaining or step == self.steps - 1:
            chosen = remaining
        elif rng.random() < self.epsilon:
            chosen = tuple(i for i, keep in zip(remaining, rng.random(len(remaining)) < self.rates[step]) if keep)
        else:
            chosen = guided
        log_ref, log_prop = self.log_probs(step, remaining, chosen, guided)
        return RevealDraw(chosen, log_ref, log_prop)
