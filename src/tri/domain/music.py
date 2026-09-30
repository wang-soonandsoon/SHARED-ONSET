"""Small, explicit monophonic music domain and independent semantic checker.

REST silences a slot, HOLD continues an active pitch, and NOTE starts a pitch.
Motion constraints concern the pitch sounding immediately before a new NOTE;
they do not connect onsets across a REST. Range and pitch-class rules apply to
all sounding slots, including HOLD. MASK/PAD belong to model input only.
Observed tokens fix token syntax. fixed_soundings additionally anchors the
original sounding pitch of visible context, which matters especially for HOLD
immediately after an editable gap. Newly revealed tokens need not have anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from sys import float_info
from types import MappingProxyType
from typing import Mapping, Sequence

from tri.errors import InvalidSpecification

REST = 0
HOLD = 1
MASK = 130
PAD = 131
VOCAB_SIZE = 130
SILENCE = -1  # Internal sounding-state value; never a token.


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InvalidSpecification(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise InvalidSpecification(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _sequence(value: object, name: str) -> tuple:
    if isinstance(value, (str, bytes)):
        raise InvalidSpecification(f"{name} must be a sequence, not text")
    try:
        return tuple(value)
    except TypeError as exc:
        raise InvalidSpecification(f"{name} must be an iterable sequence") from exc


def _mapping(value: object, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise InvalidSpecification(f"{name} must be a mapping")
    return value


def note_token(pitch: int) -> int:
    return _integer(pitch, "MIDI pitch", 0, 127) + 2


def token_pitch(token: int) -> int | None:
    """Return a NOTE's MIDI pitch; REST/HOLD have no intrinsic pitch."""
    value = _integer(token, "output token", 0, VOCAB_SIZE - 1)
    return value - 2 if value >= 2 else None


@dataclass(frozen=True)
class CountRule:
    positions: tuple[int, ...]
    count: int

    def __post_init__(self) -> None:
        positions = tuple(_integer(p, "count position", 0, 2**63 - 1) for p in _sequence(self.positions, "count positions"))
        if len(set(positions)) != len(positions):
            raise InvalidSpecification("CountRule positions must not repeat")
        count = _integer(self.count, "onset count", 0, len(positions))
        object.__setattr__(self, "positions", tuple(sorted(positions)))
        object.__setattr__(self, "count", count)


@dataclass(frozen=True)
class MusicSpec:
    length: int
    pitches: tuple[int, ...]
    observed: Mapping[int, int] = field(default_factory=dict)
    initial_pitch: int | None = None
    enforce_end: bool = False
    end_pitch: int | None = None
    equal_onsets: tuple[tuple[int, int], ...] = ()
    onset_counts: tuple[CountRule, ...] = ()
    pitch_ranges: Mapping[int, tuple[int, int]] = field(default_factory=dict)
    pitch_classes: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    max_adjacent_interval: int | None = None
    motion_cost: float = 0.0
    fixed_soundings: Mapping[int, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = _integer(self.length, "length", 1, 2**31 - 1)
        pitches = tuple(_integer(p, "working MIDI pitch", 0, 127) for p in _sequence(self.pitches, "pitches"))
        if len(set(pitches)) != len(pitches):
            raise InvalidSpecification("Working pitches must be unique")
        pitches = tuple(sorted(pitches))
        position = lambda p: _integer(p, "position", 0, length - 1)
        observed = {}
        for pos, token in _mapping(self.observed, "observed").items():
            pos = position(pos)
            token = _integer(token, "observed output token", 0, VOCAB_SIZE - 1)
            if token >= 2 and token - 2 not in pitches:
                raise InvalidSpecification("Observed NOTE pitch must belong to pitches")
            observed[pos] = token
        anchors = {}
        for pos, pitch in _mapping(self.fixed_soundings, "fixed_soundings").items():
            pos = position(pos)
            if pos not in observed:
                raise InvalidSpecification("fixed_soundings keys must be observed positions")
            anchors[pos] = None if pitch is None else _integer(pitch, "fixed sounding pitch", 0, 127)
        boundaries = {}
        for name in ("initial_pitch", "end_pitch"):
            pitch = getattr(self, name)
            boundaries[name] = None if pitch is None else _integer(pitch, name, 0, 127)
        if not isinstance(self.enforce_end, bool):
            raise InvalidSpecification("enforce_end must be a bool")
        if not self.enforce_end and self.end_pitch is not None:
            raise InvalidSpecification("end_pitch requires enforce_end=True")
        pairs = []
        for pair in _sequence(self.equal_onsets, "equal_onsets"):
            pair = _sequence(pair, "onset pair")
            if len(pair) != 2:
                raise InvalidSpecification("An equal_onsets item must contain two positions")
            pairs.append((position(pair[0]), position(pair[1])))
        counts = _sequence(self.onset_counts, "onset_counts")
        for rule in counts:
            if not isinstance(rule, CountRule):
                raise InvalidSpecification("onset_counts must contain CountRule instances")
            for pos in rule.positions:
                position(pos)
        ranges = {}
        for pos, bounds in _mapping(self.pitch_ranges, "pitch_ranges").items():
            bounds = _sequence(bounds, "pitch range")
            if len(bounds) != 2:
                raise InvalidSpecification("Pitch range requires (minimum, maximum)")
            low, high = (_integer(p, "range pitch", 0, 127) for p in bounds)
            if low > high:
                raise InvalidSpecification("Pitch range minimum exceeds maximum")
            ranges[position(pos)] = (low, high)
        classes = {}
        for pos, values in _mapping(self.pitch_classes, "pitch_classes").items():
            classes[position(pos)] = tuple(sorted(set(_integer(p, "pitch class", 0, 11) for p in _sequence(values, "pitch classes"))))
        interval = self.max_adjacent_interval
        if interval is not None:
            interval = _integer(interval, "max_adjacent_interval", 0, 127)
        if isinstance(self.motion_cost, bool) or not isinstance(self.motion_cost, Real):
            raise InvalidSpecification("motion_cost must be a finite nonnegative number")
        cost = float(self.motion_cost)
        if not isfinite(cost) or cost < 0:
            raise InvalidSpecification("motion_cost must be a finite nonnegative number")
        if cost > float_info.max / (length * 127):
            raise InvalidSpecification("motion_cost is too large for finite sequence scores")
        for name, value in {
            "length": length, "pitches": pitches, "observed": MappingProxyType(observed),
            **boundaries, "equal_onsets": tuple(pairs), "onset_counts": counts,
            "pitch_ranges": MappingProxyType(ranges), "pitch_classes": MappingProxyType(classes),
            "max_adjacent_interval": interval, "motion_cost": cost,
            "fixed_soundings": MappingProxyType(anchors),
        }.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    violations: tuple[str, ...]
    soft_score: float


def verify_music(tokens: Sequence[int], spec: MusicSpec) -> VerificationResult:
    """Simulate notes directly, without using compiled factors or inference.

    A bad sequence is returned as violations; a bad request is rejected when
    constructing MusicSpec. soft_score is meaningful for valid sequences.
    """
    try:
        values = _sequence(tokens, "tokens")
    except InvalidSpecification as exc:
        return VerificationResult(False, (str(exc),), 0.0)
    violations: list[str] = []
    if len(values) != spec.length:
        return VerificationResult(False, (f"length: expected {spec.length}, got {len(values)}",), 0.0)
    if any(isinstance(t, bool) or not isinstance(t, Integral) or not 0 <= t < VOCAB_SIZE for t in values):
        return VerificationResult(False, ("tokens must be output IDs 0..129; MASK/PAD are input only",), 0.0)
    values = tuple(int(t) for t in values)
    sounding = spec.initial_pitch
    jumps = 0
    for i, token in enumerate(values):
        if i in spec.observed and token != spec.observed[i]:
            violations.append(f"slot {i}: observed token changed")
        previous = sounding
        if token == REST:
            sounding = None
        elif token == HOLD:
            if sounding is None:
                violations.append(f"slot {i}: HOLD without an active pitch")
        else:
            sounding = token - 2
            if sounding not in spec.pitches:
                violations.append(f"slot {i}: NOTE pitch {sounding} outside working vocabulary")
            if previous is not None:
                jump = abs(sounding - previous)
                jumps += jump
                if spec.max_adjacent_interval is not None and jump > spec.max_adjacent_interval:
                    violations.append(f"slot {i}: adjacent interval {jump} exceeds limit")
        if sounding is not None:
            if i in spec.pitch_ranges:
                low, high = spec.pitch_ranges[i]
                if not low <= sounding <= high:
                    violations.append(f"slot {i}: sounding pitch outside range")
            if i in spec.pitch_classes and sounding % 12 not in spec.pitch_classes[i]:
                violations.append(f"slot {i}: sounding pitch class disallowed")
        if i in spec.fixed_soundings and sounding != spec.fixed_soundings[i]:
            violations.append(f"slot {i}: fixed sounding pitch changed from {spec.fixed_soundings[i]} to {sounding}")
    for left, right in spec.equal_onsets:
        if (values[left] >= 2) != (values[right] >= 2):
            violations.append(f"onset equality failed at ({left}, {right})")
    for rule in spec.onset_counts:
        count = sum(values[pos] >= 2 for pos in rule.positions)
        if count != rule.count:
            violations.append(f"onset count at {rule.positions}: expected {rule.count}, got {count}")
    if spec.enforce_end and sounding != spec.end_pitch:
        violations.append(f"end sounding pitch: expected {spec.end_pitch}, got {sounding}")
    return VerificationResult(not violations, tuple(violations), -spec.motion_cost * jumps)
