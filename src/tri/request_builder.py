"""Build requests from visible MIDI semantics, without leaking hidden targets."""
from collections.abc import Sequence
import numbers

from tri.domain.music import MusicSpec, CountRule
from tri.errors import InvalidSpecification


def sounding_pitches(tokens, initial_pitch):
    result, active = [], initial_pitch
    if initial_pitch is not None and (not isinstance(initial_pitch, numbers.Integral) or not 0 <= initial_pitch <= 127):
        raise InvalidSpecification("invalid initial sounding pitch")
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, numbers.Integral) or not 0 <= token <= 129:
            raise InvalidSpecification("source must contain output tokens only")
        token = int(token)
        if token == 0:
            active = None
        elif token >= 2:
            active = token - 2
        elif active is None:
            raise InvalidSpecification("source HOLD has no anchor")
        result.append(active)
    return result


def build_two_gap_request(source_tokens: Sequence[int], *, initial_pitch: int | None = None,
                          spans: Sequence[Sequence[int]] | None = None,
                          onsets_per_span: int = 1,
                          base_pitches: Sequence[int] = tuple(range(60, 73)),
                          motion_cost: float = 0.02) -> MusicSpec:
    """Use source only to construct observations, then discard hidden targets.

    Fixed note soundings are part of the preserved MIDI context. In particular,
    the audible pitch of a visible HOLD is known even if its onset is masked.
    No hidden onset pattern, hidden interior note, or reference answer is stored
    in MusicSpec. Counts are explicitly user-specified, not copied from truth.
    The vocabulary combines a fixed prior range with visible pitches only.
    """
    tokens = tuple(source_tokens)
    source_soundings = sounding_pitches(tokens, initial_pitch)
    length = len(tokens)
    if spans is None:
        if length < 16:
            raise InvalidSpecification("default two-gap request requires at least 16 cells")
        spans = (tuple(range(length // 4, length // 4 + 3)),
                 tuple(range(3 * length // 4, 3 * length // 4 + 3)))
    spans = tuple(tuple(span) for span in spans)
    if len(spans) != 2 or not spans[0] or len(spans[0]) != len(spans[1]):
        raise InvalidSpecification("exactly two equal, nonempty gap lengths required")
    for span in spans:
        if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) or not 0 <= i < length for i in span):
            raise InvalidSpecification("invalid gap position")
        if tuple(sorted(span)) != span or any(b != a + 1 for a, b in zip(span, span[1:])):
            raise InvalidSpecification("each gap must be contiguous and ordered")
    editable = tuple(i for span in spans for i in span)
    if len(set(editable)) != len(editable):
        raise InvalidSpecification("gaps must not overlap")
    visible = {i: int(v) for i, v in enumerate(tokens) if i not in editable}
    anchors = {i: source_soundings[i] for i in visible}
    vocabulary = set(base_pitches) | {v - 2 for v in visible.values() if v >= 2} | {p for p in anchors.values() if p is not None}
    if initial_pitch is not None:
        vocabulary.add(initial_pitch)
    return MusicSpec(length=length, pitches=tuple(sorted(vocabulary)), observed=visible,
                     fixed_soundings=anchors, initial_pitch=initial_pitch,
                     equal_onsets=tuple(zip(*spans)),
                     onset_counts=tuple(CountRule(span, onsets_per_span) for span in spans),
                     motion_cost=motion_cost)
