import pytest

from tri.domain.music import verify_music
from tri.errors import InvalidSpecification
from tri.request_builder import build_two_gap_request


def test_hidden_interior_notes_do_not_change_request_or_vocabulary():
    # The hidden 90 vs 30 pitch ends before visible context resumes. It must
    # not leak into working vocabulary, anchors or onset-count request.
    left = [0, 92, 0, 62, 1, 0, 66, 0, 62, 0]
    right = [0, 32, 0, 62, 1, 0, 66, 0, 62, 0]
    kwargs = dict(spans=((1, 2), (6, 7)))
    a, b = build_two_gap_request(left, **kwargs), build_two_gap_request(right, **kwargs)
    assert a == b
    assert 90 not in a.pitches and 30 not in a.pitches
    assert 1 not in a.observed and 2 not in a.fixed_soundings


def test_visible_hold_pitch_is_explicit_context_and_cannot_change():
    original = [0, 62, 1, 1, 0, 66, 1, 1]
    spec = build_two_gap_request(original, spans=((1, 2), (5, 6)))
    assert spec.fixed_soundings[3] == 60
    assert spec.fixed_soundings[7] == 64
    corrupted = [0, 66, 1, 1, 0, 66, 1, 1]
    assert all(corrupted[i] == t for i, t in spec.observed.items())
    assert not verify_music(corrupted, spec).valid
    assert verify_music(original, spec).valid


@pytest.mark.parametrize("spans", [((1, 2), (2, 3)), ((1, 3), (5, 6)), ((1,), (5, 6))])
def test_malformed_gaps_rejected(spans):
    with pytest.raises(InvalidSpecification):
        build_two_gap_request([0] * 10, spans=spans)
