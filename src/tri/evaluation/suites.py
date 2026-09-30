"""Visible-context music requests and descriptive output statistics."""
from dataclasses import dataclass

import numpy as np

from tri.domain.music import CountRule, MusicSpec
from tri.errors import InvalidSpecification
from tri.request_builder import sounding_pitches

SUITES = ('unknown4','unknown8','partial8','known8','harmony8')


def make_request(tokens, initial_pitch, suite, chord_features=None):
    if suite not in SUITES:
        raise InvalidSpecification(f'unknown suite {suite}')
    length=len(tokens)
    width=4 if suite=='unknown4' else 8
    if length<4*width:
        raise InvalidSpecification(f'{suite} needs at least {4*width} cells')
    starts=(length//4-width//2,3*length//4-width//2)
    spans=tuple(tuple(range(start,start+width)) for start in starts)
    editable=set(spans[0]+spans[1])
    if suite=='known8':
        editable.difference_update(spans[0])
    elif suite=='partial8':
        editable.difference_update((spans[0][0],spans[0][3]))
    visible={i:int(t) for i,t in enumerate(tokens) if i not in editable}
    source_sound=sounding_pitches(tokens,initial_pitch)
    anchors={i:source_sound[i] for i in visible}
    # The working register is fixed before any request. Only visible context
    # and its actual boundary sounding pitches extend it; hidden notes do not.
    pitches=set(range(48,85))|{t-2 for t in visible.values() if t>=2}|{p for p in anchors.values() if p is not None}
    if initial_pitch is not None:
        pitches.add(initial_pitch)
    count=sum(visible[i]>=2 for i in spans[0]) if suite=='known8' else 2
    classes={}
    if suite=='harmony8':
        if chord_features is None:
            raise InvalidSpecification('harmony suite needs external chord features')
        for i in editable:
            if chord_features[i,37] and not chord_features[i,36]:
                classes[i]=tuple(int(p) for p in np.flatnonzero(chord_features[i,12:24]))
    return MusicSpec(length=length,pitches=tuple(sorted(pitches)),observed=visible,
                     initial_pitch=initial_pitch,fixed_soundings=anchors,
                     equal_onsets=tuple(zip(*spans)),
                     onset_counts=tuple(CountRule(span,count) for span in spans),
                     pitch_classes=classes,motion_cost=.02)


def music_statistics(tokens, source, spec, chord_features):
    """Descriptive proxies; none is a replacement for listening judgments."""
    generated=sounding_pitches(tokens,spec.initial_pitch)
    editable=[i for i in range(spec.length) if i not in spec.observed]
    onset=[i for i in editable if tokens[i]>=2]
    harmony=[]
    for i in editable:
        if generated[i] is not None and chord_features[i,37] and not chord_features[i,36]:
            harmony.append(float(chord_features[i,12+generated[i]%12]>0))
    jumps=[]
    for i in range(1,spec.length):
        if (i in editable or i-1 in editable) and tokens[i]>=2 and generated[i-1] is not None:
            jumps.append(abs(generated[i]-generated[i-1]))
    return {
        'editable_cells':len(editable),'editable_onsets':len(onset),
        'editable_rest_fraction':float(np.mean([tokens[i]==0 for i in editable])) if editable else None,
        'editable_pitch_classes':sorted({generated[i]%12 for i in editable if generated[i] is not None}),
        'chord_tone_fraction':float(np.mean(harmony)) if harmony else None,
        'chord_evaluated_cells':len(harmony),
        'mean_edit_boundary_or_internal_jump':float(np.mean(jumps)) if jumps else None,
        'reconstruction_token_accuracy_diagnostic':float(np.mean([tokens[i]==source[i] for i in editable])) if editable else None,
        'onset_pattern': [int(tokens[i]>=2) for i in editable],
        'editable_tokens':[int(tokens[i]) for i in editable],
    }
