"""Predeclared evidence workloads on existing, unchanged solver targets.

Three independent process repetitions quantify timing variation, not extra
musical examples. Controlled singleton probes reproduce the earlier REST
diagnostic; learned probes follow the already saved original paired joint draw as a feasible witness. No target
is selected by a new solver's performance or by listening.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from dataclasses import replace

from tri.evaluation.solver_cases import SolverCase, load_case, save_cases


def with_probes(case, repetition):
    pairs = sorted(case.spec.equal_onsets)
    spans = [[pair[side] for pair in pairs] for side in (0, 1)]
    free = [[pos for pos in span if pos not in case.spec.observed] for span in spans]
    # A fully visible first span must not produce vacuous fixed-token queries.
    side = 0 if len(free[0]) >= 3 else 1
    positions = free[side]
    if len(positions) < 3 or case.witness is None:
        raise ValueError('Decision probes require three free positions and a fixed witness')
    chosen = [positions[0], positions[len(positions)//2], positions[-1]]
    controlled = case.metadata['family'] == 'controlled'
    tokens = [0 if controlled else int(case.witness[pos]) for pos in chosen]
    probes = [{f'y{pos}': token} for pos, token in zip(chosen, tokens)]
    labels = ['early_singleton', 'middle_singleton', 'late_singleton']
    probes.append({**probes[0], **probes[2]})
    labels.append('early_and_late_joint')
    # The final probe changes both chains when both have unknown tokens;
    # otherwise use two late unknown cells, avoiding a duplicate scalar hit.
    tail_positions = [positions[-1] for positions in free if positions]
    if len(tail_positions) == 1:
        tail_positions = positions[-2:]
    probes.append({f'y{pos}': int(case.witness[pos]) for pos in tail_positions})
    labels.append('late_joint_witness')
    offsets = [{name: next(j for j, pair in enumerate(pairs) if int(name[1:]) in pair)
                for name in evidence} for evidence in probes]
    metadata = copy.deepcopy(case.metadata)
    metadata.update(source_case_id=case.case_id, timing_repetition=repetition,
                    query_evidence=probes, query_labels=labels, query_offsets=offsets,
                    query_side=side, sweeps=[],
                    query_source=('fixed REST singletons; joint witness final probe' if controlled
                                  else 'existing original paired joint draw as witness; no ranking or new sample'),
                    experiment='decision_equal_prefix_v1')
    return SolverCase(f'decision_{case.case_id}_r{repetition}', case.spec, case.logq,
                      metadata, case.witness)


def prepare(source='runs/solver_study/cases/cases.json', output='runs/decision_study/cases'):
    manifest = json.loads(Path(source).read_text())
    by_id = {entry['case_id']: entry for entry in manifest['cases']}
    identifiers = [f'controlled_L{L}_D{D}_K{K}_v0of{L}_note_intervalany_s{seed}'
                   for L, D, K in ((32, 16, 8), (16, 32, 4), (64, 16, 16))
                   for seed in (20260921, 20260922, 20260923)]
    identifiers += [f'learned_{stage}_{kind}' for stage in ('short', 'bar8', 'bar16')
                    for kind in ('unknown', 'partial', 'known', 'harmony')]
    original = [load_case(Path(by_id[name]['path'])) for name in identifiers]
    previous_path = Path(source).parent.parent / 'final_comparison/results.jsonl'
    previous = [json.loads(line) for line in previous_path.read_text().splitlines()]
    witnesses = {row['case_id']: row['measurements']['cold_full_sample']['draw']['tokens']
                 for row in previous if row['backend'] == 'paired'
                 and 'cold_full_sample' in row.get('measurements', {})}
    original = [replace(case, witness=tuple(witnesses[case.case_id]),
                        metadata={**case.metadata, 'query_witness_source':
                                  {'path': str(previous_path.resolve()), 'backend': 'paired',
                                   'operation': 'cold_full_sample', 'selection': 'the existing single draw; no ranking or resampling'}})
                if case.witness is None else case for case in original]
    # Interleave independent repetitions across targets rather than repeating
    # one target in a single process with a warmed allocator/message cache.
    cases = [with_probes(case, repetition) for repetition in range(3) for case in original]
    return save_cases(cases, Path(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/solver_study/cases/cases.json')
    parser.add_argument('--out', default='runs/decision_study/cases')
    args = parser.parse_args()
    manifest = prepare(args.source, args.out)
    print(json.dumps({'targets': 21, 'repetitions': 3, 'records': len(manifest['cases'])}))
