from dataclasses import replace

import numpy as np

from tri.evaluation.decision_cases import with_probes
from tri.evaluation.solver_cases import make_controlled_case


def test_probes_preserve_original_target_and_repeat_identity():
    case = make_controlled_case(L=8, D=4, K=2)
    first, second = (with_probes(case, r) for r in (0, 1))
    assert first.spec is case.spec and first.logq is case.logq
    assert first.witness == case.witness
    assert first.case_id != second.case_id
    assert first.metadata['source_case_id'] == second.metadata['source_case_id'] == case.case_id
    assert first.metadata['query_evidence'] == second.metadata['query_evidence']
    assert first.metadata['query_evidence'][:3] == [{'y1': 0}, {'y5': 0}, {'y8': 0}]
    assert 'timing_repetition' not in case.metadata
    assert first.metadata['query_evidence'][3] == {'y1': 0, 'y8': 0}


def test_fully_known_first_span_probes_unknown_second_span_with_fixed_witness():
    original = make_controlled_case(L=8, D=4, K=2, observed_fraction=1.)
    case = replace(original, metadata={**original.metadata, 'family': 'learned'})
    query = with_probes(case, 0)
    assert query.metadata['query_side'] == 1
    assert len(query.metadata['query_evidence'][-1]) == 2
    assert query.metadata['query_evidence'][-1] != query.metadata['query_evidence'][2]
    for evidence in query.metadata['query_evidence']:
        for name, token in evidence.items():
            pos = int(name[1:])
            assert pos not in case.spec.observed
            assert token == case.witness[pos]
    np.testing.assert_array_equal(query.logq, case.logq)
