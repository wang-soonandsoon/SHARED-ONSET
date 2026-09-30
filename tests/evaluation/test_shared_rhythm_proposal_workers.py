import os

import pytest

from tri.evaluation import shared_rhythm_extension as study
from tri.evaluation.shared_rhythm_cases import make_shared_rhythm_case
from tri.evaluation.solver_cases import save_cases
from tri.errors import ZeroMass


MODES = ('onset_rejection_visible', 'onset_rejection_boundary_early', 'onset_rejection_visible_early')


def payload(tmp_path, backend):
    case = make_shared_rhythm_case(R=3, L=2, D=3, K=1, beta=.02)
    entry = save_cases([case], tmp_path / 'inputs')['cases'][0]
    return {**entry, 'backend': backend, 'draw_seeds': [231, 232, 233],
            'cpu': min(os.sched_getaffinity(0)), 'budget': {'max_factor_entries': 67108864,
            'max_workspace_bytes': 536870912}, 'max_proposals': 1000}


@pytest.mark.parametrize('backend', MODES)
def test_new_proposal_isolated_worker_keeps_full_samples_and_attempt_units(tmp_path, backend):
    row = study.isolated(payload(tmp_path, backend), worker_seconds=10, rss_limit_mib=512,
                         log_prefix=tmp_path / backend)
    assert row['status'] == 'completed', row
    assert row['verified_samples'] == 3
    assert not row['target_normalizer_available']
    assert 'target_partition' not in row['measurements']
    assert row['stats']['proposal_unit'] == 'shared_rhythm_attempt_with_completed_accept_or_reject_decision'
    assert row['stats']['cumulative_accepted_samples'] == 3
    assert row['stats']['cumulative_full_candidates'] >= 3
    assert row['last_rejection_progress']['progress']['in_flight'] is False


def test_frozen_feasible_input_zero_mass_is_a_correctness_failure(tmp_path, monkeypatch):
    def impossible(*args, **kwargs):
        raise ZeroMass('A finite-q witness exists, so this is a solver bug')
    monkeypatch.setattr(study, 'make_solver', impossible)
    events = []
    study.worker(payload(tmp_path, MODES[0]), emit=events.append)
    assert events[-1]['status'] == 'correctness_failure'
