from dataclasses import replace
import json
import time

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec
from tri.evaluation.query_workload import QueryWorkloadRecorder, json_value
from tri.sampling.research_methods import research_decode


def setup_case(width=4):
    left = tuple(range(1, width + 1))
    right = tuple(range(width + 2, 2 * width + 2))
    observed = {0: 62, width + 1: 62, 2 * width + 2: 62}
    spec = MusicSpec(length=2 * width + 3, pitches=(60, 62), observed=observed,
        fixed_soundings={i: 60 for i in observed}, equal_onsets=tuple(zip(left, right)),
        onset_counts=(CountRule(left, width // 2), CountRule(right, width // 2)), motion_cost=.1)
    def provider(state, noise):
        known = sum(value is not None for value in state)
        logits = np.full((len(state), 130), -3.)
        logits[:, 0] = .1 * known
        logits[:, 1] = 1.3 * noise
        logits[:, 62] = -.3 * noise
        logits[:, 64] = .2
        return logits - logsumexp(logits, axis=1, keepdims=True)
    return spec, provider


@pytest.mark.parametrize('method', ['one_shot_joint', 'tri_direct', 'smc_4'])
def test_observational_wrappers_preserve_real_decoder_seeded_outputs(method):
    spec, provider = setup_case()
    seed = 513
    expected = research_decode(method, spec, provider, steps=4, seed=seed, backend='paired')
    recorder = QueryWorkloadRecorder(provider)
    with recorder.instrument_factories():
        started = time.perf_counter()
        actual = research_decode(method, spec, recorder, steps=4, seed=seed, backend='paired')
        elapsed = time.perf_counter() - started
    assert actual.tokens == expected.tokens
    assert actual.model_calls == expected.model_calls == len(recorder.models)
    assert json_value(actual.trace) == json_value(expected.trace)
    tape = recorder.tape(actual, elapsed)
    json.dumps(tape, allow_nan=False)
    summary = tape['summary']
    assert summary['model_calls'] == len(tape['engines'])
    assert summary['public_nonempty_partition_conditions'] == 0
    assert summary['repeated_public_partition_conditions'] == 0
    assert summary['provider_seconds'] > 0
    assert summary['solver_constructor_and_public_api_seconds'] > 0
    assert summary['instrumented_decode_wall_seconds'] >= summary['provider_seconds'] + summary['solver_constructor_and_public_api_seconds']
    assert all(call['application_level'] for call in tape['application_calls'])
    assert all('rng_state_before' in call and 'sampled_assignment' in call
               for call in tape['application_calls'] if call['operation'] == 'sample_batch')
    for engine in tape['engines']:
        assert engine['q_array_key'] in recorder.arrays
        assert engine['spec']['length'] == spec.length
    if method == 'one_shot_joint':
        assert summary['model_calls'] == 1
        assert summary['public_solver_operations'] == {'log_partition': 1, 'sample_batch': 1}
        assert summary['actual_forward_calls'] == 3
        assert summary['internal_full_assignment_clamps_eliminable_by_direct_weight'] == 1
        assert summary['internal_partial_batch_probability_queries'] == 0
    elif method == 'smc_4':
        assert summary['engines_with_multiple_public_samples_at_same_q'] >= 1
        assert summary['public_sample_calls_per_engine']['0'] == 4


def test_trace_rng_state_reproduces_recorded_batch_on_same_original_engine():
    from tri.evaluation.cohorts import deserialize_spec
    from tri.inference.music_backends import make_music_engine
    spec, provider = setup_case(width=2)
    recorder = QueryWorkloadRecorder(provider)
    with recorder.instrument_factories():
        result = research_decode('one_shot_joint', spec, recorder, steps=4, seed=77, backend='paired')
    tape = recorder.tape(result, 1.)
    engine_record = tape['engines'][0]
    engine = make_music_engine(deserialize_spec(engine_record['spec']), recorder.arrays[engine_record['q_array_key']], backend='paired')
    for call in tape['application_calls']:
        if call['operation'] == 'log_partition':
            assert engine.log_partition(call['evidence']) == pytest.approx(call['log_partition'])
        else:
            rng = np.random.Generator(getattr(np.random, call['rng_bit_generator'])())
            rng.bit_generator.state = call['rng_state_before']
            draw = engine.sample_batch(call['variables'], rng, call['evidence'])
            assert draw.assignment == call['sampled_assignment']
            assert draw.log_probability == pytest.approx(call['log_probability'])


def test_fixed_q_counts_and_probability_changes_are_measured_without_hashes():
    spec, provider = setup_case()
    recorder = QueryWorkloadRecorder(provider)
    state = tuple(spec.observed.get(i) for i in range(spec.length))
    recorder(state, 1.)
    recorder(state, 1.)
    recorder(state, .5)
    assert [record['unique_full_q_id'] for record in recorder.models] == [0, 0, 1]
    assert recorder.models[1]['same_state_noise_as_model_call'] == 0
    assert recorder.models[1]['full_q_equal_previous_call']
    assert not recorder.models[2]['full_q_equal_previous_call']
    assert recorder.models[2]['common_unknown_mean_total_variation_vs_previous_call'] > 0
    assert recorder.models[2]['common_unknown_max_probability_change_vs_previous_call'] > 0


def test_public_condition_queries_and_internal_sampling_queries_stay_distinct():
    from tri.inference import music_backends
    spec, provider = setup_case()
    recorder = QueryWorkloadRecorder(provider)
    state = tuple(spec.observed.get(i) for i in range(spec.length))
    with recorder.instrument_factories():
        q = recorder(state, 1.)
        engine = music_backends.make_music_engine(spec, q, backend='paired')
        engine.log_partition()
        engine.log_partition({'y4': 0})
        engine.log_partition({'y4': 0})
        engine.sample_batch(['y2'], np.random.default_rng(133))
    summary = recorder.summarize(1.)
    assert summary['public_nonempty_partition_conditions'] == 2
    assert summary['repeated_public_partition_conditions'] == 1
    assert summary['internal_partial_batch_probability_queries'] == 1
    condition_events = [event for event in recorder.events if event['kind'] == 'solver_api'
                        and event['application_level'] and event['evidence'] == {'y4': 0}]
    assert condition_events[0]['prefix_steps_if_base_messages_retained'] == 3
    assert condition_events[0]['changed_aligned_steps'] == [3]
    assert condition_events[1]['repeated_partition_condition_in_engine']


def test_factory_patch_is_restored_after_failure():
    from tri.inference import music_backends
    original = music_backends.make_music_engine
    spec, provider = setup_case()
    recorder = QueryWorkloadRecorder(provider)
    with pytest.raises(RuntimeError, match='test exception'):
        with recorder.instrument_factories():
            raise RuntimeError('test exception')
    assert music_backends.make_music_engine is original


@pytest.mark.parametrize('method', ['one_shot_joint', 'tri_direct', 'smc_4'])
@pytest.mark.parametrize('backend', ['product_reuse', 'product_prefix', 'paired_reuse', 'paired_sparse'])
def test_replay_fixes_application_trajectory_and_checks_candidate_probabilities(tmp_path, method, backend):
    from tri.evaluation.query_workload import replay_trace
    spec, provider = setup_case(width=2)
    recorder = QueryWorkloadRecorder(provider)
    with recorder.instrument_factories():
        started = time.perf_counter()
        result = research_decode(method, spec, recorder, steps=4, seed=61, backend='paired')
        elapsed = time.perf_counter() - started
    tape = recorder.tape(result, elapsed)
    path = tmp_path / 'trace.json'
    path.write_text(json.dumps(tape))
    np.savez_compressed(tmp_path / 'q_snapshots.npz', **recorder.arrays)
    replay = replay_trace(path, backend)
    assert replay['status'] == 'completed'
    assert replay['engine_instances'] == len(tape['engines'])
    assert replay['application_calls'] == len(tape['application_calls'])
    assert replay['validation']['maximum_log_error'] < 1e-10
    assert replay['validation']['partition_checks'] == result.model_calls
    assert replay['validation']['complete_sample_checks'] >= 1
    assert replay['solver_seconds'] > 0
    assert replay['observed_neural_plus_replayed_solver_seconds'] == pytest.approx(
        tape['summary']['provider_seconds'] + replay['solver_seconds'])
    if method == 'smc_4':
        assert replay['maximum_required_live_engine_instances'] <= 4
        assert replay['maximum_required_live_engine_instances'] < replay['engine_instances']
    else:
        assert replay['maximum_required_live_engine_instances'] == 1
    for constructor in replay['constructor_measurements']:
        assert constructor['budget'] == {'max_factor_entries': 2_000_000, 'max_workspace_bytes': 536_870_912,
                                         'max_oracle_assignments': 1_000_000}


def test_replay_rejects_changed_saved_partition_and_missing_budget(tmp_path):
    from tri.evaluation.query_workload import replay_trace
    spec, provider = setup_case(width=2)
    recorder = QueryWorkloadRecorder(provider)
    with recorder.instrument_factories():
        result = research_decode('one_shot_joint', spec, recorder, steps=4, seed=7, backend='paired')
    tape = recorder.tape(result, 1.)
    np.savez_compressed(tmp_path / 'q_snapshots.npz', **recorder.arrays)
    tape['application_calls'][0]['log_partition'] += 1
    path = tmp_path / 'trace.json'
    path.write_text(json.dumps(tape))
    with pytest.raises(AssertionError, match='saved partition'):
        replay_trace(path, 'product_prefix')
    del tape['engines'][0]['budget']
    path.write_text(json.dumps(tape))
    with pytest.raises(ValueError, match='Budget'):
        replay_trace(path, 'product_prefix')


def test_cached_internal_clamp_is_not_counted_as_another_forward():
    from tri.inference import music_backends
    spec, provider = setup_case(width=2)
    recorder = QueryWorkloadRecorder(provider)
    state = tuple(spec.observed.get(i) for i in range(spec.length))
    with recorder.instrument_factories():
        engine = music_backends.make_music_engine(spec, recorder(state, 1.), backend='paired')
        engine.log_partition()
        # Identical RNG and engine produce identical projected assignment. The
        # second internal probability query uses the scalar partition cache.
        engine.sample_batch(['y1'], np.random.default_rng(94))
        engine.sample_batch(['y1'], np.random.default_rng(94))
    summary = recorder.summarize(1.)
    assert summary['internal_partial_batch_probability_queries'] == 2
    assert summary['internal_partial_probability_queries_with_actual_forward'] == 1
    assert summary['internal_partial_probability_queries_already_scalar_cached'] == 1


def _save_small_tape(path):
    spec, provider = setup_case(width=2)
    recorder = QueryWorkloadRecorder(provider)
    with recorder.instrument_factories():
        result = research_decode('tri_direct', spec, recorder, steps=2, seed=47, backend='paired')
    tape = recorder.tape(result, 1.)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tape))
    np.savez_compressed(path.with_name('q_snapshots.npz'), **recorder.arrays)
    return path


def test_replay_reports_candidate_memory_before_checker_starts(tmp_path):
    from tri.evaluation.query_workload import replay_trace
    events = []
    result = replay_trace(_save_small_tape(tmp_path / 'trace.json'), 'product_prefix', phase_callback=events.append)
    assert [event['phase'] for event in events] == ['candidate_started', 'candidate_completed', 'validation_started']
    candidate = events[1]['candidate']
    assert 'validation' not in candidate
    assert candidate['candidate_peak_rss_mib'] >= candidate['baseline_peak_rss_mib'] > 0
    assert result['candidate_peak_rss_mib'] == candidate['candidate_peak_rss_mib']
    assert result['worker_peak_rss_after_validation_mib'] >= candidate['candidate_peak_rss_mib']
    assert result['validation']['performed']


def test_isolated_replay_has_fresh_pid_cpu_and_separate_validation(tmp_path):
    import os
    from tri.evaluation.query_replay import isolated_replay
    cpu = min(os.sched_getaffinity(0))
    path = _save_small_tape(tmp_path / 'trace.json')
    result = isolated_replay({'trace': str(path), 'backend': 'product_prefix', 'cpu': cpu}, timeout_seconds=15)
    assert result['status'] == 'completed', result
    assert result['worker_pid'] != os.getpid()
    assert result['worker_cpu'] == cpu
    assert result['validation']['performed']
    assert result['validation']['maximum_log_error'] < 1e-10
    assert result['candidate_peak_rss_mib'] < 512
    assert result['elapsed_worker_seconds'] > result['solver_seconds']


def test_worker_validation_failure_keeps_candidate_but_never_marks_verified(monkeypatch):
    import os
    import tri.evaluation.query_replay as module
    def failing_replay(*args, phase_callback, **kwargs):
        phase_callback({'phase': 'candidate_started', 'baseline_peak_rss_mib': 42.})
        phase_callback({'phase': 'candidate_completed', 'candidate': {'status': 'candidate_completed',
                        'candidate_peak_rss_mib': 52., 'solver_seconds': .1}})
        phase_callback({'phase': 'validation_started'})
        raise AssertionError('independent sample probability differs')
    monkeypatch.setattr(module, 'replay_trace', failing_replay)
    events = []
    result = module.replay_worker({'trace': 'unused.json', 'backend': 'product_prefix',
                                   'cpu': min(os.sched_getaffinity(0))}, emit=events.append)
    assert result['status'] == 'correctness_failure'
    assert result['failed_phase'] == 'validation_started'
    assert result['solver_seconds'] == .1
    assert result['candidate_peak_rss_mib'] == 52.
    assert 'validation' not in result
    assert events[-1]['event'] == 'done'


@pytest.mark.parametrize('limit,status', [('time', 'timeout'), ('memory', 'rss_limit')])
def test_isolated_replay_preserves_resource_failures(tmp_path, limit, status):
    import os
    from tri.evaluation.query_replay import isolated_replay
    path = _save_small_tape(tmp_path / 'trace.json')
    result = isolated_replay({'trace': str(path), 'backend': 'product_prefix',
                             'cpu': min(os.sched_getaffinity(0))},
                             timeout_seconds=.001 if limit == 'time' else 15,
                             rss_limit_mib=.001 if limit == 'memory' else 512)
    assert result['status'] == status
    assert 'failed_phase' in result
    assert result['elapsed_worker_seconds'] < 15


def test_suite_schedule_and_resume_preserve_failures_without_retry(tmp_path, monkeypatch):
    import os
    import tri.evaluation.query_replay as module
    root = tmp_path / 'traces'
    for stage in module.STAGES:
        for method in module.METHODS:
            folder = root / stage / method
            folder.mkdir(parents=True)
            (folder / 'trace.json').write_text('{}')
            (folder / 'q_snapshots.npz').write_bytes(b'not loaded by mocked worker')
    scheduled = module.replay_schedule(root)
    assert len(scheduled) == len({job['job_id'] for job in scheduled}) == 108
    assert scheduled == module.replay_schedule(root)
    for offset in range(0, 108, 4):
        assert {job['backend'] for job in scheduled[offset:offset + 4]} == set(module.BACKENDS)
        assert len({(job['stage'], job['method'], job['repeat']) for job in scheduled[offset:offset + 4]}) == 1
    calls = []
    def fake_worker(payload, **kwargs):
        calls.append(payload)
        return {'status': 'completed' if len(calls) % 2 else 'timeout'}
    monkeypatch.setattr(module, 'isolated_replay', fake_worker)
    output = tmp_path / 'results'
    settings = {'repeats': 1, 'backends': module.BACKENDS[:2], 'cpu': min(os.sched_getaffinity(0))}
    first = module.run_replay_suite(root, output, **settings)
    assert len(calls) == first['completed'] == 18
    assert first['status_counts'] == {'completed': 9, 'timeout': 9}
    second = module.run_replay_suite(root, output, **settings, resume=True)
    assert len(calls) == second['completed'] == 18
    assert second['status_counts'] == first['status_counts']
    with pytest.raises(ValueError, match='matching explicit'):
        module.run_replay_suite(root, output, **settings)
    with pytest.raises(ValueError, match='matching explicit'):
        module.run_replay_suite(root, output, **{**settings, 'repeats': 2}, resume=True)
