"""Instrument actual frozen-checkpoint decoders without changing their decisions.

The resulting tape separates application API calls from a solver's internal
partition/forward calls. It records every constructed engine's MusicSpec,
original full-130 q, evidence, requested variables, RNG state and sampled batch.
A later solver replay can keep the actual decoder trajectory fixed; replay is
not a second interactive-user log and is not itself a new end-to-end decode.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from tri.domain.music import verify_music
from tri.inference.exact import Budget
from tri.runtime import atomic_json, atomic_jsonl, exclusive_run, read_jsonl


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return '-inf' if value < 0 else 'inf' if value > 0 else 'nan'
    return value


def spec_record(spec):
    # Keep this independent of dataset/torch imports for small CPU probe tests.
    return {'length': spec.length, 'pitches': list(spec.pitches),
            'observed': {str(k): v for k, v in spec.observed.items()},
            'initial_pitch': spec.initial_pitch, 'enforce_end': spec.enforce_end,
            'end_pitch': spec.end_pitch,
            'equal_onsets': [list(pair) for pair in spec.equal_onsets],
            'onset_counts': [{'positions': list(rule.positions), 'count': rule.count} for rule in spec.onset_counts],
            'pitch_ranges': {str(k): list(v) for k, v in spec.pitch_ranges.items()},
            'pitch_classes': {str(k): list(v) for k, v in spec.pitch_classes.items()},
            'max_adjacent_interval': spec.max_adjacent_interval, 'motion_cost': spec.motion_cost,
            'fixed_soundings': {str(k): v for k, v in spec.fixed_soundings.items()}}


class QueryWorkloadRecorder:
    """Single-threaded observational wrappers around provider/factory calls.

    Private forward instrumentation is available for the original paired
    backend. Prefix counts describe a counterfactual opportunity if the base
    messages had been retained; the recorder never performs that optimization.
    """

    def __init__(self, provider, *, synchronize=None):
        self.provider = provider
        self.synchronize = synchronize or (lambda: None)
        self.models = []
        self.engines = []
        self.events = []
        self.arrays = {}
        self._q_unique = []
        self._provider_inputs = {}
        self._model_states = []
        self._stack = []
        self._started = time.perf_counter()
        self.hook_seconds = 0.0

    def _event(self, kind, **fields):
        event = {'event_id': len(self.events), 'kind': kind,
                 'offset_seconds': time.perf_counter() - self._started, **fields}
        self.events.append(event)
        return event

    def __call__(self, state, noise):
        self.synchronize()
        started = time.perf_counter()
        q = self.provider(state, noise)
        self.synchronize()
        elapsed = time.perf_counter() - started
        hook_started = time.perf_counter()
        snapshot = np.array(q, dtype=np.float64, copy=True)
        call_id = len(self.models)
        state = tuple(state)
        unique_id = next((i for i, previous in enumerate(self._q_unique) if np.array_equal(snapshot, previous)), None)
        if unique_id is None:
            unique_id = len(self._q_unique)
            self._q_unique.append(snapshot)
        array_key = f'model_q_{call_id:04d}'
        self.arrays[array_key] = snapshot
        input_key = state, float(noise)
        same_input = self._provider_inputs.get(input_key)
        self._provider_inputs[input_key] = call_id
        record = {'model_call_id': call_id, 'unique_full_q_id': unique_id,
                  'q_array_key': array_key, 'state': list(state), 'noise': float(noise),
                  'unknown_positions': [i for i, value in enumerate(state) if value is None],
                  'same_state_noise_as_model_call': same_input,
                  'provider_seconds': elapsed,
                  'provider_timing_scope': 'input tensors + neural forward + full130 float64 logsoftmax + device-to-host'}
        if self.models:
            previous = self.models[-1]
            previous_q = self.arrays[previous['q_array_key']]
            common = [i for i, (a, b) in enumerate(zip(self._model_states[-1], state)) if a is None and b is None]
            record['previous_model_call_id'] = previous['model_call_id']
            record['full_q_equal_previous_call'] = bool(np.array_equal(snapshot, previous_q))
            record['common_unknown_positions_vs_previous_call'] = common
            if common:
                unchanged_rows = np.all(snapshot[common] == previous_q[common], axis=1)
                record['common_unknown_exact_unchanged_row_count'] = int(unchanged_rows.sum())
                record['common_unknown_exact_changed_row_count'] = int((~unchanged_rows).sum())
                difference = np.abs(np.exp(snapshot[common]) - np.exp(previous_q[common]))
                record['common_unknown_mean_total_variation_vs_previous_call'] = float(.5 * difference.sum(axis=1).mean())
                record['common_unknown_max_probability_change_vs_previous_call'] = float(difference.max())
            record['chronological_comparison_caveat'] = 'SMC adjacent calls may belong to different particle branches, not successive states of one trajectory.'
        self.models.append(record)
        self._model_states.append(state)
        self._event('model', model_call_id=call_id, seconds=elapsed)
        self.hook_seconds += time.perf_counter() - hook_started
        return q

    def _prefix(self, engine, baseline_choices, evidence):
        choices, _ = engine._choices(evidence)
        changed = [step for step, positions in enumerate(zip(*engine.spans))
                   if any(choices[position] != baseline_choices[position] for position in positions)]
        first = min(changed, default=engine.width)
        return {'aligned_width': engine.width, 'changed_aligned_steps': changed,
                'prefix_steps_if_base_messages_retained': first,
                'prefix_fraction_if_base_messages_retained': first / engine.width,
                'zero_local_support': any(not options for options in choices)}

    def _wrap_engine(self, engine, engine_id):
        baseline_choices = engine._choices({})[0] if hasattr(engine, '_choices') and hasattr(engine, 'spans') else None
        unknown = {f'y{i}' for i in range(engine.spec.length) if i not in engine.spec.observed}
        seen_conditions = set()
        base_forward_seen = False

        def wrap_query(name, original):
            def wrapped(*args, **kwargs):
                hook_started = time.perf_counter()
                if name == 'sample_batch':
                    variables = tuple(args[0] if args else kwargs['variables'])
                    rng = args[1] if len(args) > 1 else kwargs['rng']
                    evidence = args[2] if len(args) > 2 else kwargs.get('evidence')
                    details = {'variables': list(variables), 'rng_bit_generator': type(rng.bit_generator).__name__,
                               'rng_state_before': copy.deepcopy(rng.bit_generator.state)}
                else:
                    evidence = args[0] if args else kwargs.get('evidence')
                    details = {}
                evidence = dict(evidence or {})
                public = not self._stack
                parent = self._stack[-1] if self._stack else None
                condition_key = tuple(sorted(evidence.items()))
                if name == 'log_partition':
                    details['repeated_partition_condition_in_engine'] = condition_key in seen_conditions
                    seen_conditions.add(condition_key)
                    if evidence and not public:
                        details['internal_probability_role'] = ('complete_assignment_weight_can_be_evaluated_directly'
                            if unknown <= evidence.keys() else 'partial_batch_probability_query')
                if baseline_choices is not None:
                    details.update(self._prefix(engine, baseline_choices, evidence))
                event = self._event('solver_api', engine_id=engine_id, operation=name,
                                    application_level=public, parent_event_id=parent,
                                    evidence=evidence, **details)
                self._stack.append(event['event_id'])
                self.hook_seconds += time.perf_counter() - hook_started
                hook_before = self.hook_seconds
                started = time.perf_counter()
                try:
                    result = original(*args, **kwargs)
                except BaseException as error:
                    event['error'] = {'type': type(error).__name__, 'message': str(error)}
                    raise
                finally:
                    elapsed = time.perf_counter() - started
                    event['seconds_including_nested_probe_hooks'] = elapsed
                    event['seconds_excluding_nested_probe_hooks'] = max(0.0, elapsed - (self.hook_seconds - hook_before))
                    self._stack.pop()
                hook_started = time.perf_counter()
                if name == 'sample_batch':
                    event['sampled_assignment'] = dict(result.assignment)
                    event['log_probability'] = float(result.log_probability)
                    event['log_clamped_partition'] = float(result.log_clamped_partition)
                    event['full_current_unknown_batch'] = unknown <= set(variables) | evidence.keys()
                else:
                    event['log_partition'] = float(result)
                event['backend_stats'] = json_value(engine.last_stats)
                self.hook_seconds += time.perf_counter() - hook_started
                return result
            return wrapped

        if hasattr(engine, '_forward'):
            original_forward = engine._forward

            def forward(evidence, save=False):
                nonlocal base_forward_seen
                hook_started = time.perf_counter()
                evidence = dict(evidence or {})
                details = self._prefix(engine, baseline_choices, evidence) if baseline_choices is not None else {}
                event = self._event('solver_forward', engine_id=engine_id, evidence=evidence, save_layers=bool(save),
                                    parent_event_id=self._stack[-1] if self._stack else None,
                                    unconditional_forward_previously_computed=base_forward_seen, **details)
                self.hook_seconds += time.perf_counter() - hook_started
                started = time.perf_counter()
                result = original_forward(evidence, save=save)
                event['seconds'] = time.perf_counter() - started
                hook_started = time.perf_counter()
                event['log_partition'] = float(result[0])
                if not evidence:
                    base_forward_seen = True
                self.hook_seconds += time.perf_counter() - hook_started
                return result
            engine._forward = forward
        engine.log_partition = wrap_query('log_partition', engine.log_partition)
        engine.sample_batch = wrap_query('sample_batch', engine.sample_batch)
        return engine

    @contextmanager
    def instrument_factories(self):
        from tri.inference import music_backends
        original = music_backends.make_music_engine

        def factory(spec, log_probs, backend='ve', budget=None):
            started = time.perf_counter()
            engine = original(spec, log_probs, backend=backend, budget=budget)
            elapsed = time.perf_counter() - started
            hook_started = time.perf_counter()
            if not self.models:
                raise RuntimeError('A traced factory must follow an actual probability-provider call')
            model_call = next((record for record in reversed(self.models)
                               if np.array_equal(log_probs, self.arrays[record['q_array_key']])), None)
            if model_call is None:
                raise RuntimeError('Factory q does not equal any recorded provider output')
            engine_id = len(self.engines)
            record = {'engine_id': engine_id, 'backend_requested': backend, 'backend_actual': engine.backend_name,
                      'model_call_id': model_call['model_call_id'], 'unique_full_q_id': model_call['unique_full_q_id'],
                      'q_array_key': model_call['q_array_key'], 'spec': spec_record(spec),
                      'construct_seconds': elapsed,
                      'budget': {name: int(getattr(engine.budget, name)) for name in ('max_factor_entries', 'max_workspace_bytes', 'max_oracle_assignments')},
                      'budget_annotation': 'Observed from actual engine at factory construction'}
            self.engines.append(record)
            self._event('solver_construct', engine_id=engine_id, model_call_id=model_call['model_call_id'], seconds=elapsed)
            wrapped = self._wrap_engine(engine, engine_id)
            self.hook_seconds += time.perf_counter() - hook_started
            return wrapped

        with patch.object(music_backends, 'make_music_engine', side_effect=factory):
            yield self

    def summarize(self, elapsed):
        public = [event for event in self.events if event['kind'] == 'solver_api' and event['application_level']]
        internal = [event for event in self.events if event['kind'] == 'solver_api' and not event['application_level']]
        forwards = [event for event in self.events if event['kind'] == 'solver_forward']
        partial = [event for event in internal if event.get('internal_probability_role') == 'partial_batch_probability_query']
        full_weight = [event for event in internal if event.get('internal_probability_role') == 'complete_assignment_weight_can_be_evaluated_directly']
        condition_visits = defaultdict(Counter)
        for event in public:
            if event['operation'] == 'log_partition':
                condition_visits[event['engine_id']][tuple(sorted(event['evidence'].items()))] += 1
        public_by_engine = Counter(event['engine_id'] for event in public)
        samples_by_engine = Counter(event['engine_id'] for event in public if event['operation'] == 'sample_batch')
        provider_seconds = sum(model['provider_seconds'] for model in self.models)
        solver_seconds = sum(engine['construct_seconds'] for engine in self.engines) + sum(event['seconds_excluding_nested_probe_hooks'] for event in public)
        ratios = [event['prefix_fraction_if_base_messages_retained'] for event in partial]
        forward_parents = {event['parent_event_id'] for event in forwards}
        actual_partial = [event for event in partial if event['event_id'] in forward_parents]
        actual_full = [event for event in full_weight if event['event_id'] in forward_parents]
        actual_ratios = [event['prefix_fraction_if_base_messages_retained'] for event in actual_partial]
        return {
            'model_calls': len(self.models), 'distinct_full_q_arrays': len(self._q_unique),
            'model_calls_repeating_identical_state_and_noise': sum(model['same_state_noise_as_model_call'] is not None for model in self.models),
            'chronologically_adjacent_model_q_changed': sum(not model['full_q_equal_previous_call'] for model in self.models[1:]),
            'chronologically_adjacent_model_q_comparisons': max(0, len(self.models) - 1),
            'engine_instances': len(self.engines), 'public_solver_api_calls': len(public),
            'public_solver_operations': dict(Counter(event['operation'] for event in public)),
            'public_calls_per_engine': {str(key): value for key, value in public_by_engine.items()},
            'public_sample_calls_per_engine': {str(key): value for key, value in samples_by_engine.items()},
            'engines_with_multiple_public_samples_at_same_q': sum(value > 1 for value in samples_by_engine.values()),
            'repeated_public_partition_conditions': sum(sum(max(0, count - 1) for count in visits.values()) for visits in condition_visits.values()),
            'public_nonempty_partition_conditions': sum(event['operation'] == 'log_partition' and bool(event['evidence']) for event in public),
            'internal_solver_api_calls': len(internal), 'actual_forward_calls': len(forwards),
            'internal_full_assignment_clamps_eliminable_by_direct_weight': len(full_weight),
            'internal_partial_batch_probability_queries': len(partial),
            'partial_probability_queries_with_positive_reusable_prefix': sum(value > 0 for value in ratios),
            'partial_probability_query_mean_reusable_prefix_fraction': float(np.mean(ratios)) if ratios else None,
            'internal_full_assignment_clamps_with_actual_forward': len(actual_full),
            'internal_partial_probability_queries_with_actual_forward': len(actual_partial),
            'internal_partial_probability_queries_already_scalar_cached': len(partial) - len(actual_partial),
            'actual_partial_forwards_with_positive_reusable_prefix': sum(value > 0 for value in actual_ratios),
            'actual_partial_forward_mean_reusable_prefix_fraction': float(np.mean(actual_ratios)) if actual_ratios else None,
            'provider_seconds': provider_seconds, 'solver_constructor_and_public_api_seconds': solver_seconds,
            'instrumented_decode_wall_seconds': elapsed, 'recorded_probe_hook_seconds': self.hook_seconds,
            'wall_minus_provider_solver_seconds': elapsed - provider_seconds - solver_seconds,
            'timing_note': 'Provider calls are device-synchronized. Solver durations sum construction and application API exclusive of measured nested recording hooks; wall time also contains recorder analysis and decoder/schedule logic. No claim of uninstrumented production latency.',
            'interpretation': 'Public API queries are real decoder calls. Internal batch-probability clamps are implementation/interface work, not user edits. Positive prefix fractions describe reuse opportunity, not reuse executed by the original paired backend.'}

    def tape(self, result, elapsed, *, metadata=None):
        return json_value({'version': 1, 'source': 'actual_decoder_execution', 'metadata': metadata or {},
            'summary': self.summarize(elapsed), 'model_calls': self.models, 'engines': self.engines,
            'events': self.events, 'application_calls': [event for event in self.events if event['kind'] == 'solver_api' and event['application_level']],
            'decoder_result': {'tokens': result.tokens, 'model_calls': result.model_calls,
                               'trace': result.trace, 'diagnostics': result.diagnostics},
            'replay_contract': 'Use saved factory specs/full q and ordered application_calls. Reset each sample RNG from rng_state_before; record sampled_assignment is the actual original outcome. Candidate draws must not alter later saved neural states, specs, q, evidence or reveal variables. Internal API/forward events are diagnostics, never additional application replay requests.'})


def audit_revision(revision_root):
    """Summarize existing decoder traces; q equality is unavailable in old logs."""
    rows = []
    for stage in ('short', 'bar8', 'bar16'):
        path = Path(revision_root) / stage / 'evaluation' / 'results.jsonl'
        original = read_jsonl(path)
        for method in ('one_shot_joint', 'tri_direct', 'smc_4'):
            selected = [row for row in original if row['method'] == method]
            calls = [row['model_calls'] for row in selected]
            rows.append({'stage': stage, 'method': method, 'records': len(selected),
                         'mean_model_calls': float(np.mean(calls)), 'minimum_model_calls': min(calls),
                         'maximum_model_calls': max(calls), 'stored_decoder_traces': sum(bool(row.get('trace')) for row in selected),
                         'q_values_logged': False, 'q_equality_inferable_from_old_records': False})
    return {'source': 'existing_revision_records_read_only', 'rows': rows,
            'limitation': 'Old results store model-call counts and decoder step/clamp summaries, not neural q arrays or every solver API call. They cannot alone establish fixed-q query-reuse frequency.'}


def run_probe(revision_root, output, *, device='cuda:0', cpu=4,
              stages=('short', 'bar8', 'bar16'), methods=('one_shot_joint', 'tri_direct', 'smc_4')):
    import os
    import torch
    from threadpoolctl import threadpool_limits
    from tri.data.chords import load_chord_sidecar
    from tri.evaluation.batch import load_evaluation_windows
    from tri.evaluation.cohorts import deserialize_spec, file_identity
    from tri.models.grid import ModelProbabilityProvider
    from tri.models.train import load_checkpoint
    from tri.sampling.research_methods import research_decode

    if cpu in (2, 3):
        raise ValueError('Probe CPU must not share the reserved comparison CPU 2/3 core')
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    revision_root, output = Path(revision_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.startswith('cuda') else (lambda: None)
    config = {'version': 1, 'revision_root': str(revision_root), 'stages': list(stages), 'methods': list(methods),
              'device': device, 'cpu': cpu, 'backend': 'paired', 'repeats': 1,
              'selection': 'first frozen unknown request per stage, original evaluation replicate0 seed',
              'neural_warmup_calls_per_stage': 1, 'no_training': True,
              'source_kind': 'actual_decoder_execution_not_interactive_user_logs',
              'budget': {name: int(getattr(Budget(), name)) for name in ('max_factor_entries', 'max_workspace_bytes', 'max_oracle_assignments')}}
    with exclusive_run(output / 'probe.lock'), threadpool_limits(limits=1):
        if (output / 'summary.json').exists():
            raise ValueError('Probe output already completed; use a new directory rather than overwriting evidence')
        atomic_json(output / 'config.json', config)
        atomic_json(output / 'revision_audit.json', audit_revision(revision_root))
        summaries = []
        run_started = time.perf_counter()
        atomic_json(output / 'status.json', {'status': 'running', 'planned': len(stages) * len(methods), 'completed': 0})
        for stage in stages:
            evaluation = json.loads((revision_root / stage / 'evaluation' / 'config.json').read_text())
            cohort_path = Path(evaluation['cohort']['path'])
            if file_identity(cohort_path) != evaluation['cohort']:
                raise ValueError('Frozen cohort identity changed')
            cohort = json.loads(cohort_path.read_text())
            case_index, case = next((index, case) for index, case in enumerate(cohort['requests']) if case['kind'] == 'unknown')
            dataset = Path(cohort['options']['dataset']['path'])
            chords = Path(cohort['options']['chords']['path'])
            for name, path in [('dataset', dataset), ('chords', chords)]:
                if file_identity(path) != cohort['options'][name]:
                    raise ValueError('Frozen source data changed')
            checkpoint = Path(evaluation['checkpoint']['path'])
            if file_identity(checkpoint) != evaluation['checkpoint']:
                raise ValueError('Frozen checkpoint changed')
            window = next(window for window in load_evaluation_windows(dataset, limit=2**31-1, split='validation')
                          if window.source_index == case['source_index'])
            condition = load_chord_sidecar(dataset, chords)['chord_features'][window.source_index]
            spec = deserialize_spec(case['spec'])
            model = load_checkpoint(checkpoint, device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            provider = ModelProbabilityProvider(model, tuple(i for i in range(spec.length) if i not in spec.observed),
                                                 condition if model.config.condition_dim else None)
            initial = tuple(spec.observed.get(i) for i in range(spec.length))
            synchronize()
            warmup_started = time.perf_counter()
            provider(initial, 1.)
            synchronize()
            warmup_seconds = time.perf_counter() - warmup_started
            original = read_jsonl(revision_root / stage / 'evaluation' / 'results.jsonl')
            seed = evaluation['seed'] + case_index * 10000
            for method in methods:
                original_row = next(row for row in original if row['cohort_case_id'] == case['case_id']
                                    and row['method'] == method and row['replicate'] == 0)
                recorder = QueryWorkloadRecorder(provider, synchronize=synchronize)
                metadata = {'stage': stage, 'method': method, 'case_id': case['case_id'], 'case_index': case_index,
                            'kind': case['kind'], 'work_id': case['work_id'], 'source_index': case['source_index'],
                            'source_start_cell': case['source_start_cell'], 'checkpoint': evaluation['checkpoint'],
                            'cohort': evaluation['cohort'], 'seed': seed, 'steps': evaluation['steps'],
                            'original_backend': evaluation['backend'], 'actual_backend': 'paired',
                            'warmup_seconds_excluded': warmup_seconds,
                            'original_evaluation_record': {'request_id': original_row['request_id'],
                                'model_calls': original_row['model_calls'], 'elapsed_decode_seconds': original_row['elapsed_decode_seconds']}}
                with recorder.instrument_factories():
                    synchronize()
                    started = time.perf_counter()
                    result = research_decode(method, spec, recorder, steps=evaluation['steps'], seed=seed,
                                             backend='paired', budget=Budget())
                    synchronize()
                    elapsed = time.perf_counter() - started
                if result.model_calls != len(recorder.models) or not verify_music(result.tokens, spec).valid:
                    raise RuntimeError('Instrumented decoder failed call accounting or independent music verification')
                metadata['original_tokens_exact_match'] = list(result.tokens) == original_row['raw_tokens']
                metadata['original_model_calls_exact_match'] = result.model_calls == original_row['model_calls']
                tape = recorder.tape(result, elapsed, metadata=metadata)
                folder = output / stage / method
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / 'q_snapshots.npz').open('wb') as stream:
                    np.savez_compressed(stream, **recorder.arrays)
                atomic_json(folder / 'trace.json', tape)
                summary = {'stage': stage, 'method': method, 'case_id': case['case_id'],
                           'trace': str(folder / 'trace.json'), 'q_snapshots': str(folder / 'q_snapshots.npz'),
                           'original_tokens_exact_match': metadata['original_tokens_exact_match'],
                           'original_model_calls_exact_match': metadata['original_model_calls_exact_match'],
                           **tape['summary']}
                summaries.append(summary)
                atomic_jsonl(output / 'results.jsonl', summaries)
                atomic_json(output / 'status.json', {'status': 'running', 'planned': len(stages) * len(methods),
                    'completed': len(summaries), 'current_stage': stage, 'current_method': method,
                    'elapsed_seconds': time.perf_counter() - run_started})
                print(json.dumps({'stage': stage, 'method': method, 'model_calls': result.model_calls,
                                  'elapsed_seconds': elapsed, 'original_tokens_match': metadata['original_tokens_exact_match']}), flush=True)
            del model, provider
        report = {'status': 'completed', 'config': config, 'decodes': summaries,
                  'elapsed_seconds': time.perf_counter() - run_started,
                  'scope': 'Three frozen validation requests, three actual decoders, one replicate each. No interactive editing sessions or new human observations.',
                  'replay_scope': 'Saved q/spec/application-call tapes can support a fixed-trajectory solver replay. Such replay must be reported separately from actual end-to-end reruns.'}
        atomic_json(output / 'summary.json', report)
        atomic_json(output / 'status.json', {'status': 'completed', 'planned': len(stages) * len(methods),
                                           'completed': len(summaries), 'elapsed_seconds': report['elapsed_seconds']})
        return report



def _deserialize_trace_spec(data):
    from tri.domain.music import CountRule, MusicSpec
    data = dict(data)
    for key in ('observed', 'fixed_soundings', 'pitch_ranges', 'pitch_classes'):
        data[key] = {int(k): v for k, v in data.get(key, {}).items()}
    data['onset_counts'] = tuple(CountRule(tuple(rule['positions']), rule['count']) for rule in data['onset_counts'])
    return MusicSpec(**data)


def annotate_collected_probe_budget(output):
    """Add audited input metadata, without rerunning or altering measurements.

    Collection v1 passed an explicit Budget() to research_decode. Its factory
    wrappers initially omitted that object from JSON; this annotation records
    that known input and marks the field as post-collection, not a new reading.
    Additional q-row statistics are derived solely from the saved q snapshots.
    """
    output = Path(output)
    config = json.loads((output / 'config.json').read_text())
    budget = {name: int(getattr(Budget(), name)) for name in
              ('max_factor_entries', 'max_workspace_bytes', 'max_oracle_assignments')}
    note = 'Post-collection annotation: original run_probe explicitly called research_decode(..., budget=Budget()); audited source/defaults establish these unchanged input limits. No numerical data, timing or decoder trajectory was rerun.'
    config['budget'] = budget
    config['budget_annotation'] = note
    atomic_json(output / 'config.json', config)
    for path in sorted(output.glob('*/*/trace.json')):
        tape = json.loads(path.read_text())
        for engine in tape['engines']:
            engine.setdefault('budget', budget)
            engine.setdefault('budget_annotation', note)
        with np.load(path.with_name('q_snapshots.npz'), allow_pickle=False) as arrays:
            for index, model in enumerate(tape['model_calls']):
                if index == 0:
                    continue
                common = model['common_unknown_positions_vs_previous_call']
                if common:
                    a = arrays[model['q_array_key']][common]
                    b = arrays[tape['model_calls'][index - 1]['q_array_key']][common]
                    unchanged = np.all(a == b, axis=1)
                    model['common_unknown_exact_unchanged_row_count'] = int(unchanged.sum())
                    model['common_unknown_exact_changed_row_count'] = int((~unchanged).sum())
        partial = [event for event in tape['events'] if event['kind'] == 'solver_api' and event.get('internal_probability_role') == 'partial_batch_probability_query']
        full = [event for event in tape['events'] if event['kind'] == 'solver_api' and event.get('internal_probability_role') == 'complete_assignment_weight_can_be_evaluated_directly']
        forward_parents = {event['parent_event_id'] for event in tape['events'] if event['kind'] == 'solver_forward'}
        actual_partial = [event for event in partial if event['event_id'] in forward_parents]
        actual_ratios = [event['prefix_fraction_if_base_messages_retained'] for event in actual_partial]
        tape['summary'].update(
            internal_full_assignment_clamps_with_actual_forward=sum(event['event_id'] in forward_parents for event in full),
            internal_partial_probability_queries_with_actual_forward=len(actual_partial),
            internal_partial_probability_queries_already_scalar_cached=len(partial) - len(actual_partial),
            actual_partial_forwards_with_positive_reusable_prefix=sum(value > 0 for value in actual_ratios),
            actual_partial_forward_mean_reusable_prefix_fraction=float(np.mean(actual_ratios)) if actual_ratios else None)
        by_model = {engine['model_call_id']: engine for engine in tape['engines']}
        with np.load(path.with_name('q_snapshots.npz'), allow_pickle=False) as arrays:
            for index, model in enumerate(tape['model_calls']):
                if index == 0:
                    continue
                common = model['common_unknown_positions_vs_previous_call']
                if common:
                    current = by_model[model['model_call_id']]['spec']
                    previous = by_model[tape['model_calls'][index - 1]['model_call_id']]['spec']
                    if current['pitches'] != previous['pitches']:
                        raise ValueError('Effective q comparison needs the same working domain')
                    vocabulary = [0, 1] + [pitch + 2 for pitch in current['pitches']]
                    a = arrays[model['q_array_key']][np.ix_(common, vocabulary)]
                    b = arrays[tape['model_calls'][index - 1]['q_array_key']][np.ix_(common, vocabulary)]
                    unchanged = np.all(a == b, axis=1)
                    model['common_unknown_working_domain_exact_unchanged_row_count'] = int(unchanged.sum())
                    model['common_unknown_working_domain_exact_changed_row_count'] = int((~unchanged).sum())
                    model['effective_q_comparison_scope'] = 'Original working tokens REST/HOLD/NOTE(p), no renormalization. Original observed rows excluded; further local hard-rule support is not projected out. SMC comparison remains chronological, possibly across branches.'
        tape.setdefault('post_collection_annotations', []).append(
            'Added explicit audited Budget input, exact full/working-domain changed-row counts from saved q, and actual-forward versus API-only clamp counts; no new measurement.')
        atomic_json(path, tape)
    if (output / 'summary.json').exists():
        summary = json.loads((output / 'summary.json').read_text())
        summary['config'] = config
        for row in summary['decodes']:
            latest = json.loads(Path(row['trace']).read_text())
            row.update(latest['summary'])
        atomic_jsonl(output / 'results.jsonl', summary['decodes'])
        summary['unique_frozen_requests'] = len({row['case_id'] for row in summary['decodes']})
        summary['unique_works'] = len({json.loads(Path(row['trace']).read_text())['metadata']['work_id'] for row in summary['decodes']})
        atomic_json(output / 'summary.json', summary)


def _process_peak_rss_mib():
    # Unlike ru_maxrss, Linux VmHWM is reset by exec and cannot inherit the
    # launcher's earlier torch allocations. This includes interpreter/input q.
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('Replay memory accounting requires Linux /proc VmHWM')


def replay_trace(trace_path, backend, *, validate=True, phase_callback=None):
    """Measure one solver on the fixed application trajectory of a real decode.

    No model is called, no new reveal schedule is sampled, and candidate outputs
    NEVER determine a later factory's spec/q. The recorded RNG state fixes each
    call's random input but different exact algorithms may produce different
    valid samples. The measured quantity is a frozen-trajectory counterfactual,
    not an actually executed candidate neural-generation trajectory.

    Engines are created immediately before their first public API use and
    released immediately after the last use in the tape. This preserves SMC
    shared-engine calls without retaining all historical caches. Validation is
    a separate pass after timed candidate work, so reference-solver allocations
    do not alter the candidate call sequence or its timings.
    """
    import gc
    from tri.inference.music_backends import make_music_engine
    from tri.inference.product_chain import ProductChainMusicInference

    trace_path = Path(trace_path).resolve()
    load_started = time.perf_counter()
    tape = json.loads(trace_path.read_text())
    with np.load(trace_path.with_name('q_snapshots.npz'), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    records = {record['engine_id']: record for record in tape['engines']}
    specs = {key: _deserialize_trace_spec(record['spec']) for key, record in records.items()}
    for record in records.values():
        if 'budget' not in record:
            raise ValueError('Replay requires an explicitly recorded or audited engine Budget')
    calls = tape['application_calls']
    if any(not call['application_level'] or call['operation'] not in ('log_partition', 'sample_batch') for call in calls):
        raise ValueError('Replay accepts application-level partition/sample calls only')
    if any(calls[i]['event_id'] >= calls[i + 1]['event_id'] for i in range(len(calls) - 1)):
        raise ValueError('Application calls must preserve recorded chronological order')
    last_use = {call['engine_id']: index for index, call in enumerate(calls)}
    input_load_seconds = time.perf_counter() - load_started
    live = {}
    constructor_rows, outcomes = [], []
    maximum_live_engines = 0
    total_constructor_seconds = 0.0
    total_query_seconds = 0.0
    baseline_peak_rss_mib = _process_peak_rss_mib()
    if phase_callback:
        phase_callback({'phase': 'candidate_started', 'baseline_peak_rss_mib': baseline_peak_rss_mib})
    phase_started = time.perf_counter()
    for index, call in enumerate(calls):
        engine_id = call['engine_id']
        record = records[engine_id]
        if engine_id not in live:
            started = time.perf_counter()
            engine = make_music_engine(specs[engine_id], arrays[record['q_array_key']], backend=backend,
                                       budget=Budget(**record['budget']))
            elapsed = time.perf_counter() - started
            live[engine_id] = engine
            maximum_live_engines = max(maximum_live_engines, len(live))
            total_constructor_seconds += elapsed
            constructor_rows.append({'engine_id': engine_id, 'seconds': elapsed, 'budget': record['budget']})
        engine = live[engine_id]
        evidence = {str(k): int(v) for k, v in call['evidence'].items()}
        outcome = {'event_id': call['event_id'], 'engine_id': engine_id, 'operation': call['operation']}
        if call['operation'] == 'log_partition':
            started = time.perf_counter()
            value = engine.log_partition(evidence)
            elapsed = time.perf_counter() - started
            outcome['log_partition'] = float(value)
        else:
            generator_name = call['rng_bit_generator']
            if generator_name not in ('PCG64', 'PCG64DXSM', 'Philox', 'SFC64', 'MT19937'):
                raise ValueError('Unsupported recorded numpy bit generator')
            rng = np.random.Generator(getattr(np.random, generator_name)())
            rng.bit_generator.state = copy.deepcopy(call['rng_state_before'])
            started = time.perf_counter()
            sampled = engine.sample_batch(tuple(call['variables']), rng, evidence)
            elapsed = time.perf_counter() - started
            outcome.update(assignment=dict(sampled.assignment), log_probability=float(sampled.log_probability),
                           log_clamped_partition=float(sampled.log_clamped_partition),
                           equals_original_sampled_assignment=dict(sampled.assignment) == call['sampled_assignment'])
        outcome['seconds'] = elapsed
        outcome['stats'] = json_value(engine.last_stats)
        outcomes.append(outcome)
        total_query_seconds += elapsed
        if last_use[engine_id] == index:
            del live[engine_id]
            del engine
    candidate_phase_wall_seconds = time.perf_counter() - phase_started
    candidate_peak_rss_mib = _process_peak_rss_mib()
    if live:
        raise RuntimeError('Replay retained engines past their last application use')
    solver_seconds = total_constructor_seconds + total_query_seconds
    original = tape['summary']
    candidate_result = json_value({'status': 'candidate_completed', 'source': 'frozen_actual_decoder_trajectory_replay',
        'trace': str(trace_path), 'metadata': tape['metadata'], 'backend': backend,
        'application_calls': len(calls), 'engine_instances': len(records),
        'maximum_required_live_engine_instances': maximum_live_engines,
        'engine_lifetime_rule': 'Construct at first recorded public use; release immediately after last public use. Shared SMC instances survive until their final use; this is required-live lifetime, not a claim to match all original Python reference lifetimes.',
        'input_load_seconds_excluded': input_load_seconds, 'constructor_seconds': total_constructor_seconds,
        'query_seconds': total_query_seconds, 'solver_seconds': solver_seconds,
        'candidate_phase_wall_seconds': candidate_phase_wall_seconds,
        'baseline_peak_rss_mib': baseline_peak_rss_mib,
        'candidate_peak_rss_mib': candidate_peak_rss_mib,
        'candidate_incremental_peak_rss_mib': max(0., candidate_peak_rss_mib - baseline_peak_rss_mib),
        'candidate_memory_scope': 'Process VmHWM read before validation; includes interpreter and all loaded frozen q/input snapshots, excludes later independent checker allocations.',
        'observed_original_provider_seconds': original['provider_seconds'],
        'observed_original_solver_seconds': original['solver_constructor_and_public_api_seconds'],
        'observed_neural_plus_replayed_solver_seconds': original['provider_seconds'] + solver_seconds,
        'counterfactual_cost_note': 'This sum combines measured original neural-provider work with measured candidate solver replay. It is not a measured candidate end-to-end generation, and excludes recorder overhead, setup/validation and decoder orchestration.',
        'constructor_measurements': constructor_rows, 'call_measurements': outcomes})
    if phase_callback:
        phase_callback({'phase': 'candidate_completed', 'candidate': candidate_result})
    # The factory backend may own closures/caches; release them before any
    # independent probability validation is constructed.
    gc.collect()

    validation = {'performed': bool(validate), 'partition_checks': 0, 'complete_sample_checks': 0,
                  'partial_sample_checks': 0, 'maximum_log_error': 0.0, 'seconds': 0.0}
    if validate:
        if phase_callback:
            phase_callback({'phase': 'validation_started'})
        validation_started = time.perf_counter()
        references = {}
        normalizers = {}

        def check_close(actual, expected, label):
            actual, expected = float(actual), float(expected)
            if math.isfinite(actual) and math.isfinite(expected):
                error = abs(actual - expected)
                if error > 1e-8:
                    raise AssertionError(f'{label}: log error {error}')
                validation['maximum_log_error'] = max(validation['maximum_log_error'], error)
            elif actual != expected:
                raise AssertionError(f'{label}: incompatible finite/zero support')

        def reference(engine_id):
            if engine_id not in references:
                record = records[engine_id]
                references[engine_id] = ProductChainMusicInference(specs[engine_id], arrays[record['q_array_key']],
                                                                  budget=Budget(**record['budget']))
            return references[engine_id]

        for index, (call, outcome) in enumerate(zip(calls, outcomes)):
            engine_id = call['engine_id']
            evidence = {str(k): int(v) for k, v in call['evidence'].items()}
            key = engine_id, tuple(sorted(evidence.items()))
            if call['operation'] == 'log_partition':
                check_close(outcome['log_partition'], call['log_partition'], 'saved partition')
                normalizers[key] = float(call['log_partition'])
                validation['partition_checks'] += 1
            else:
                spec = specs[engine_id]
                q = arrays[records[engine_id]['q_array_key']]
                assignment = outcome['assignment']
                if set(assignment) != set(call['variables']):
                    raise AssertionError('Replay returned a different variable set')
                if any(name in evidence and evidence[name] != value for name, value in assignment.items()):
                    raise AssertionError('Replay sample contradicts supplied evidence')
                combined = {**evidence, **assignment}
                tokens = [spec.observed.get(i, combined.get(f'y{i}')) for i in range(spec.length)]
                base = normalizers.get(key)
                if base is None:
                    base = reference(engine_id).log_partition(evidence)
                    normalizers[key] = base
                if all(token is not None for token in tokens):
                    if any(i in spec.observed and f'y{i}' in combined and combined[f'y{i}'] != spec.observed[i]
                           for i in spec.observed):
                        raise AssertionError('Replay sample changed original observation')
                    checked = verify_music(tokens, spec)
                    if not checked.valid:
                        raise AssertionError('Replay full sample failed original music semantics')
                    clamped = checked.soft_score + sum(q[i, tokens[i]] for i in range(spec.length) if i not in spec.observed)
                    validation['complete_sample_checks'] += 1
                else:
                    # Partial outputs require their marginal mass, not the
                    # weight of any one completed path. Positive clamped mass
                    # also checks that the returned token batch is extendable.
                    clamped = reference(engine_id).log_partition(combined)
                    if not math.isfinite(clamped):
                        raise AssertionError('Replay partial sample has no supported complete sequence')
                    validation['partial_sample_checks'] += 1
                check_close(outcome['log_clamped_partition'], clamped, 'sample clamped mass')
                check_close(outcome['log_probability'], clamped - base, 'sample conditional probability')
            if last_use[engine_id] == index:
                references.pop(engine_id, None)
        validation['seconds'] = time.perf_counter() - validation_started
    return {**candidate_result, 'status': 'completed', 'validation': validation,
            'worker_peak_rss_after_validation_mib': _process_peak_rss_mib()}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision-root', default='runs/revision')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()
    run_probe(args.revision_root, args.out, device=args.device, cpu=args.cpu)


if __name__ == '__main__':
    main()
