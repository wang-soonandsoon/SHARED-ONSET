"""Executable synthetic checks of exact backends and finite-particle behavior.

These are algorithm diagnostics, not trained-model or music-quality evidence.
The reference terminal distribution below is computed from original token
paths, independently of the compiler, exact music engines and particle code.
"""

from dataclasses import asdict
from itertools import combinations, product
import json
import math
from numbers import Integral
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import InvalidSpecification, ZeroMass
from tri.inference import Budget, ExactInference, FactorGraph, LogFactor
from tri.inference.music_backends import make_music_engine
from tri.sampling.direct import direct_decode
from tri.sampling.particles import path_is, smc_decode


def _backend_case(width, pitch_count, rng):
    pitches = tuple(range(60, 60 + pitch_count))
    length = 2 * width + 3
    left = tuple(range(1, width + 1))
    right = tuple(range(width + 2, 2 * width + 2))
    count = max(1, width // 2)
    anchor = pitches[-1]
    observed = {0: 0, width + 1: 0, length - 1: 1}
    spec = MusicSpec(length, pitches, observed=observed,
                     fixed_soundings={0: None, width + 1: None, length - 1: anchor},
                     equal_onsets=tuple(zip(left, right)),
                     onset_counts=(CountRule(left, count), CountRule(right, count)),
                     pitch_ranges={i: (pitches[0], pitches[-1]) for i in left + right},
                     pitch_classes={**{i: tuple(p % 12 for p in pitches[:-1]) for i in left},
                                    **{i: tuple(p % 12 for p in pitches[1:]) for i in right}},
                     max_adjacent_interval=7, motion_cost=.04)
    logits = rng.normal(size=(length, 130))
    q = logits - logsumexp(logits, axis=1, keepdims=True)
    # A known legal witness fixes a common batch event for ALL backends.
    witness = dict(observed)
    for span, pitch in ((left, pitches[0]), (right, anchor)):
        for j, position in enumerate(span):
            witness[position] = pitch + 2 if j < count else 1
    assert verify_music(tuple(witness[i] for i in range(length)), spec).valid
    requested = tuple((left + right)[::max(1, width // 2)])[:4]
    batch = {f"y{i}": witness[i] for i in requested}
    return spec, q, batch


def _backend_benchmarks(seed, widths=(2, 4, 6, 8), pitch_counts=(3, 12)):
    rng = np.random.default_rng(seed)
    budget = Budget()
    cases = []
    for width in widths:
        for pitch_count in pitch_counts:
            spec, q, batch = _backend_case(width, pitch_count, rng)
            rows = []
            for backend in ("ve", "template", "automaton", "auto"):
                started = time.perf_counter()
                try:
                    engine = make_music_engine(spec, q, backend, budget)
                    log_z = engine.log_partition()
                    partition_time = time.perf_counter() - started
                    partition_stats = dict(engine.last_stats)
                    began_batch = time.perf_counter()
                    log_batch = engine.log_probability(batch)
                    batch_time = time.perf_counter() - began_batch
                    if not math.isfinite(log_z) or not math.isfinite(log_batch):
                        raise RuntimeError("Known legal positive-mass witness unexpectedly has zero mass")
                    rows.append({"backend": backend, "selected_backend": engine.backend_name,
                                 "status": "ok", "log_partition": log_z,
                                 "common_batch_log_probability": log_batch,
                                 "build_and_partition_seconds": partition_time,
                                 "common_batch_query_seconds": batch_time,
                                 "partition_stats": partition_stats,
                                 "planning_stats": engine.planning_stats})
                except Exception as exc:
                    # Budget failures remain comparison outcomes, not vanished
                    # rows. Unexpected exceptions are distinguished by type.
                    rows.append({"backend": backend, "status": "failed", "error_type": type(exc).__name__,
                                 "reason": str(exc), "elapsed_seconds": time.perf_counter() - started})
            successful = [row for row in rows if row["status"] == "ok"]
            z_error = max((abs(row["log_partition"] - successful[0]["log_partition"]) for row in successful), default=None)
            p_error = max((abs(row["common_batch_log_probability"] - successful[0]["common_batch_log_probability"]) for row in successful), default=None)
            cases.append({"width_per_gap": width, "working_pitches": pitch_count,
                          "sequence_length": spec.length, "onsets_per_gap": max(1, width // 2),
                          "common_batch_assignment": batch, "backends": rows,
                          "successful_backends": len(successful),
                          "max_log_partition_disagreement": z_error,
                          "max_batch_log_probability_disagreement": p_error,
                          "agreement_passed": len(successful) >= 2 and z_error < 1e-9 and p_error < 1e-9})
    return {"scope": "identical synthetic MusicSpec and q for each backend; full HOLD/anchor/count/motion semantics",
            "budget": asdict(budget), "cases": cases,
            "timing_note": "One CPU timing per case, includes construction; diagnostic only, not a rigorous speedup study.",
            "all_comparable_cases_agree": all(case["agreement_passed"] for case in cases)}


def _tiny_spec():
    return MusicSpec(2, (60,), equal_onsets=((0, 1),))


def _tiny_provider(state, noise):
    # Before the first reveal P(NOTE)=.9; thereafter it is .1. This deliberately
    # inconsistent conditional table makes local-exact != terminal-exact visible.
    note_probability = .1 if any(token is not None for token in state) else .9
    q = np.full((2, 130), -np.inf)
    q[:, 0] = math.log1p(-note_probability)
    q[:, 62] = math.log(note_probability)
    return q


def _subsets(values):
    for size in range(len(values) + 1):
        yield from combinations(values, size)


def _enumerate_reference_target(steps=4):
    """All reference and direct-proposal ORIGINAL-token paths, no factor engine."""
    spec = _tiny_spec()
    alphabet = (0, 62)  # Exactly the positive support of the fixed provider.
    terminal_reference, terminal_direct = {}, {}
    paths = 0

    def visit(state, step, reference_mass, direct_mass):
        nonlocal paths
        missing = tuple(i for i, token in enumerate(state) if token is None)
        if not missing:
            paths += 1
            terminal_reference[state] = terminal_reference.get(state, 0.) + reference_mass
            terminal_direct[state] = terminal_direct.get(state, 0.) + direct_mass
            return
        if step >= steps:
            raise AssertionError("Reference schedule did not finish")
        q = np.exp(_tiny_provider(state, 1 - step / steps))
        full_masses = {}
        for y in product(*[(token,) if token is not None else alphabet for token in state]):
            checked = verify_music(y, spec)
            if checked.valid:
                full_masses[y] = math.exp(checked.soft_score) * math.prod(q[i, y[i]] for i in missing)
        z = sum(full_masses.values())
        rate = 1 / (steps - step)
        possible_sets = (missing,) if step == steps - 1 else tuple(_subsets(missing))
        for selected in possible_sets:
            rho = 1.0 if step == steps - 1 else rate ** len(selected) * (1 - rate) ** (len(missing) - len(selected))
            for values in product(alphabet, repeat=len(selected)):
                updated = list(state)
                for i, value in zip(selected, values):
                    updated[i] = value
                updated = tuple(updated)
                q_batch = math.prod(q[i, value] for i, value in zip(selected, values))
                clamped = sum(weight for y, weight in full_masses.items() if all(y[i] == value for i, value in zip(selected, values)))
                proposal = 0.0 if not z else clamped / z
                visit(updated, step + 1, reference_mass * rho * q_batch,
                      direct_mass * rho * proposal)

    visit((None, None), 0, 1., 1.)
    weighted = {y: probability * math.exp(verify_music(y, spec).soft_score)
                for y, probability in terminal_reference.items() if verify_music(y, spec).valid}
    normalizer = sum(weighted.values())
    target = {y: weight / normalizer for y, weight in weighted.items()}
    direct = {y: terminal_direct.get(y, 0.) for y in target}
    return {"paths": paths, "reference": terminal_reference, "target": target,
            "target_normalizer": normalizer, "direct": direct}


def _state_key(tokens):
    return ",".join(str(int(token)) for token in tokens)


def _particle_benchmarks(trials, seed):
    truth = _enumerate_reference_target(4)
    target = truth["target"]
    rng = np.random.default_rng(seed)
    methods = (("direct", 1), ("path_is", 4), ("path_is", 16), ("smc", 4), ("smc", 16))
    rows = []
    for method, count in methods:
        selected_counts = {y: 0 for y in target}
        weighted_totals = {y: 0. for y in target}
        normalizers, esses, calls, failures = [], [], [], []
        started = time.perf_counter()
        for trial in range(trials):
            child_seed = int(rng.integers(0, 2**63 - 1))
            try:
                if method == "direct":
                    result = direct_decode(_tiny_spec(), _tiny_provider, steps=4, seed=child_seed,
                                           backend="automaton", track_path_weights=True)
                    tokens, outputs, weights = result.tokens, (result.tokens,), (1.,)
                    estimate, ess = math.exp(result.log_path_weight), 1.
                else:
                    function = path_is if method == "path_is" else smc_decode
                    result = function(_tiny_spec(), _tiny_provider, particles=count, steps=4,
                                      seed=child_seed, backend="automaton")
                    tokens, outputs, weights = result.tokens, result.particles, result.normalized_weights
                    estimate, ess = math.exp(result.log_normalizer_estimate), result.ess
                if tokens not in target:
                    raise RuntimeError("Particle method returned a token sequence outside the exact target support")
                selected_counts[tokens] += 1
                for output, weight in zip(outputs, weights):
                    if output is not None and weight > 0:
                        if output not in target:
                            raise RuntimeError("Positive-weight particle is outside exact target support")
                        weighted_totals[output] += estimate * weight
                normalizers.append(estimate)
                esses.append(ess)
                calls.append(result.model_calls)
            except Exception as exc:
                failures.append({"trial": trial, "seed": child_seed, "error_type": type(exc).__name__, "reason": str(exc)})
                if isinstance(exc, ZeroMass):
                    normalizers.append(0.0)
        successes = sum(selected_counts.values())
        empirical = {y: selected_counts[y] / successes for y in target} if successes else {}
        weighted_sum = sum(weighted_totals.values())
        pooled = {y: weighted_totals[y] / weighted_sum for y in target} if weighted_sum else {}
        mean_z = float(np.mean(normalizers)) if normalizers else None
        stderr_z = float(np.std(normalizers, ddof=1) / math.sqrt(len(normalizers))) if len(normalizers) > 1 else None
        rows.append({"method": method, "particles": count, "attempted_trials": trials,
                     "successful_trials": successes, "failures": failures,
                     "selected_counts": {_state_key(y): value for y, value in selected_counts.items()},
                     "selected_distribution": {_state_key(y): value for y, value in empirical.items()},
                     "selected_empirical_tv": .5 * sum(abs(empirical[y] - target[y]) for y in target) if empirical else None,
                     "pooled_weighted_distribution": {_state_key(y): value for y, value in pooled.items()},
                     "pooled_weighted_tv": .5 * sum(abs(pooled[y] - target[y]) for y in target) if pooled else None,
                     "mean_normalizer_estimate": mean_z, "normalizer_standard_error": stderr_z,
                     "normalizer_trials": len(normalizers),
                     "mean_ess": float(np.mean(esses)) if esses else None,
                     "mean_model_calls": float(np.mean(calls)) if calls else None,
                     "total_model_calls_successful_trials": sum(calls),
                     "elapsed_seconds": time.perf_counter() - started})
    return {"scope": "two-slot state-dependent categorical music model; exhaustive original-token reference paths",
            "steps": 4, "trials_per_method": trials, "reference_paths_enumerated": truth["paths"],
            "reference_total_mass": sum(truth["reference"].values()),
            "reference_terminal_distribution": {_state_key(y): p for y, p in truth["reference"].items()},
            "exact_target_distribution": {_state_key(y): p for y, p in target.items()},
            "exact_target_normalizer": truth["target_normalizer"],
            "exact_direct_distribution": {_state_key(y): p for y, p in truth["direct"].items()},
            "exact_direct_tv": .5 * sum(abs(truth["direct"][y] - target[y]) for y in target),
            "methods": rows,
            "interpretation": "Selected empirical TV measures finite-particle output, not exactness. Pooled weighted TV is a separate weighted-measure diagnostic. Increasing particles need not improve every finite run; no music quality is evaluated."}


def _records_benchmark(trials, seed):
    domains = {"id_a": (0, 1, 2, 3), "id_b": (0, 1, 2, 3), "category_a": (0, 1, 2), "category_b": (0, 1, 2)}
    probabilities = {"id_a": np.array([.55, .25, .15, .05]), "id_b": np.array([.1, .2, .3, .4]),
                     "category_a": np.array([.2, .5, .3]), "category_b": np.array([.6, .3, .1])}
    allowed_a = ({0, 1}, {1}, {1, 2}, {2})
    allowed_b = ({0}, {0, 2}, {1, 2}, {0, 1})
    def valid(record):
        a, b, ca, cb = record
        return a == b and ca in allowed_a[a] and cb in allowed_b[b]
    factors = [LogFactor((name,), np.log(p)) for name, p in probabilities.items()]
    equality = np.full((4, 4), -np.inf)
    np.fill_diagonal(equality, 0.)
    factors.append(LogFactor(("id_a", "id_b"), equality))
    for name, category, allowed in (("id_a", "category_a", allowed_a), ("id_b", "category_b", allowed_b)):
        values = np.full((4, 3), -np.inf)
        for identifier, options in enumerate(allowed):
            for option in options:
                values[identifier, option] = 0.
        factors.append(LogFactor((name, category), values))
    engine = ExactInference(FactorGraph(domains, tuple(factors)))
    names = tuple(domains)
    oracle = {}
    for record in product(*domains.values()):
        if valid(record):
            oracle[record] = math.prod(probabilities[name][value] for name, value in zip(names, record))
    z = sum(oracle.values())
    log_z = engine.log_partition()
    rng = np.random.default_rng(seed)
    raw_valid = joint_valid = 0
    batch_probability_error = 0.
    for _ in range(trials):
        raw = tuple(int(rng.choice(len(probabilities[name]), p=probabilities[name])) for name in names)
        raw_valid += valid(raw)
        sampled = engine.sample_batch(names, rng)
        record = tuple(sampled.assignment[name] for name in names)
        joint_valid += valid(record)
        if record in oracle:
            batch_probability_error = max(batch_probability_error, abs(sampled.log_probability - math.log(oracle[record] / z)))
        else:
            batch_probability_error = math.inf
    return {"scope": "synthetic linked records with a shared unknown identifier and local category restrictions; no learned LLM",
            "trials": trials, "original_assignments": math.prod(len(domain) for domain in domains.values()),
            "valid_original_assignments": len(oracle), "raw_valid": raw_valid, "joint_valid": joint_valid,
            "raw_valid_rate": raw_valid / trials, "joint_valid_rate": joint_valid / trials,
            "exact_unconstrained_valid_probability": z,
            "log_partition": log_z, "independent_log_partition": math.log(z),
            "log_partition_absolute_error": abs(log_z - math.log(z)),
            "max_joint_log_probability_error": batch_probability_error,
            "verified": joint_valid == trials and abs(log_z - math.log(z)) < 1e-12 and batch_probability_error < 1e-12}


def research_benchmarks(output_dir, trials=128, seed=20260912):
    """Write one reproducible diagnostic report; all failed cases are retained."""
    if isinstance(trials, bool) or not isinstance(trials, Integral) or trials < 1:
        raise InvalidSpecification("trials must be a positive integer")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    backends = _backend_benchmarks(seed)
    particles = _particle_benchmarks(int(trials), seed + 1)
    records = _records_benchmark(int(trials), seed + 2)
    unexpected_backend_errors = [row for case in backends["cases"] for row in case["backends"]
                                 if row["status"] == "failed" and row["error_type"] != "BudgetExceeded"]
    report = {"scope": "synthetic mathematical and computational diagnostics; no trained-model, publication, or music-quality conclusion",
              "seed": int(seed), "trials": int(trials), "config": {"trials": int(trials), "seed": int(seed)},
              "outputs": {"report": str((directory / "report.json").resolve())}, "music_backends": backends,
              "finite_particles": particles, "linked_records": records,
              "checks_passed": backends["all_comparable_cases_agree"] and records["verified"] and not unexpected_backend_errors
                               and all(not row["failures"] for row in particles["methods"]),
              "elapsed_seconds": time.perf_counter() - started}
    report["status"] = "completed" if report["checks_passed"] else "failed"
    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report
