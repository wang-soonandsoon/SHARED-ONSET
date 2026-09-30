import json

import pytest

from tri.benchmarks_research import (
    _backend_benchmarks, _enumerate_reference_target, _particle_benchmarks,
    _records_benchmark, research_benchmarks,
)
from tri.errors import InvalidSpecification


def test_reference_target_and_direct_are_independently_enumerated():
    exact = _enumerate_reference_target(4)
    assert exact["paths"] == 64
    assert sum(exact["reference"].values()) == pytest.approx(1.)
    assert exact["reference"][(62, 62)] == pytest.approx(.27)
    assert exact["reference"][(0, 0)] == pytest.approx(.07)
    assert exact["target_normalizer"] == pytest.approx(.34)
    assert exact["target"][(62, 62)] == pytest.approx(27 / 34)
    assert exact["direct"][(62, 62)] == pytest.approx(81 / 82)
    assert sum(exact["direct"].values()) == pytest.approx(1.)


def test_same_probability_case_compares_all_backends():
    report = _backend_benchmarks(618, widths=(2,), pitch_counts=(3,))
    assert report["all_comparable_cases_agree"]
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["successful_backends"] == 4
    assert case["max_log_partition_disagreement"] < 1e-11
    assert case["max_batch_log_probability_disagreement"] < 1e-11


def test_finite_particle_report_preserves_compute_counts_and_empirical_limits():
    report = _particle_benchmarks(4, 628)
    assert report["reference_total_mass"] == pytest.approx(1.)
    assert report["exact_target_normalizer"] == pytest.approx(.34)
    assert report["exact_direct_tv"] > .19
    assert len(report["methods"]) == 5
    for row in report["methods"]:
        assert row["successful_trials"] == 4
        assert row["failures"] == []
        assert sum(row["selected_counts"].values()) == 4
        assert 0 <= row["selected_empirical_tv"] <= 1
        assert 0 <= row["pooled_weighted_tv"] <= 1
        assert row["mean_normalizer_estimate"] > 0
        assert row["mean_model_calls"] >= 1
        assert 1 <= row["mean_ess"] <= row["particles"] + 1e-12


def test_linked_records_retain_two_contexts_and_match_original_enumeration():
    report = _records_benchmark(32, 619)
    assert report["original_assignments"] == 144
    assert report["valid_original_assignments"] == 10
    assert report["exact_unconstrained_valid_probability"] == pytest.approx(.0604)
    assert report["joint_valid"] == 32
    assert report["verified"]


def test_report_has_runner_cache_contract(tmp_path, monkeypatch):
    import tri.benchmarks_research as module
    original = module._backend_benchmarks
    monkeypatch.setattr(module, "_backend_benchmarks", lambda seed: original(seed, widths=(2,), pitch_counts=(3,)))
    result = research_benchmarks(tmp_path, trials=2, seed=681)
    assert result["status"] == "completed"
    assert result["config"] == {"trials": 2, "seed": 681}
    assert result["outputs"]["report"] == str(tmp_path / "report.json")
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved == result


def test_invalid_trial_count_is_rejected(tmp_path):
    for value in (0, -1, True, 1.5):
        with pytest.raises(InvalidSpecification):
            research_benchmarks(tmp_path, trials=value)
