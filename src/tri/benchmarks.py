"""Bounded inference comparison; standard DP, not a novelty experiment."""
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from tri.inference.factors import FactorGraph, LogFactor
from tri.inference.exact import ExactInference
from tri.inference.rhythm import SharedOnsetCountInference


def rhythm_factor_graph(log_probs, flags, count):
    """Compact generic baseline: shared binary states and prefix counters.

    Uses the same unknown template for every replica; no weak one-context
    template or exponentially dense count factor is used for comparison.
    """
    replicas, length, vocabulary = log_probs.shape
    domains, factors = {}, []
    for j in range(length):
        z, c = f"z{j}", f"c{j}"
        domains[z] = (0, 1)
        domains[c] = (count,) if j == length-1 else tuple(range(min(count, j+1)+1))
        for r in range(replicas):
            y = f"y{r}_{j}"
            domains[y] = tuple(range(vocabulary))
            factors.append(LogFactor((y,), log_probs[r, j]))
            relation = np.where(np.asarray(flags)[:, None] == np.arange(2)[None], 0.0, -np.inf)
            factors.append(LogFactor((y, z), relation))
        if j == 0:
            table = np.where(np.arange(2)[:, None] == np.array(domains[c])[None], 0.0, -np.inf)
            factors.append(LogFactor((z, c), table))
        else:
            previous = f"c{j-1}"
            table = np.where(np.array(domains[previous])[:, None, None] + np.arange(2)[None, :, None]
                             == np.array(domains[c])[None, None, :], 0.0, -np.inf)
            factors.append(LogFactor((previous, z, c), table))
    return FactorGraph(domains, tuple(factors))


def benchmark_inference(output_dir, *, seed=20260912):
    rng = np.random.default_rng(seed)
    rows = []
    for replicas, length, count in [(2, 8, 2), (2, 16, 4), (2, 32, 8), (4, 16, 4)]:
        vocabulary = 5
        logits = rng.normal(size=(replicas, length, vocabulary))
        log_q = logits - logsumexp(logits, axis=-1, keepdims=True)
        flags = np.array([False, False, True, True, True])
        dp_times, ve_times = [], []
        for repeat in range(3):
            start = time.perf_counter()
            dp = SharedOnsetCountInference(log_q, flags, count)
            dp_z = dp.log_partition()
            dp_times.append(time.perf_counter()-start)
            start = time.perf_counter()
            ve = ExactInference(rhythm_factor_graph(log_q, flags, count))
            ve_z = ve.log_partition()
            ve_times.append(time.perf_counter()-start)
        draw = dp.sample_batch(((0, 0), (replicas-1, 0), (0, length-1)), rng)
        assignment = {f"y{r}_{j}": value for (r, j), value in draw.assignment.items()}
        clamp_error = abs(dp.log_clamped_partition(draw.assignment) - ve.log_clamped_partition(assignment))
        probability_error = abs(draw.log_probability - ve.log_probability(assignment))
        if not np.isclose(dp_z, ve_z, atol=1e-10, rtol=1e-10) or max(clamp_error, probability_error) > 1e-9:
            raise RuntimeError("rhythm DP and compact generic baseline disagree")
        rows.append({"replicas": replicas, "length": length, "categories": vocabulary, "onset_count": count,
                     "log_z": dp_z, "log_z_absolute_error": abs(dp_z-ve_z),
                     "clamp_absolute_error": clamp_error, "batch_logprob_absolute_error": probability_error,
                     "dp_build_and_partition_seconds": dp_times,
                     "ve_build_and_partition_seconds": ve_times,
                     "dp_median_seconds": float(np.median(dp_times)), "ve_median_seconds": float(np.median(ve_times)),
                     "dp_stats": dp.last_stats})
    report = {"scope": "independent categories, shared unknown onset template and fixed count ONLY",
              "seed": seed, "timing": "CPU, construction plus first partition; three repetitions",
              "baseline": "compact shared binary variables and prefix counters; weighted min-fill VE",
              "limitations": ["No HOLD/sounding-state or local pitch coupling in this benchmark.",
                              "Standard DP validation; does not establish a novel TRI speedup.",
                              "No neural model, compilation amortization, or end-to-end music quality comparison."],
              "cases": rows}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
