import math
import unittest
from unittest.mock import patch

import numpy as np

from tri.errors import BudgetExceeded, InvalidSpecification, ZeroMass
from tri.inference import Budget, EnumeratedInference, ExactInference, FactorGraph, LogFactor


def equality_graph():
    return FactorGraph({"x": (4, 9), "y": (-1, 7)}, (
        LogFactor(("x",), np.log([0.9, 0.1])),
        LogFactor(("y",), np.log([0.2, 0.8])),
        LogFactor(("x", "y"), np.array([[0., -np.inf], [-np.inf, 0.]])),
    ))


class ExactInferenceTests(unittest.TestCase):
    def test_100_seeded_graphs_against_independent_enumeration(self):
        for seed in range(100):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                names = [f"v{i}" for i in range(int(rng.integers(1, 6)))]
                domains = {v: tuple(int(i * 3 - 4) for i in range(int(rng.integers(1, 4)))) for v in names}
                factors = []
                for _ in range(int(rng.integers(0, 8))):
                    scope = tuple(rng.choice(names, size=int(rng.integers(0, min(3, len(names)) + 1)), replace=False))
                    shape = tuple(len(domains[v]) for v in scope)
                    values = rng.normal(size=shape) * 20
                    values = np.where(rng.random(size=shape) < 0.15, -np.inf, values)
                    factors.append(LogFactor(scope, values))
                graph = FactorGraph(domains, tuple(factors))
                exact = ExactInference(graph)
                oracle = EnumeratedInference(graph)
                np.testing.assert_allclose(exact.log_partition(), oracle.log_partition(), atol=1e-11)
                explicit = ExactInference(graph, order=tuple(reversed(names)))
                np.testing.assert_allclose(explicit.log_partition(), oracle.log_partition(), atol=1e-11)
                evidence = {names[0]: domains[names[0]][0]} if seed % 2 else {}
                np.testing.assert_allclose(exact.log_clamped_partition(evidence), oracle.log_partition(evidence), atol=1e-11)
                if np.isneginf(oracle.log_partition(evidence)):
                    with self.assertRaises(ZeroMass):
                        exact.marginal_log_probs(names[0], evidence)
                    with self.assertRaises(ZeroMass):
                        exact.sample_batch(names, rng, evidence)
                    continue
                for variable in names:
                    np.testing.assert_allclose(exact.marginal_log_probs(variable, evidence),
                                               oracle.marginal_log_probs(variable, evidence), atol=1e-11)
                batch = exact.sample_batch(names[::2], rng, evidence)
                np.testing.assert_allclose(batch.log_probability, oracle.log_probability(batch.assignment, evidence), atol=1e-11)
                np.testing.assert_allclose(batch.log_clamped_partition,
                                           oracle.log_partition({**evidence, **batch.assignment}), atol=1e-11)

    def test_clamp_keeps_old_model_probability(self):
        engine = ExactInference(equality_graph())
        self.assertAlmostEqual(math.exp(engine.log_partition()), 0.26)
        self.assertAlmostEqual(math.exp(engine.log_clamped_partition({"x": 9})), 0.08)
        self.assertAlmostEqual(math.exp(engine.log_probability({"x": 9})), 0.08 / 0.26)
        # Renormalizing q_x on the clamped vocabulary would incorrectly give .8.
        self.assertNotAlmostEqual(math.exp(engine.log_clamped_partition({"x": 9})), 0.8)

    def test_joint_samples_obey_relation_and_report_batch_mass(self):
        engine = ExactInference(equality_graph())
        rng = np.random.default_rng(49)
        count = 0
        for _ in range(600):
            sample = engine.sample_batch(("y", "x"), rng)
            self.assertIn((sample.assignment["x"], sample.assignment["y"]), ((4, -1), (9, 7)))
            self.assertAlmostEqual(sample.log_probability, engine.log_probability(sample.assignment))
            count += sample.assignment["x"] == 9
        self.assertLess(abs(count / 600 - 0.08 / 0.26), 0.065)
        partial = engine.sample_batch(("x",), rng)
        self.assertEqual(set(partial.assignment), {"x"})
        self.assertAlmostEqual(partial.log_probability, engine.log_probability(partial.assignment))

    def test_empty_batch_and_observed_variable(self):
        engine = ExactInference(equality_graph())
        empty = engine.sample_batch((), np.random.default_rng(2), {"x": 4})
        self.assertEqual(empty.assignment, {})
        self.assertEqual(empty.log_probability, 0.)
        self.assertAlmostEqual(empty.log_clamped_partition, math.log(.18))
        np.testing.assert_array_equal(engine.marginal_log_probs("x", {"x": 4}), [0, -np.inf])
        self.assertEqual(engine.log_probability({"x": 9}, {"x": 4}), -np.inf)

    def test_input_snapshots_are_immutable(self):
        raw = np.array([1., 2.])
        scope = ["x"]
        factor = LogFactor(scope, raw)
        domains = {"x": [1, 2]}
        factors = [factor]
        graph = FactorGraph(domains, factors)
        raw[:] = 900
        scope[0] = "wrong"
        domains["x"].append(3)
        factors.clear()
        self.assertEqual(graph.domains["x"], (1, 2))
        self.assertEqual(graph.factors[0].scope, ("x",))
        np.testing.assert_array_equal(graph.factors[0].values, [1., 2.])
        with self.assertRaises(ValueError):
            graph.factors[0].values.flags.writeable = True
        with self.assertRaises(TypeError):
            graph.domains["z"] = (1,)

    def test_malformed_graphs_are_rejected(self):
        constructors = (
            lambda: LogFactor(("x", "x"), np.zeros((2, 2))),
            lambda: LogFactor(("x",), np.zeros((2, 2))),
            lambda: LogFactor(("x",), [0, np.nan]),
            lambda: LogFactor(("x",), [0, np.inf]),
            lambda: FactorGraph({"x": (1, 1)}, ()),
            lambda: FactorGraph({"x": ()}, ()),
            lambda: FactorGraph({"x": (1.2,)}, ()),
            lambda: FactorGraph({"x": (True,)}, ()),
            lambda: FactorGraph({"x": (1,)}, (LogFactor(("x",), [0, 0]),)),
            lambda: FactorGraph({"x": (1,)}, (LogFactor(("z",), [0]),)),
        )
        for make in constructors:
            with self.assertRaises(InvalidSpecification):
                make()

    def test_invalid_queries_and_orders(self):
        engine = ExactInference(equality_graph())
        with self.assertRaises(InvalidSpecification):
            ExactInference(engine.graph, order=("x", "x"))
        for evidence in ({"x": 0}, {"unknown": 4}, {"x": 4.0}, {"x": True}):
            with self.assertRaises(InvalidSpecification):
                engine.log_partition(evidence)
        with self.assertRaises(InvalidSpecification):
            engine.sample_batch(("x", "x"), np.random.default_rng(0))
        with self.assertRaises(InvalidSpecification):
            engine.sample_batch("x", np.random.default_rng(0))
        with self.assertRaises(InvalidSpecification):
            engine.marginal_log_probs("missing")

    def test_scalar_disconnected_and_singleton_domains(self):
        graph = FactorGraph({"a": (8,), "b": (-2, 3, 7)}, (LogFactor((), np.array(7.5)),))
        for backend in (ExactInference, EnumeratedInference):
            engine = backend(graph)
            self.assertAlmostEqual(engine.log_partition(), 7.5 + math.log(3))
            np.testing.assert_allclose(engine.marginal_log_probs("b"), np.full(3, -math.log(3)))
            np.testing.assert_array_equal(engine.marginal_log_probs("a"), [0])
            self.assertEqual(backend(FactorGraph({}, ())).log_partition(), 0.)

    def test_extreme_weights_and_zero_mass(self):
        graph = FactorGraph({"x": (0, 1)}, (LogFactor(("x",), [-10000., -10001.]), LogFactor((), np.array(3000.))))
        for backend in (ExactInference, EnumeratedInference):
            engine = backend(graph)
            self.assertAlmostEqual(engine.log_partition(), -7000 + math.log1p(math.exp(-1)))
            np.testing.assert_allclose(np.exp(engine.marginal_log_probs("x")), [1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))])
            empty = backend(FactorGraph({"x": (0, 1)}, (LogFactor(("x",), [-np.inf, -np.inf]),)))
            self.assertEqual(empty.log_partition(), -np.inf)
            with self.assertRaises(ZeroMass):
                empty.sample_batch((), np.random.default_rng(0))
            with self.assertRaises(ZeroMass):
                empty.log_probability({})

    def test_budget_stops_dense_intermediate_before_allocation(self):
        domains = {"center": (0, 1), **{f"leaf{i}": (0, 1) for i in range(7)}}
        factors = tuple(LogFactor(("center", f"leaf{i}"), np.zeros((2, 2))) for i in range(7))
        graph = FactorGraph(domains, factors)
        good = ExactInference(graph, Budget(max_factor_entries=16))
        self.assertAlmostEqual(good.log_partition(), 8 * math.log(2))
        self.assertLessEqual(good.last_stats["max_factor_entries"], 4)
        bad = ExactInference(graph, Budget(max_factor_entries=16), order=tuple(domains))
        with patch("tri.inference.exact.np.zeros", side_effect=AssertionError("Allocated before budget check")):
            with self.assertRaises(BudgetExceeded):
                bad.log_partition()
        workspace = ExactInference(equality_graph(), Budget(max_workspace_bytes=100))
        with self.assertRaises(BudgetExceeded):
            workspace.log_partition()
        with self.assertRaises(BudgetExceeded):
            ExactInference(graph, Budget(max_factor_entries=2))
        with self.assertRaises(BudgetExceeded):
            EnumeratedInference(graph, Budget(max_oracle_assignments=100)).log_partition()

    def test_partition_cache_is_bounded_and_inputs_unchanged(self):
        engine = ExactInference(FactorGraph({"x": tuple(range(200))}, ()))
        for i in range(200):
            evidence = {"x": i}
            self.assertEqual(engine.log_partition(evidence), 0.)
            self.assertEqual(evidence, {"x": i})
        self.assertEqual(len(engine._cache), 128)
        engine.log_partition({"x": 199})
        self.assertTrue(engine.last_stats["cache_hit"])


if __name__ == "__main__":
    unittest.main()
