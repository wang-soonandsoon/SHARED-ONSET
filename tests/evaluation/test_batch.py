from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tri.errors import BudgetExceeded, UnsupportedSpec, VerificationError, ZeroMass
from tri.evaluation.batch import EvaluationWindow, evaluate_windows, load_evaluation_windows
from tri.sampling.baselines import METHODS


GOOD = (62, 1, 0, 0, 64, 1, 0, 0, 65, 1, 0, 0, 67, 1, 0, 0)


class BatchEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, names=("b", "a", "train", "test"), starts=(32, 0, 0, 0),
                splits=("validation", "val", "train", "test"), order=None, **overrides):
        values = {
            "tokens": np.asarray([GOOD] * len(names), dtype=np.int64),
            "work_ids": np.asarray(names), "start_cells": np.asarray(starts, dtype=np.int64),
            "initial_pitches": np.full(len(names), -1, dtype=np.int64),
            "time_signatures": np.tile([4, 4], (len(names), 1)), "splits": np.asarray(splits),
        }
        if order is not None:
            values = {name: array[order] for name, array in values.items()}
        values.update(overrides)
        path = self.root / f"dataset_{len(list(self.root.glob('*.npz')))}.npz"
        np.savez(path, **values)
        return path

    def window(self, work="a", start=0, tokens=GOOD, index=0):
        return EvaluationWindow(index, work, start, tuple(tokens), None, (4, 4), "validation")

    @staticmethod
    def factory(spec):
        return lambda tokens, noise: np.full((spec.length, 130), -np.log(130))

    @staticmethod
    def success_decoder(method, spec, provider, **kwargs):
        provider(tuple(spec.observed.get(i) for i in range(spec.length)), 1.0)
        return SimpleNamespace(tokens=GOOD, model_calls=1, trace=())

    def run_batch(self, windows=None, decoder=None, methods=("tri_direct",), **kwargs):
        return evaluate_windows(windows or [self.window()], self.root / "result",
                                provider_factory=kwargs.pop("provider_factory", self.factory),
                                decoder=decoder or self.success_decoder, methods=methods, seed=100, **kwargs)

    def rows(self, report):
        return [json.loads(line) for line in Path(report["outputs"]["results"]).read_text().splitlines()]

    def test_validation_selection_is_stable_under_row_permutation(self):
        first = load_evaluation_windows(self.dataset(), limit=8)
        second = load_evaluation_windows(self.dataset(order=[3, 0, 2, 1]), limit=8, split="val")
        self.assertEqual([(w.work_id, w.start_cell) for w in first], [("a", 0), ("b", 32)])
        self.assertEqual([w.request_id for w in first], [w.request_id for w in second])
        self.assertNotEqual([w.source_index for w in first], [w.source_index for w in second])

    def test_unselected_cross_split_work_is_rejected(self):
        path = self.dataset(names=("a", "x", "x"), starts=(0, 0, 32), splits=("validation", "train", "test"))
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            load_evaluation_windows(path, limit=1)

    def test_no_validation_never_falls_back(self):
        path = self.dataset(names=("a", "b"), starts=(0, 0), splits=("train", "test"))
        with self.assertRaisesRegex(ValueError, "no validation windows"):
            load_evaluation_windows(path)

    def test_duplicate_identity_and_array_mismatch_are_rejected(self):
        path = self.dataset(names=("a", "a"), starts=(0, 0), splits=("validation", "validation"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_evaluation_windows(path)
        with self.assertRaisesRegex(ValueError, "array length mismatch"):
            load_evaluation_windows(self.dataset(initial_pitches=np.array([-1])))

    def test_invalid_tokens_in_unselected_rows_are_rejected(self):
        tokens = np.asarray([GOOD] * 4)
        tokens[3, 0] = 130
        with self.assertRaisesRegex(ValueError, "output IDs"):
            load_evaluation_windows(self.dataset(tokens=tokens))

    def test_forced_failure_stays_in_denominator_and_call_count(self):
        def decode(method, spec, provider, **kwargs):
            result = self.success_decoder(method, spec, provider, **kwargs)
            if kwargs["seed"] == 100:
                raise BudgetExceeded("forced budget failure")
            return result
        report = self.run_batch([self.window("a"), self.window("b", index=1)], decoder=decode)
        summary = report["by_method"]["tri_direct"]
        self.assertEqual((summary["attempted"], summary["returned"], summary["valid"], summary["exported"]), (2, 1, 1, 1))
        self.assertEqual(summary["completion_rate"], 0.5)
        self.assertEqual(summary["model_calls_total"], 2)
        self.assertEqual(summary["timed_decodes"], 2)
        self.assertEqual(self.rows(report)[0]["status"], "budget_exceeded")

    def test_raw_invalid_output_is_saved_and_not_exported(self):
        bad = list(GOOD)
        bad[4] = 130
        def decode(method, spec, provider, **kwargs):
            provider(tuple(spec.observed.get(i) for i in range(spec.length)), 1.0)
            return SimpleNamespace(tokens=tuple(bad), model_calls=1, trace=())
        report = self.run_batch(decoder=decode, methods=("raw_reference",))
        row = self.rows(report)[0]
        self.assertEqual(row["status"], "returned_invalid")
        self.assertTrue(row["returned"])
        self.assertFalse(row["valid"])
        self.assertFalse(row["exported"])
        self.assertEqual(row["raw_tokens"], bad)
        self.assertEqual(list(Path(report["outputs"]["midi_dir"]).glob("*.mid")), [])

    def test_fixed_hold_token_can_hide_changed_sounding_pitch(self):
        source = list(GOOD)
        source[4:8] = [62, 1, 1, 1]
        changed = source.copy()
        changed[4] = 66
        def decode(method, spec, provider, **kwargs):
            return SimpleNamespace(tokens=tuple(changed), model_calls=0, trace=())
        report = self.run_batch([self.window(tokens=source)], decoder=decode)
        row = self.rows(report)[0]
        self.assertTrue(row["verification"]["fixed_tokens_preserved"])
        self.assertFalse(row["verification"]["fixed_sounding_pitches_preserved"])
        self.assertFalse(row["valid"])
        self.assertEqual(row["status"], "returned_invalid")

    def test_hidden_reference_never_reaches_factory_or_manifest(self):
        source = list(GOOD)
        source[4], source[12] = 113, 116  # hidden NOTE pitches 111 and114
        seen = []
        def factory(spec):
            seen.append(spec)
            self.assertNotIn(111, spec.pitches)
            self.assertNotIn(114, spec.pitches)
            self.assertTrue(all(position not in spec.observed for position in (4, 5, 6, 12, 13, 14)))
            return self.factory(spec)
        report = self.run_batch([self.window(tokens=source)], provider_factory=factory)
        manifest = json.loads(Path(report["outputs"]["requests"]).read_text())["requests"][0]
        self.assertEqual(len(seen), 1)
        self.assertNotIn("tokens", manifest)
        self.assertNotIn("original_tokens", manifest)
        self.assertNotIn("clean_tokens", manifest)
        self.assertNotIn("4", manifest["observed_tokens"])
        self.assertNotIn(111, manifest["working_pitches"])

    def test_all_methods_share_request_object_and_per_request_seed(self):
        seen = []
        def decode(method, spec, provider, **kwargs):
            seen.append((method, spec, kwargs["seed"]))
            return self.success_decoder(method, spec, provider, **kwargs)
        report = self.run_batch(decoder=decode, methods=METHODS)
        self.assertEqual({seed for _, _, seed in seen}, {100})
        self.assertTrue(all(spec is seen[0][1] for _, spec, _ in seen))
        self.assertEqual(len(self.rows(report)), 4)
        self.assertTrue(all(summary["attempted"] == 1 for summary in report["by_method"].values()))

    def test_export_failure_keeps_independent_validity(self):
        def bad_exporter(*args, **kwargs):
            raise OSError("forced disk failure")
        report = self.run_batch(exporter=bad_exporter)
        row = self.rows(report)[0]
        self.assertEqual(row["status"], "export_failure")
        self.assertTrue(row["valid"])
        self.assertFalse(row["exported"])
        self.assertEqual(report["by_method"]["tri_direct"]["completion_rate"], 1.0)

    def test_expected_failure_classes_remain_distinct(self):
        for error, status in [(ZeroMass("z"), "zero_mass"), (UnsupportedSpec("u"), "unsupported"),
                              (VerificationError("v"), "verifier_failure")]:
            def fail(*args, **kwargs):
                raise error
            report = self.run_batch(decoder=fail)
            self.assertEqual(self.rows(report)[0]["status"], status)
            self.assertEqual(report["by_method"]["tri_direct"]["attempted"], 1)

    def test_unexpected_error_is_obvious_and_does_not_stop_next_request(self):
        def decode(method, spec, provider, **kwargs):
            if kwargs["seed"] == 100:
                raise RuntimeError("broken implementation")
            return self.success_decoder(method, spec, provider, **kwargs)
        report = self.run_batch([self.window("a"), self.window("b")], decoder=decode)
        self.assertEqual(report["status"], "needs_attention")
        self.assertEqual(report["unexpected_errors"], 1)
        rows = self.rows(report)
        self.assertIn("traceback", rows[0]["error"])
        self.assertEqual(rows[1]["status"], "exported")

    def test_reported_backend_calls_must_match_observed_calls(self):
        def decode(method, spec, provider, **kwargs):
            return SimpleNamespace(tokens=GOOD, model_calls=9, trace=())
        report = self.run_batch(decoder=decode)
        row = self.rows(report)[0]
        self.assertTrue(row["returned"])
        self.assertEqual(row["raw_tokens"], list(GOOD))
        self.assertEqual(row["model_calls"], 0)
        self.assertEqual(row["backend_model_calls"], 9)
        self.assertEqual(row["status"], "unexpected_error")

    def test_invalid_rerun_removes_only_its_stale_midi(self):
        first = self.run_batch()
        midi_dir = Path(first["outputs"]["midi_dir"])
        managed = midi_dir / "request_0000_tri_direct.mid"
        self.assertTrue(managed.exists())
        unrelated = midi_dir / "user_kept.mid"
        unrelated.write_bytes(b"keep")
        def invalid(method, spec, provider, **kwargs):
            return SimpleNamespace(tokens=(130,) * len(GOOD), model_calls=0, trace=())
        second = self.run_batch(decoder=invalid)
        self.assertEqual(self.rows(second)[0]["status"], "returned_invalid")
        self.assertFalse(managed.exists())
        self.assertEqual(unrelated.read_bytes(), b"keep")

    def test_unexpected_export_implementation_error_is_not_hidden(self):
        def broken_exporter(*args, **kwargs):
            raise RuntimeError("exporter implementation bug")
        report = self.run_batch(exporter=broken_exporter)
        row = self.rows(report)[0]
        self.assertEqual(row["status"], "unexpected_error")
        self.assertTrue(row["valid"])
        self.assertFalse(row["exported"])
        self.assertIn("traceback", row["error"])
        self.assertEqual(report["status"], "needs_attention")

    def test_smaller_rerun_and_changed_methods_remove_all_previous_managed_outputs(self):
        first = self.run_batch([self.window("a"), self.window("b")], methods=METHODS)
        midi_dir = Path(first["outputs"]["midi_dir"])
        self.assertEqual(len(list(midi_dir.glob("*.mid"))), 8)
        unrelated = midi_dir / "request_0000_personal_method.mid"
        unrelated.write_bytes(b"user file")
        outside = self.root / "outside.mid"
        outside.write_bytes(b"do not delete")
        # A corrupted or hand-edited old result must not authorize deletion of
        # arbitrary external paths; cleanup uses owned basenames only.
        Path(first["outputs"]["results"]).write_text(json.dumps({"midi": {"path": str(outside)}}) + "\n")
        second = self.run_batch([self.window("a")], methods=("one_shot_joint",))
        self.assertEqual({path.name for path in midi_dir.glob("*.mid")},
                         {"request_0000_one_shot_joint.mid", unrelated.name})
        self.assertEqual(outside.read_bytes(), b"do not delete")
        self.assertEqual(unrelated.read_bytes(), b"user file")
        self.assertEqual(len(self.rows(second)), 1)

    def test_owned_symlink_cleanup_does_not_delete_external_target(self):
        report = self.run_batch()
        midi_dir = Path(report["outputs"]["midi_dir"])
        outside = self.root / "outside.mid"
        outside.write_bytes(b"keep external target")
        alias = midi_dir / "request_9999_raw_reference.mid"
        alias.symlink_to(outside)
        self.run_batch()
        self.assertFalse(alias.exists())
        self.assertEqual(outside.read_bytes(), b"keep external target")


if __name__ == "__main__":
    unittest.main()
