from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mido
import numpy as np

from tri.data.chords import (CHORD_FEATURE_DIM, ChordSourceError, align_chords,
                             encode_chord_features, load_chord_sidecar,
                             read_chord_intervals, read_tempo_map)


class ChordAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.shared = self.root / "data"

    def tearDown(self):
        self.temp.cleanup()

    def midi(self, work="001", changes=((0, 500000),), extra_changes=()):
        directory = self.shared / "raw" / "pop909" / work
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{work}.mid"
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        for events in (changes, extra_changes):
            track, previous = mido.MidiTrack(), 0
            for tick, tempo in events:
                track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=tick - previous))
                previous = tick
            midi.tracks.append(track)
        midi.save(path)
        return path

    def annotations(self, work="001", text="0 100 C:maj\n"):
        path = self.shared / "raw" / "pop909" / work / "chord_midi.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def dataset(self, works=("001",), starts=(8,), splits=("validation",), *, token_value=0):
        path = self.shared / "processed" / "windows" / "windows.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, work_ids=np.asarray(works), start_cells=np.asarray(starts, dtype=np.int64),
                 splits=np.asarray(splits), tokens=np.full((len(works), 8), token_value, dtype=np.int64))
        return path

    def test_tempo_changes_apply_after_their_tick_and_use_absolute_time(self):
        tempo = read_tempo_map(self.midi(changes=((0, 500000), (480, 1000000))))
        self.assertEqual(tempo.tick_to_seconds(0), 0)
        self.assertEqual(tempo.tick_to_seconds(240), 0.25)
        self.assertEqual(tempo.tick_to_seconds(480), 0.5)
        self.assertEqual(tempo.tick_to_seconds(960), 1.5)
        np.testing.assert_array_equal(tempo.cell_seconds(8, 2), [1.5, 1.75, 2.0])

    def test_initial_default_and_tempo_events_on_other_tracks(self):
        tempo = read_tempo_map(self.midi(changes=(), extra_changes=((480, 1000000),)))
        self.assertTrue(tempo.default_initial_tempo_used)
        self.assertEqual(tempo.tick_to_seconds(480), 0.5)
        self.assertEqual(tempo.tick_to_seconds(960), 1.5)
        self.assertEqual(tempo.tempos, (500000, 1000000))

    def test_duplicate_equal_tempos_merge_but_conflicts_fail(self):
        tempo = read_tempo_map(self.midi(extra_changes=((0, 500000), (480, 500000))))
        self.assertEqual(tempo.ticks, (0,))
        with self.assertRaises(ChordSourceError) as caught:
            read_tempo_map(self.midi(extra_changes=((0, 600000),)))
        self.assertEqual(caught.exception.reason, "conflicting_tempo")

    def test_integer_accumulation_preserves_exact_annotation_boundary(self):
        tempo = read_tempo_map(self.midi(changes=((0, 837989), (73920, 840336))))
        seconds = tempo.tick_to_seconds(92160)
        self.assertEqual(seconds, 160.983074)
        intervals = read_chord_intervals(self.annotations(text="0 160.983074 Eb:min\n160.983074 200 Bb:min/b3\n"))
        self.assertEqual(intervals.label_at(seconds), "Bb:min/b3")

    def test_half_open_boundaries_gaps_and_N_are_distinct(self):
        intervals = read_chord_intervals(self.annotations(text="0.5 1 N\n1 2 A:min/5\n3 4 Bb:min/b3\n"))
        for point, expected in [(0, None), (0.5, "N"), (1, "A:min/5"), (2, None), (2.5, None), (3, "Bb:min/b3"), (4, None)]:
            self.assertEqual(intervals.label_at(point), expected)

    def test_malformed_or_overlapping_annotations_are_rejected(self):
        for text in ("0 2 C:maj\n1 3 D:min\n", "nan 1 C:maj\n", "0 0 C:maj\n", "-1 2 C:maj\n", "0 nope C:maj\n"):
            with self.assertRaises(ChordSourceError):
                read_chord_intervals(self.annotations(text=text))

    def test_features_preserve_root_slash_bass_and_extended_chord_tones(self):
        features, known = encode_chord_features(np.asarray(["A:min/5", "Bb:min/b3", "C:maj9", "N", "", "X", "C:unsupported"]))
        self.assertEqual(features.shape, (7, CHORD_FEATURE_DIM))
        np.testing.assert_array_equal(known, [True, True, True, True, False, False, False])
        self.assertEqual(set(np.flatnonzero(features[0, :12])), {9})
        self.assertEqual(set(np.flatnonzero(features[0, 12:24])), {0, 4, 9})
        self.assertEqual(set(np.flatnonzero(features[0, 24:36])), {4})
        self.assertEqual(set(np.flatnonzero(features[1, 24:36])), {1})
        self.assertEqual(set(np.flatnonzero(features[2, 12:24])), {0, 2, 4, 7, 11})
        np.testing.assert_array_equal(features[3, 36:], [1, 1])
        self.assertFalse(np.any(features[3, :36]))
        self.assertFalse(np.any(features[4:]))

    def test_annotation_coverage_can_mask_an_otherwise_supported_label(self):
        features, known = encode_chord_features(np.asarray([["C:maj", "N"]]), np.asarray([[False, True]]))
        np.testing.assert_array_equal(known, [[False, True]])
        self.assertFalse(np.any(features[0, 0]))

    def test_alignment_preserves_row_order_splits_and_long_labels(self):
        self.midi("001", changes=((0, 500000), (480, 1000000)))
        self.midi("002")
        self.annotations("001", "0 1 Eb:min\n1 100 Bb:min/b3\n")
        self.annotations("002", "0 100 A:min7/5\n")
        dataset = self.dataset(works=("002", "001", "002"), starts=(8, 8, 16), splits=("test", "validation", "test"))
        report = align_chords(dataset, self.shared, self.root / "aligned")
        sidecar = load_chord_sidecar(dataset, report["outputs"]["chords"])
        np.testing.assert_array_equal(sidecar["work_ids"], ["002", "001", "002"])
        np.testing.assert_array_equal(sidecar["splits"], ["test", "validation", "test"])
        self.assertEqual(sidecar["cell_seconds"][1, 0], 1.5)
        self.assertEqual(sidecar["chord_labels"][0, 0], "A:min7/5")
        self.assertEqual(sidecar["chord_labels"][1, 0], "Bb:min/b3")
        self.assertEqual(report["variable_tempo_works"], 1)

    def test_failed_annotation_keeps_row_and_usable_times(self):
        self.midi("001")
        self.midi("002")
        self.annotations("001", "0 2 C:maj\n1 3 D:min\n")
        self.annotations("002")
        dataset = self.dataset(works=("001", "002"), starts=(8, 8), splits=("train", "validation"))
        report = align_chords(dataset, self.shared, self.root / "aligned")
        sidecar = load_chord_sidecar(dataset, report["outputs"]["chords"])
        np.testing.assert_array_equal(sidecar["row_status"], ["source_error", "aligned"])
        self.assertTrue(np.all(np.isfinite(sidecar["cell_seconds"][0])))
        self.assertFalse(np.any(sidecar["chord_known"][0]))
        self.assertTrue(np.all(sidecar["chord_labels"][0] == ""))
        self.assertEqual(report["failures"][0]["reason"], "overlapping_annotation")

    def test_bad_midi_retains_row_with_nan_times(self):
        path = self.midi()
        path.write_bytes(b"broken midi")
        dataset = self.dataset()
        report = align_chords(dataset, self.shared, self.root / "aligned")
        sidecar = load_chord_sidecar(dataset, report["outputs"]["chords"])
        self.assertTrue(np.all(np.isnan(sidecar["cell_seconds"])))
        self.assertEqual(sidecar["row_status"].tolist(), ["source_error"])

    def test_partial_coverage_retains_N_and_does_not_carry_last_chord(self):
        self.midi()
        self.annotations(text="0 0.25 N\n0.5 0.75 A:min/5\n")
        dataset = self.dataset(starts=(0,))
        report = align_chords(dataset, self.shared, self.root / "aligned")
        sidecar = load_chord_sidecar(dataset, report["outputs"]["chords"])
        self.assertEqual(sidecar["row_status"].tolist(), ["partial"])
        self.assertEqual(sidecar["chord_labels"][0].tolist(), ["N", "N", "", "", "A:min/5", "A:min/5", "", ""])
        self.assertEqual(report["no_chord_cells"], 2)
        self.assertEqual(report["uncovered_cells"], 4)

    def test_melody_target_contents_do_not_change_alignment(self):
        self.midi()
        self.annotations()
        dataset = self.dataset(token_value=0)
        first = align_chords(dataset, self.shared, self.root / "first")
        self.dataset(token_value=129)
        second = align_chords(dataset, self.shared, self.root / "second")
        with np.load(first["outputs"]["chords"]) as a, np.load(second["outputs"]["chords"]) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])

    def test_loader_rejects_identity_or_feature_mask_mismatch(self):
        self.midi()
        self.annotations()
        dataset = self.dataset()
        report = align_chords(dataset, self.shared, self.root / "aligned")
        with np.load(report["outputs"]["chords"]) as source:
            arrays = {key: np.array(source[key]) for key in source.files}
        arrays["start_cells"] += 1
        bad = self.root / "bad.npz"
        np.savez(bad, **arrays)
        with self.assertRaisesRegex(ValueError, "identity"):
            load_chord_sidecar(dataset, bad)
        arrays["start_cells"] -= 1
        arrays["chord_features"][0, 0, 37] = 0
        np.savez(bad, **arrays)
        with self.assertRaisesRegex(ValueError, "known mask"):
            load_chord_sidecar(dataset, bad)

    def test_output_cannot_overwrite_raw_or_input_directory(self):
        self.midi()
        self.annotations()
        dataset = self.dataset()
        for destination in (self.shared / "raw" / "new", dataset.parent):
            with self.assertRaises(ValueError):
                align_chords(dataset, self.shared, destination)

    def test_unsupported_label_is_covered_but_feature_unknown_and_reported(self):
        self.midi()
        self.annotations(text="0 100 C:unsupported/slash\n")
        dataset = self.dataset()
        report = align_chords(dataset, self.shared, self.root / "aligned")
        sidecar = load_chord_sidecar(dataset, report["outputs"]["chords"])
        self.assertEqual(sidecar["row_status"].tolist(), ["aligned"])
        self.assertTrue(np.all(sidecar["chord_known"]))
        self.assertFalse(np.any(sidecar["chord_feature_known"]))
        self.assertEqual(sidecar["chord_labels"][0, 0], "C:unsupported/slash")
        self.assertEqual(report["unsupported_labels"]["C:unsupported/slash"]["cells"], 8)

    def test_unexpected_source_error_is_reported_with_traceback(self):
        self.midi()
        self.annotations()
        dataset = self.dataset()
        with patch("tri.data.chords.read_chord_intervals", side_effect=RuntimeError("forced implementation bug")):
            report = align_chords(dataset, self.shared, self.root / "aligned")
        self.assertEqual(report["status"], "needs_attention")
        self.assertEqual(report["unexpected_errors"], 1)
        self.assertIn("traceback", report["failures"][0])


if __name__ == "__main__":
    unittest.main()
