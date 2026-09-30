"""Small executable demonstrations, with their limitations saved alongside outputs."""
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import MusicSpec, CountRule, verify_music
from tri.domain.compiler import compile_music
from tri.inference.exact import ExactInference
from tri.sampling.direct import direct_decode
from tri.request_builder import build_two_gap_request, sounding_pitches


def _write_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def core_demo(output_dir, *, seed=20260912):
    """Two unknown shared onset patterns with different pitch-class rules."""
    spec = MusicSpec(length=6, pitches=(60, 64), observed={0: 0, 3: 0},
                     equal_onsets=((1, 4), (2, 5)),
                     onset_counts=(CountRule((1, 2), 1), CountRule((4, 5), 1)),
                     pitch_classes={1: (0,), 2: (0,), 4: (4,), 5: (4,)}, motion_cost=0.03)

    def provider(tokens, noise):
        logits = np.full((6, 130), -9.0)
        context = sum(i + 1 for i, value in enumerate(tokens) if value is not None and value >= 2)
        logits[:, 0] = 0.1 + noise
        logits[:, 1] = 0.7
        logits[:, 62] = 0.5 + 0.02 * context
        logits[:, 66] = 0.3 + noise
        return logits - logsumexp(logits, axis=1, keepdims=True)

    initial = tuple(spec.observed.get(i) for i in range(6))
    q = provider(initial, 1.0)
    engine = ExactInference(compile_music(spec, q))
    log_z = engine.log_partition()
    order_stats = dict(engine.last_stats)
    masses = []
    for assignment in product((0, 1, 62, 66), repeat=4):
        y = [0, assignment[0], assignment[1], 0, assignment[2], assignment[3]]
        checked = verify_music(y, spec)
        if checked.valid:
            masses.append(sum(q[i, y[i]] for i in (1, 2, 4, 5)) + checked.soft_score)
    oracle_z = float(logsumexp(masses))
    if not np.isclose(log_z, oracle_z, atol=1e-10, rtol=1e-10):
        raise RuntimeError("compiler/VE disagrees with independent original-token enumeration")
    decoded = direct_decode(spec, provider, steps=3, seed=seed, epsilon=0.25, track_path_weights=True)
    return _write_report({"scope": "six-cell exact relational core demonstration", "seed": seed,
                          "log_partition": log_z, "independent_log_partition": oracle_z,
                          "absolute_error": abs(log_z-oracle_z), "valid_original_sequences": len(masses),
                          "initial_inference_stats": order_stats, "decode": asdict(decoded),
                          "verified": verify_music(decoded.tokens, spec).valid,
                          "not_claimed": ["terminal-exact unweighted sampling", "music quality", "novel algorithmic speedup"]}, output_dir)


def music_demo(dataset, checkpoint, output_dir, *, seed=20260912, device="cpu", steps=4):
    from tri.data.midi import write_grid_midi, read_melody
    from tri.models.grid import ModelProbabilityProvider
    from tri.models.train import load_checkpoint

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset, allow_pickle=False) as data:
        # Training-split demonstration only; never select a request by test loss.
        available = np.flatnonzero(data["splits"].astype(str) == "train")
        if not len(available):
            raise ValueError("no training-split window available for demonstration")
        index = int(available[0])
        original = np.array(data["tokens"][index], dtype=np.int64)
        work_id, start = str(data["work_ids"][index]), int(data["start_cells"][index])
        raw_initial = int(data["initial_pitches"][index])
        initial = None if raw_initial < 0 else raw_initial
        meter = tuple(int(v) for v in data["time_signatures"][index])
    if len(original) < 16:
        raise ValueError("music demonstration requires at least 16 cells")
    spans = (tuple(range(len(original)//4, len(original)//4+3)),
             tuple(range(3*len(original)//4, 3*len(original)//4+3)))
    editable = tuple(i for span in spans for i in span)
    spec = build_two_gap_request(original, initial_pitch=initial, spans=spans)
    visible = dict(spec.observed)
    source_soundings = sounding_pitches(original, initial)
    anchors = dict(spec.fixed_soundings)
    model = load_checkpoint(checkpoint, device)
    if model.config.length != len(original):
        raise ValueError("checkpoint and window lengths differ")
    provider = ModelProbabilityProvider(model, editable)
    decoded = direct_decode(spec, provider, steps=steps, seed=seed, track_path_weights=True)
    tokens_equal = all(decoded.tokens[i] == int(original[i]) for i in visible)
    generated_soundings = sounding_pitches(decoded.tokens, initial)
    sounding_equal = all(generated_soundings[i] == source_soundings[i] for i in visible)
    if not tokens_equal or not sounding_equal:
        raise RuntimeError("fixed MIDI context changed")
    exports = {}
    for name, tokens in [("original", original), ("completion", decoded.tokens)]:
        exports[name] = write_grid_midi(tokens, output_dir / f"{name}.mid", initial_pitch=initial, time_signature=meter)
        restored = read_melody(output_dir / f"{name}.mid").tokens
        expected = np.asarray(tokens).copy()
        if expected[0] == 1:
            expected[0] = initial + 2  # explicit standalone left-boundary anchor
        if not np.array_equal(restored, expected):
            raise RuntimeError("MIDI output failed grid roundtrip")
    request = {"work_id": work_id, "source_start_cell": start, "source_split": "train",
               "editable_spans": spans, "shared_onsets": True, "onsets_per_span": 1,
               "working_pitches": list(spec.pitches), "initial_pitch": initial,
               "observed_tokens": visible, "fixed_soundings": anchors, "recorded_meter": meter}
    (output_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    return _write_report({"scope": "real-MIDI model-to-joint-decoder plumbing demonstration",
                          "request": request, "decode": asdict(decoded), "exports": exports,
                          "fixed_tokens_preserved": tokens_equal, "fixed_sounding_pitches_preserved": sounding_equal,
                          "independent_verifier_passed": verify_music(decoded.tokens, spec).valid,
                          "limitations": ["Standalone monophonic windows; source accompaniment is not exported.",
                                          "Training-split demonstration, not held-out music-quality evidence.",
                                          "No chord-label or accompaniment conditioner trained yet.",
                                          "Path weight recorded; no particle resampling or terminal-exact claim."]}, output_dir)
