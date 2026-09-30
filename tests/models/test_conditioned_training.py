import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tri.data.chords import encode_chord_features
from tri.errors import InvalidSpecification
from tri.models.grid import GridConfig, GridDenoiser, ModelProbabilityProvider, two_gap_mask
from tri.models.train import load_checkpoint
from tri.models.training import _EpochSampler, fit, load_training_inputs, validation_ce


@pytest.fixture(autouse=True)
def small_cpu_tests():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_inputs(root: Path, *, train=5, val=3, test=2, unknown_first=False):
    root.mkdir(parents=True, exist_ok=True)
    length, count = 16, train + val + test
    tokens = np.array([([62 + i % 4, 1, 0, 0] * 4) for i in range(count)], dtype=np.int64)
    work_ids = np.array([f"work{i:02d}" for i in range(count)])
    splits = np.array(["train"] * train + ["validation"] * val + ["test"] * test)
    identity = {"work_ids": work_ids, "start_cells": np.zeros(count, dtype=np.int64), "splits": splits}
    dataset = root / "windows.npz"
    np.savez(dataset, tokens=tokens, **identity)
    labels = np.full((count, length), "C:maj", dtype="U8")
    known = np.ones((count, length), dtype=bool)
    statuses = np.full(count, "aligned", dtype="U12")
    if unknown_first:
        labels[0] = ""
        known[0] = False
        statuses[0] = "source_error"
    features, feature_known = encode_chord_features(labels, known)
    chords = root / "chords.npz"
    np.savez(chords, **identity, chord_labels=labels, chord_known=known,
             cell_seconds=np.tile(np.arange(length + 1, dtype=float) * 0.125, (count, 1)),
             row_status=statuses, chord_features=features, chord_feature_known=feature_known)
    return dataset, chords


def rewrite_archive(path, update):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    update(arrays)
    np.savez(path, **arrays)


def tiny_config(dropout=0.2):
    return GridConfig(length=16, hidden=8, layers=1, heads=2, dropout=dropout, condition_dim=38)


def test_legacy_checkpoint_and_new_conditioned_provider(tmp_path):
    torch.manual_seed(100)
    legacy = GridDenoiser(GridConfig(length=16, hidden=8, layers=1, heads=2)).eval()
    assert not any(key.startswith("condition.") for key in legacy.state_dict())
    legacy_path = tmp_path / "legacy.pt"
    torch.save({"model_config": {"length": 16, "hidden": 8, "layers": 1, "heads": 2, "dropout": 0.0},
                "model_state": legacy.state_dict()}, legacy_path)
    restored = load_checkpoint(legacy_path)
    state = tuple([None, 62, 1, 0] * 4)
    assert np.array_equal(ModelProbabilityProvider(legacy, (0,))(state, 1.0),
                          ModelProbabilityProvider(restored, (0,))(state, 1.0))
    model = GridDenoiser(tiny_config()).eval()
    features, _ = encode_chord_features(np.full(16, "C:maj"))
    provider = ModelProbabilityProvider(model, (0, 4, 8, 12), condition=features)
    first = provider(state, 0.7)
    features[:] = 0
    assert np.array_equal(first, provider(state, 0.7))
    assert first.shape == (16, 130)
    assert np.isfinite(first).all()
    assert np.allclose(np.exp(first).sum(-1), 1.0, atol=1e-12)
    other = ModelProbabilityProvider(model, (0, 4, 8, 12), condition=features)
    assert not np.allclose(first, other(state, 0.7))


@pytest.mark.parametrize("condition", [None, torch.zeros(1, 16, 37), torch.zeros(1, 16, 38, dtype=torch.long), torch.full((1, 16, 38), float("nan"))])
def test_conditioned_model_requires_valid_features(condition):
    model = GridDenoiser(tiny_config())
    with pytest.raises(InvalidSpecification):
        model(torch.zeros((1, 16), dtype=torch.long), torch.ones(1), torch.ones((1, 16), dtype=torch.bool), condition)
    with pytest.raises(InvalidSpecification):
        ModelProbabilityProvider(model, (0,))


def test_config_and_conditioned_checkpoint_smoke(tmp_path):
    with pytest.raises(InvalidSpecification):
        GridConfig(condition_dim=-1)
    model = GridDenoiser(tiny_config())
    with pytest.raises(InvalidSpecification):
        ModelProbabilityProvider(model, (0,), condition=np.zeros((15, 38)))
    with pytest.raises(InvalidSpecification):
        ModelProbabilityProvider(model, (0,), condition=np.zeros((16, 38), dtype=complex))
    torch.save({"model_config": model.config.__dict__, "model_state": model.state_dict()}, tmp_path / "conditioned.pt")
    restored = load_checkpoint(tmp_path / "conditioned.pt")
    assert restored.config.condition_dim == 38
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_training_gap_widths_cover_three_through_eight_and_small_lengths():
    generator = torch.Generator().manual_seed(99)
    mask = two_gap_mask(200, 32, "cpu", generator=generator, min_gap_cells=3, max_gap_cells=8)
    widths = mask.sum(-1) // 2
    assert set(widths.tolist()) == set(range(3, 9))
    assert not mask[:, [0, 16, 31]].any()
    assert torch.equal(mask[:, :16].sum(-1), mask[:, 16:].sum(-1))
    tiny = two_gap_mask(5, 8, "cpu", generator=generator, min_gap_cells=3, max_gap_cells=8)
    assert not tiny[:, [0, 4, 7]].any()
    assert (tiny.sum(-1) == 4).all()


def test_epoch_sampler_uses_every_training_window_before_reshuffling():
    ids = torch.tensor([1, 3, 5, 7, 9])
    sampler = _EpochSampler(ids, torch.Generator().manual_seed(22))
    first_epoch = torch.cat([sampler.take(3), sampler.take(2)])
    assert torch.equal(first_epoch.sort().values, ids)
    sampler.take(3)
    assert sampler.epoch == 1


def test_inputs_keep_all_training_rows_and_unknown_chords(tmp_path):
    dataset, chords = make_inputs(tmp_path, train=40, val=3, test=2, unknown_first=True)
    inputs = load_training_inputs(dataset, chords)
    assert len(inputs["train_ids"]) == 40
    assert len(inputs["validation_ids"]) == 3
    assert not inputs["known"][0].any()
    assert not inputs["condition"][0].any()


def test_sidecar_identity_and_nonfinite_features_rejected(tmp_path):
    dataset, chords = make_inputs(tmp_path / "identity")
    rewrite_archive(chords, lambda arrays: arrays["start_cells"].__setitem__(0, 16))
    with pytest.raises(ValueError, match="identity"):
        load_training_inputs(dataset, chords)
    dataset, chords = make_inputs(tmp_path / "nonfinite")
    rewrite_archive(chords, lambda arrays: arrays["chord_features"].__setitem__((0, 0, 0), np.nan))
    with pytest.raises(ValueError, match="finite"):
        load_training_inputs(dataset, chords)


def test_work_leakage_and_no_validation_fallback(tmp_path):
    dataset, chords = make_inputs(tmp_path / "leak")
    for path in (dataset, chords):
        def change(arrays):
            arrays["work_ids"][5] = arrays["work_ids"][0]
            arrays["start_cells"][5] = 16
        rewrite_archive(path, change)
    with pytest.raises(InvalidSpecification, match="leaks"):
        load_training_inputs(dataset, chords)
    dataset, chords = make_inputs(tmp_path / "noval", val=0)
    with pytest.raises(InvalidSpecification, match="no test fallback"):
        load_training_inputs(dataset, chords)


def test_validation_uses_only_validation_ids_and_eligible_position_denominator(tmp_path):
    dataset, chords = make_inputs(tmp_path)
    inputs = load_training_inputs(dataset, chords)
    model = GridDenoiser(tiny_config(0.0))
    for parameter in model.parameters():
        parameter.data.zero_()
    # Distinct test targets would trigger target-range failure if ever evaluated.
    inputs["tokens"][np.flatnonzero(inputs["splits"] == "test")] = 130
    first = validation_ce(model, inputs, 2, torch.device("cpu"))
    second = validation_ce(model, inputs, 3, torch.device("cpu"))
    assert first == pytest.approx(np.log(130), abs=1e-6)
    assert second == pytest.approx(first, abs=1e-6)


def test_resume_exact_optimization_trajectory_and_validation_selection(tmp_path):
    dataset, chords = make_inputs(tmp_path / "data", unknown_first=True)
    kwargs = dict(batch_size=3, seed=171, device="cpu", config=tiny_config(),
                  lr=1e-3, eval_every=2, save_every=2, max_gap_cells=4)
    full = fit(dataset, chords, tmp_path / "full", steps=6, **kwargs)
    fit(dataset, chords, tmp_path / "split", steps=3, **kwargs)
    resumed = fit(dataset, chords, tmp_path / "split", steps=6, resume=True, **kwargs)
    a = torch.load(tmp_path / "full" / "last.pt", weights_only=True)
    b = torch.load(tmp_path / "split" / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 6
    for name, value in a["model_state"].items():
        assert torch.equal(value, b["model_state"][name]), name
    for name in ("data_rng", "torch_cpu_rng"):
        assert torch.equal(a[name], b[name])
    assert a["sampler"]["cursor"] == b["sampler"]["cursor"]
    assert torch.equal(a["sampler"]["order"], b["sampler"]["order"])
    for key, state in a["optimizer_state"]["state"].items():
        for name, value in state.items():
            other = b["optimizer_state"]["state"][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    full_rows = [json.loads(line) for line in (tmp_path / "full" / "training.jsonl").read_text().splitlines()]
    split_rows = [json.loads(line) for line in (tmp_path / "split" / "training.jsonl").read_text().splitlines()]
    assert [row["step"] for row in split_rows] == list(range(1, 7))
    assert [row["loss"] for row in full_rows] == [row["loss"] for row in split_rows]
    assert [row["masked_positions"] for row in full_rows] == [row["masked_positions"] for row in split_rows]
    assert resumed["train_windows"] == full["train_windows"] == 5
    assert resumed["training_chord_known_fraction"] == pytest.approx(0.8)
    assert resumed["test_targets_used"] is False
    assert resumed["best_validation_ce"] == min([resumed["initial_validation_ce"]] + [row["validation_ce"] for row in split_rows if "validation_ce" in row])
    assert load_checkpoint(tmp_path / "split" / "best.pt").config.condition_dim == 38
    status = json.loads((tmp_path / "split" / "status.json").read_text())
    assert status["status"] == "completed" and status["step"] == 6


def test_resume_rejects_input_changes_and_accidental_overwrite(tmp_path):
    dataset, chords = make_inputs(tmp_path / "data")
    kwargs = dict(steps=1, batch_size=2, config=tiny_config(0.0), eval_every=1, save_every=1)
    fit(dataset, chords, tmp_path / "run", **kwargs)
    with pytest.raises(InvalidSpecification, match="resume=True"):
        fit(dataset, chords, tmp_path / "run", **kwargs)
    dataset.touch()
    with pytest.raises(InvalidSpecification, match="inputs differ"):
        fit(dataset, chords, tmp_path / "run", resume=True, **kwargs)


def test_nonfinite_training_marks_failed_and_retains_durable_checkpoint(tmp_path, monkeypatch):
    import tri.models.training as training
    dataset, chords = make_inputs(tmp_path / "data")
    monkeypatch.setattr(training, "masked_diffusion_loss", lambda *args: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(RuntimeError, match="Nonfinite training"):
        fit(dataset, chords, tmp_path / "run", steps=1, config=tiny_config())
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["step"] == 0
    assert torch.load(tmp_path / "run" / "last.pt", weights_only=True)["step"] == 0


def test_resume_repairs_interrupted_best_write_without_requiring_log(tmp_path, monkeypatch):
    import tri.models.training as training
    dataset, chords = make_inputs(tmp_path / "data")
    save = training._atomic_checkpoint
    def interrupted_save(path, value):
        if path.name == "best.pt":
            raise OSError("interrupted between last and best writes")
        save(path, value)
    monkeypatch.setattr(training, "_atomic_checkpoint", interrupted_save)
    kwargs = dict(batch_size=2, config=tiny_config(0.0), eval_every=1, save_every=1)
    with pytest.raises(OSError, match="interrupted"):
        fit(dataset, chords, tmp_path / "run", steps=1, **kwargs)
    assert (tmp_path / "run" / "last.pt").exists()
    assert not (tmp_path / "run" / "best.pt").exists()
    monkeypatch.setattr(training, "_atomic_checkpoint", save)
    report = fit(dataset, chords, tmp_path / "run", steps=1, resume=True, **kwargs)
    assert report["status"] == "completed"
    assert (tmp_path / "run" / "best.pt").exists()
    (tmp_path / "run" / "training.jsonl").unlink()
    report = fit(dataset, chords, tmp_path / "run", steps=2, resume=True, **kwargs)
    assert report["status"] == "completed"
    assert report["log_history_complete"] is False
    assert [json.loads(line)["step"] for line in (tmp_path / "run" / "training.jsonl").read_text().splitlines()] == [2]
