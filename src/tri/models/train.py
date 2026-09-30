"""Reproducible, bounded training smoke; not a full scientific experiment."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from tri.errors import InvalidSpecification
from tri.models.grid import GridConfig, GridDenoiser, MASK, masked_diffusion_loss, two_gap_mask


def load_checkpoint(path: str | Path, device: str = "cpu") -> GridDenoiser:
    saved = torch.load(path, map_location=device, weights_only=True)
    model = GridDenoiser(GridConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model_state"])
    return model.eval()


def train_smoke(dataset: str | Path, output_dir: str | Path, *, steps: int = 120,
                batch_size: int = 16, seed: int = 20260912, device: str = "cpu",
                config: GridConfig | None = None) -> dict:
    if steps < 1 or batch_size < 1:
        raise InvalidSpecification("steps and batch_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset, allow_pickle=False) as data:
        tokens = np.array(data["tokens"], dtype=np.int64)
        splits = np.asarray(data["splits"]).astype(str)
        splits = np.where(splits == "validation", "val", splits)
        work_ids = np.asarray(data["work_ids"]).astype(str)
    if tokens.ndim != 2 or len(tokens) != len(splits) or len(tokens) != len(work_ids):
        raise InvalidSpecification("invalid windows dataset")
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(work_ids[splits == left]) & set(work_ids[splits == right]):
            raise InvalidSpecification("work-level split leakage")
    train_ids = np.flatnonzero(splits == "train")
    val_ids = np.flatnonzero(splits == "val")
    if not len(train_ids):
        raise InvalidSpecification("no training windows")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Intentionally a small overfit check; do not call this full-data training.
    selected = rng.permutation(train_ids)[:32]
    train_tokens = torch.as_tensor(tokens[selected], dtype=torch.long, device=device)
    val_tokens = torch.as_tensor(tokens[val_ids[:64]], dtype=torch.long, device=device)
    config = config or GridConfig(length=tokens.shape[1])
    if config.length != tokens.shape[1]:
        raise InvalidSpecification("model length and dataset length differ")
    model = GridDenoiser(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    @torch.no_grad()
    def fixed_loss(clean):
        if not len(clean):
            return None
        model.eval()
        editable = two_gap_mask(len(clean), config.length, device, randomize=False)
        noisy = clean.clone().masked_fill(editable, MASK)
        logits = model(noisy, torch.ones(len(clean), device=device), editable)
        return float(F.cross_entropy(logits[editable], clean[editable]))

    initial = fixed_loss(train_tokens)
    losses = []
    started = time.perf_counter()
    for step in range(steps):
        model.train()
        ids = torch.randint(len(train_tokens), (batch_size,), device=device)
        clean = train_tokens[ids]
        editable = two_gap_mask(batch_size, config.length, device)
        # 1-rand is in (0,1]; no truncation, forced mask, or rejection redraw.
        u = 1.0 - torch.rand(batch_size, device=device)
        masked = (torch.rand(clean.shape, device=device) < u[:, None]) & editable
        noisy = clean.clone().masked_fill(masked, MASK)
        logits = model(noisy, u, editable)
        loss = masked_diffusion_loss(logits, clean, editable, masked, u)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append({"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(grad),
                       "masked_positions": int(masked.sum())})
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = fixed_loss(train_tokens)
    validation = fixed_loss(val_tokens)
    checkpoint = output_dir / "checkpoint.pt"
    torch.save({"model_config": asdict(config), "model_state": model.state_dict(),
                "seed": seed, "steps": steps, "scope": "32-window overfit smoke"}, checkpoint)
    restored = load_checkpoint(checkpoint, device)
    with torch.no_grad():
        clean = train_tokens[:2]
        editable = two_gap_mask(len(clean), config.length, device, randomize=False)
        noisy = clean.clone().masked_fill(editable, MASK)
        noise = torch.ones(len(clean), device=device)
        before, after = model.eval()(noisy, noise, editable), restored(noisy, noise, editable)
        restore_error = float((before - after).abs().max())
    report = {
        "scope": "real-MIDI beat-grid bootstrap, 32 training windows maximum",
        "not_validated": ["music quality", "full-data generalization", "chord/accompaniment conditioning"],
        "dataset": str(Path(dataset).resolve()), "seed": seed, "device": str(device),
        "model_config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
        "steps": steps, "batch_size": batch_size, "training_windows": len(selected),
        "training_work_ids": sorted(set(work_ids[selected])),
        "validation_windows": len(val_tokens), "validation_work_ids": sorted(set(work_ids[val_ids[:64]])),
        "initial_fixed_mask_ce": initial, "final_fixed_mask_ce": final,
        "validation_fixed_mask_ce": validation, "checkpoint_restore_max_abs_error": restore_error,
        "elapsed_seconds": elapsed, "checkpoint": str(checkpoint.resolve()),
    }
    (output_dir / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in losses))
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
