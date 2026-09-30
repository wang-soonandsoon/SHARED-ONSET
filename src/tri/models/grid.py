"""Time-conditioned grid denoiser with optional external chord features.

condition_dim=0 preserves the original model/checkpoint structure. Conditioned
models accept aligned external features; hidden melody targets are never part
of the generation provider's interface.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from tri.errors import InvalidSpecification

REST, HOLD, MASK, PAD = 0, 1, 130, 131
OUTPUT_VOCAB = 130


@dataclass(frozen=True)
class GridConfig:
    length: int = 32
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    condition_dim: int = 0

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (self.length, self.hidden, self.layers, self.heads)):
            raise InvalidSpecification("positive integer grid dimensions required")
        if self.hidden % self.heads or not 0 <= self.dropout < 1:
            raise InvalidSpecification("invalid head count or dropout")
        if not isinstance(self.condition_dim, int) or isinstance(self.condition_dim, bool) or self.condition_dim < 0:
            raise InvalidSpecification("condition_dim must be a nonnegative integer")


class GridDenoiser(nn.Module):
    def __init__(self, config: GridConfig):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(132, config.hidden)
        self.position = nn.Embedding(config.length, config.hidden)
        self.role = nn.Embedding(2, config.hidden)
        self.time = nn.Sequential(nn.Linear(1, config.hidden), nn.SiLU(), nn.Linear(config.hidden, config.hidden))
        self.condition = nn.Linear(config.condition_dim, config.hidden, bias=False) if config.condition_dim else None
        layer = nn.TransformerEncoderLayer(config.hidden, config.heads, config.hidden * 4,
                                           config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(config.hidden), nn.Linear(config.hidden, OUTPUT_VOCAB))

    def forward(self, tokens: torch.Tensor, noise: torch.Tensor, editable: torch.Tensor,
                condition: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.config.length or editable.shape != tokens.shape or noise.shape != tokens.shape[:1]:
            raise InvalidSpecification("expected tokens/editable [B,L], noise [B]")
        if tokens.dtype != torch.long or editable.dtype != torch.bool:
            raise InvalidSpecification("tokens must be int64 and editable must be boolean")
        if bool(((tokens < 0) | (tokens > PAD)).any()) or bool(((noise < 0) | (noise > 1) | ~torch.isfinite(noise)).any()):
            raise InvalidSpecification("token or noise outside supported range")
        if self.config.condition_dim:
            if not isinstance(condition, torch.Tensor) or condition.shape != (*tokens.shape, self.config.condition_dim):
                raise InvalidSpecification(f"conditioned model requires condition [B,L,{self.config.condition_dim}]")
            if not condition.is_floating_point() or not bool(torch.isfinite(condition).all()):
                raise InvalidSpecification("condition must contain finite floating-point external features")
            if condition.device != tokens.device:
                raise InvalidSpecification("condition and tokens must be on the same device")
        elif condition is not None:
            raise InvalidSpecification("unconditioned model does not accept external features")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(positions)[None] + self.role(editable.long())
        x = x + self.time(noise.to(x.dtype)[:, None])[:, None]
        if self.condition is not None:
            x = x + self.condition(condition.to(x.dtype))
        return self.output(self.encoder(x, src_key_padding_mask=(tokens == PAD)))


def masked_diffusion_loss(logits: torch.Tensor, clean: torch.Tensor,
                          editable: torch.Tensor, masked: torch.Tensor,
                          mask_probability: torch.Tensor) -> torch.Tensor:
    """Linear m(u)=u loss; mean over ELIGIBLE positions, not realized masks.

    u~Uniform(0,1), mask independently inside a preselected editable region.
    Empty realized masks contribute zero; do not redraw them or divide by
    their random count. MASK/PAD are input-only and cannot be clean targets.
    """
    if logits.shape != (*clean.shape, OUTPUT_VOCAB) or editable.shape != clean.shape or masked.shape != clean.shape:
        raise InvalidSpecification("loss tensor shapes disagree")
    if editable.dtype != torch.bool or masked.dtype != torch.bool or clean.dtype != torch.long:
        raise InvalidSpecification("loss requires boolean masks and int64 targets")
    if mask_probability.shape != clean.shape[:1] or bool((~torch.isfinite(mask_probability) | (mask_probability <= 0) | (mask_probability > 1)).any()):
        raise InvalidSpecification("mask probabilities must be finite in (0,1]")
    if bool((masked & ~editable).any()) or bool((editable.sum(-1) == 0).any()):
        raise InvalidSpecification("masked positions must be editable and each example needs an editable region")
    if bool(((clean < 0) | (clean >= OUTPUT_VOCAB)).any()):
        raise InvalidSpecification("clean targets cannot contain MASK/PAD")
    ce = F.cross_entropy(logits.transpose(1, 2), clean, reduction="none")
    per_example = (ce * masked).sum(-1) / editable.sum(-1) / mask_probability
    return per_example.mean()


def two_gap_mask(batch: int, length: int, device: torch.device | str,
                 *, generator: torch.Generator | None = None, randomize: bool = True,
                 min_gap_cells: int | None = None, max_gap_cells: int | None = None) -> torch.Tensor:
    if length < 8:
        raise InvalidSpecification("two-gap bootstrap needs at least eight cells")
    width = max(1, min(4, length // 8))
    result = torch.zeros((batch, length), dtype=torch.bool, device=device)
    if min_gap_cells is not None or max_gap_cells is not None:
        minimum = 3 if min_gap_cells is None else min_gap_cells
        maximum = 8 if max_gap_cells is None else max_gap_cells
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (minimum, maximum)) or minimum > maximum:
            raise InvalidSpecification("gap bounds must be positive integers with minimum <= maximum")
        # Keep the first/last and center cells visible; shorten gaps for tiny L.
        capacity = min(length // 2 - 1, length - length // 2 - 2)
        high_width = min(maximum, capacity)
        low_width = min(minimum, high_width)
        for b in range(batch):
            selected_width = int(torch.randint(low_width, high_width + 1, (), device=device, generator=generator)) if randomize else low_width
            for low, high in [(1, length // 2 - selected_width),
                              (length // 2 + 1, length - selected_width - 1)]:
                start = int(torch.randint(low, high + 1, (), device=device, generator=generator)) if randomize else (low + high) // 2
                result[b, start:start + selected_width] = True
        return result
    for b in range(batch):
        for low, high in [(1, length // 2 - width), (length // 2, length - width - 1)]:
            start = int(torch.randint(low, high + 1, (), device=device, generator=generator)) if randomize else (low + high) // 2
            result[b, start:start + width] = True
    return result


class ModelProbabilityProvider:
    """Partial tokens, original edit roles, time, and external chord features."""

    def __init__(self, model: GridDenoiser, editable_positions: tuple[int, ...], condition=None):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.editable_positions = tuple(editable_positions)
        self.condition = None
        if model.config.condition_dim:
            if condition is None:
                raise InvalidSpecification("conditioned provider requires external chord features")
            try:
                features = torch.as_tensor(condition, device=self.device)
            except (TypeError, ValueError) as exc:
                raise InvalidSpecification("invalid external chord features") from exc
            if features.is_complex():
                raise InvalidSpecification("external chord features must be real")
            features = features.to(torch.float32)
            if features.shape != (model.config.length, model.config.condition_dim) or not bool(torch.isfinite(features).all()):
                raise InvalidSpecification(f"provider condition must be finite [L,{model.config.condition_dim}]")
            self.condition = features.detach().clone()[None]
        elif condition is not None:
            raise InvalidSpecification("unconditioned provider does not accept chord features")

    @torch.no_grad()
    def __call__(self, tokens: tuple[int | None, ...], noise: float):
        x = torch.tensor([[MASK if v is None else v for v in tokens]], dtype=torch.long, device=self.device)
        editable = torch.zeros_like(x, dtype=torch.bool)
        editable[:, self.editable_positions] = True
        logits = self.model(x, torch.tensor([noise], device=self.device), editable, self.condition)
        # Normalize the full neural vocabulary in float64 before any rule
        # cropping; this also makes reference path probabilities consistent.
        return logits.double().log_softmax(-1)[0].cpu().numpy()
