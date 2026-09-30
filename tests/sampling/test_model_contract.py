import math

import pytest
import torch

from tri.errors import InvalidSpecification
from tri.models.grid import GridConfig, GridDenoiser, masked_diffusion_loss, two_gap_mask


def test_loss_uses_eligible_denominator_and_empty_mask_contributes_zero():
    logits = torch.zeros((2, 4, 130), requires_grad=True)
    clean = torch.zeros((2, 4), dtype=torch.long)
    eligible = torch.ones((2, 4), dtype=torch.bool)
    masked = torch.tensor([[True, False, False, False], [False] * 4])
    result = masked_diffusion_loss(logits, clean, eligible, masked, torch.tensor([0.5, 0.25]))
    assert float(result.detach()) == pytest.approx(math.log(130) / 4)
    result.backward()
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0


def test_mask_input_vocab_and_fixed_context_roles():
    torch.manual_seed(7)
    model = GridDenoiser(GridConfig(length=16, hidden=16, layers=1, heads=2)).eval()
    tokens = torch.full((2, 16), 62, dtype=torch.long)
    editable = two_gap_mask(2, 16, "cpu", randomize=False)
    tokens[editable] = 130
    assert not editable[:, 0].any() and not editable[:, -1].any()
    output = model(tokens, torch.tensor([0.3, 0.8]), editable)
    assert output.shape == (2, 16, 130)
    assert torch.isfinite(output).all()


def test_loss_rejects_hidden_invalid_masks_or_targets():
    logits = torch.zeros((1, 2, 130))
    clean = torch.tensor([[0, 130]])
    eligible = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(InvalidSpecification):
        masked_diffusion_loss(logits, clean, eligible, eligible, torch.ones(1))
