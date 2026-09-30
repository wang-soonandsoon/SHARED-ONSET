"""Check the stochastic loss normalization independently of a training run."""

from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from tri.errors import InvalidSpecification
from tri.models.grid import masked_diffusion_loss
from tri.models.train import train_smoke


class TrainingObjectiveTests(unittest.TestCase):
    def test_exact_expectation_uses_eligible_count_not_realized_masks(self):
        logits = torch.linspace(-2, 2, 6 * 130, dtype=torch.float64).reshape(1, 6, 130)
        clean = torch.tensor([[0, 62, 1, 66, 0, 62]], dtype=torch.long)
        editable = torch.tensor([[False, True, True, False, False, True]])
        positions = (1, 2, 5)
        ce = F.cross_entropy(logits.transpose(1, 2), clean, reduction="none")
        expected = float(ce[editable].mean())
        for probability in (0.13, 0.5, 0.91):
            total = 0.0
            for choices in product((False, True), repeat=3):
                mask = torch.zeros_like(editable)
                for position, selected in zip(positions, choices):
                    mask[0, position] = selected
                mass = probability ** sum(choices) * (1 - probability) ** (3 - sum(choices))
                loss = masked_diffusion_loss(logits, clean, editable, mask,
                                             torch.tensor([probability], dtype=torch.float64))
                total += mass * float(loss)
            self.assertAlmostEqual(total, expected, places=12)

    def test_empty_realized_mask_has_zero_loss_and_gradient(self):
        logits = torch.zeros((1, 4, 130), requires_grad=True)
        clean = torch.tensor([[0, 62, 1, 0]], dtype=torch.long)
        editable = torch.tensor([[False, True, True, False]])
        loss = masked_diffusion_loss(logits, clean, editable, torch.zeros_like(editable), torch.tensor([0.2]))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(int(torch.count_nonzero(logits.grad)), 0)

    def test_validation_alias_still_checks_work_level_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.npz"
            np.savez(path, tokens=np.zeros((2, 8), dtype=np.int16),
                     splits=np.array(["train", "validation"]), work_ids=np.array(["001", "001"]))
            with self.assertRaisesRegex(InvalidSpecification, "split leakage"):
                train_smoke(path, Path(directory) / "output", steps=1)


if __name__ == "__main__":
    unittest.main()
