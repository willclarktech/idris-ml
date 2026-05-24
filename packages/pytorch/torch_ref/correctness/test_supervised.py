"""Correctness tests for supervised model."""

import torch

from torch_ref.models.supervised import SUPERVISED_DATA, SupervisedModel, train_supervised_epoch
from torch_ref.training.losses import cross_entropy
from torch_ref.training.runner import get_dtype


class TestSupervised:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over 100 epochs."""
        torch.manual_seed(42)
        # Match SUPERVISED_DATA's dtype (= get_dtype(), F64 default).
        # Without the cast, nn.Linear's default-F32 weight mismatches
        # the F64 input tensors. scripts/supervised.py uses the same
        # `.to(dtype=get_dtype())` pattern.
        model = SupervisedModel().to(dtype=get_dtype())
        data = SUPERVISED_DATA
        lr = 0.03

        # Measure initial loss
        with torch.no_grad():
            losses = torch.stack([cross_entropy(model(x), y) for x, y in data])
            initial_loss = losses.mean().item()

        # Train 100 epochs
        for _ in range(100):
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            train_supervised_epoch(model, data, optimizer)

        # Measure final loss
        with torch.no_grad():
            losses = torch.stack([cross_entropy(model(x), y) for x, y in data])
            final_loss = losses.mean().item()

        assert final_loss < initial_loss

    def test_converges(self) -> None:
        """After 5000 epochs, loss should be very low and predictions correct.

        Was 1000 epochs against `< 0.3`. The actual SGD convergence
        trajectory on this 5-point / 2-feature / 3-class dataset
        (matching the script's vanilla SGD optimizer at lr=0.03):

          1000 epochs: loss=0.303, 3/5 correct (5th point wrong)
          5000 epochs: loss=0.195, 5/5 correct
          10000 epochs: loss=0.168, 5/5 correct

        The 1000-epoch budget was right at the `< 0.3` boundary
        (failing by 1%) AND mispredicting the 5th data point — the
        model needs more SGD steps to break out of a partial-fit
        local minimum where 4/5 points are well-classified and the
        5th is sacrificed. Bumped to 5000 epochs (still trivial wall
        clock, < 1s) for comfortable headroom. Alternative would be
        switching to Adam — converges 5/5 in 1000 epochs at lr=0.01 —
        but the script (scripts/supervised.py:55) uses SGD, so the
        test stays on SGD for parity.

        This was a long-standing latent failure: the dtype mismatch
        (model F32 vs SUPERVISED_DATA F64, fixed earlier in the
        commit that added the .to(dtype=get_dtype()) cast) was
        masking it pre-2026-06-04.
        """
        torch.manual_seed(42)
        # Match SUPERVISED_DATA's dtype (= get_dtype(), F64 default).
        # Without the cast, nn.Linear's default-F32 weight mismatches
        # the F64 input tensors. scripts/supervised.py uses the same
        # `.to(dtype=get_dtype())` pattern.
        model = SupervisedModel().to(dtype=get_dtype())
        data = SUPERVISED_DATA
        lr = 0.03

        loss_val = 0.0
        for _ in range(5000):
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            loss_val = train_supervised_epoch(model, data, optimizer)

        assert loss_val < 0.3

        # Check predictions match targets
        with torch.no_grad():
            for x, y in data:
                pred = model(x)
                pred_class = pred.argmax().item()
                target_class = y.argmax().item()
                assert pred_class == target_class, (
                    f"Mismatch: pred={pred_class}, target={target_class}"
                )
