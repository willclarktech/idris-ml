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
        """After 1000 epochs, loss should be very low and predictions correct.

        Known-latent failure 2026-06-04: the model converges to
        ~0.303 at 1000 epochs (right at the `< 0.3` boundary) and to
        ~0.27 at 1500 epochs (passes the loss check), but mispredicts
        the 5th data point (target class 0, predicted class 2) in
        either case — the 5-point dataset is small enough that the
        Linear(2→3) model can't perfectly separate all classes from
        only 2 features. Pre-existing issue (test was failing at the
        pre-session state too with dtype mismatch which masked this
        deeper issue). Filed for a separate fix (loosen the per-
        prediction assertion or fix the data/model).
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
        for _ in range(1000):
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
