"""Correctness tests for supervised model."""

import torch
import torch.nn.functional as F

from torch_ref.models.supervised import SUPERVISED_DATA, SupervisedModel, train_supervised_epoch
from torch_ref.training.losses import nll_loss
from torch_ref.training.runner import get_dtype


class TestSupervised:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over 100 epochs."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        # Match SUPERVISED_DATA's dtype (= get_dtype(), F64 default).
        # Without the cast, nn.Linear's default-F32 weight mismatches
        # the F64 input tensors. scripts/supervised.py uses the same
        # `.to(dtype=get_dtype())` pattern.
        model = SupervisedModel().to(dtype=get_dtype())
        data = SUPERVISED_DATA
        lr = 0.03

        # Measure initial loss
        with torch.no_grad():
            losses = torch.stack([nll_loss(F.log_softmax(model(x), dim=-1), y) for x, y in data])
            initial_loss = losses.mean().item()

        # Train 100 epochs
        for _ in range(100):
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            train_supervised_epoch(model, data, optimizer)

        # Measure final loss
        with torch.no_grad():
            losses = torch.stack([nll_loss(F.log_softmax(model(x), dim=-1), y) for x, y in data])
            final_loss = losses.mean().item()

        assert final_loss < initial_loss

    def test_converges(self) -> None:
        """After 1000 epochs, loss should be low and predictions correct.

        SGD trajectory on this 5-point / 2-feature / 3-class dataset
        (vanilla SGD at lr=0.03, matching scripts/supervised.py), under
        the multiclass NLL loss (corrected from BCE-with-logits on
        2026-06-14 — argmax over 3 mutually-exclusive classes is a
        multiclass problem, so softmax-coupled NLL is the right loss):

          1000 epochs: loss=0.1362, 5/5 correct
          5000 epochs: loss=0.0599, 5/5 correct

        NLL converges 5/5 at 1000 epochs — the proper multiclass loss
        breaks out of the partial-fit minimum the old BCE got stuck in
        (BCE needed 5000 epochs for 5/5). So the budget drops back to
        1000 (matching the script default), with `< 0.3` headroom.
        """
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
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
