"""Correctness tests for supervised model."""

import torch

from bench.models.supervised import SUPERVISED_DATA, SupervisedModel, train_supervised_epoch
from bench.training.losses import cross_entropy


class TestSupervised:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over 100 epochs."""
        torch.manual_seed(42)
        model = SupervisedModel()
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
        """After 1000 epochs, loss should be very low and predictions correct."""
        torch.manual_seed(42)
        model = SupervisedModel()
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
