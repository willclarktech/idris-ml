"""Correctness tests for RNN model."""

import torch

from bench.models.rnn import RNN_DATA, LinearRNNCell, train_rnn_epoch


class TestRnn:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over 100 epochs."""
        torch.manual_seed(42)
        model = LinearRNNCell(1, 1)
        data = RNN_DATA
        lr = 0.03

        losses = []
        for _ in range(100):
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            loss_val = train_rnn_epoch(model, data, optimizer)
            losses.append(loss_val)

        assert losses[-1] < losses[0]

    def test_converges(self) -> None:
        """After 1000 epochs, loss should be low and pattern predicted."""
        torch.manual_seed(42)
        model = LinearRNNCell(1, 1)
        data = RNN_DATA
        lr = 0.03

        loss_val = 0.0
        for _ in range(1000):
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            loss_val = train_rnn_epoch(model, data, optimizer)

        assert loss_val < 0.5

        # Check pattern prediction on a sequence
        with torch.no_grad():
            model.reset_state()
            xs, ys = data[0]  # length 3: [0,1,0] → [1,0,1]
            predictions = []
            for x in xs:
                pred = model(x)
                predictions.append((pred > 0).float().item())
            # Should roughly match the [1, 0, 1] pattern
            targets = [y.item() for y in ys]
            correct = sum(1 for p, t in zip(predictions, targets, strict=True) if p == t)
            assert correct >= 2, f"Only {correct}/3 correct: {predictions} vs {targets}"
