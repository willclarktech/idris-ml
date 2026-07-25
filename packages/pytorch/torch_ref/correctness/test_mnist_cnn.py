"""Correctness tests for MNIST CNN."""

import torch

# evaluate/get_mnist_loaders/train_epoch carry bare-DataLoader params in the
# model file, so their imported types are partially unknown under strict mode.
from torch_ref.models.mnist_cnn import (
    MnistCNN,
    evaluate,  # pyright: ignore[reportUnknownVariableType]
    get_mnist_loaders,  # pyright: ignore[reportUnknownVariableType]
    train_epoch,  # pyright: ignore[reportUnknownVariableType]
)
from torch_ref.training.runner import get_dtype


def test_loss_decreases() -> None:
    """Training loss should decrease over 3 epochs."""
    torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
    model = MnistCNN().to(get_dtype())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader, _ = get_mnist_loaders(batch_size=64)  # pyright: ignore[reportUnknownVariableType]

    losses: list[float] = []
    for _ in range(3):
        loss = train_epoch(model, train_loader, optimizer)
        losses.append(loss)

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_accuracy_above_threshold() -> None:
    """After 3 epochs, test accuracy should exceed 95%."""
    torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
    model = MnistCNN().to(get_dtype())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader, test_loader = get_mnist_loaders(  # pyright: ignore[reportUnknownVariableType]
        batch_size=64
    )

    for _ in range(3):
        train_epoch(model, train_loader, optimizer)

    _, accuracy = evaluate(model, test_loader)
    assert accuracy > 0.95, f"Accuracy too low: {accuracy:.4f}"
