"""Correctness tests for 1D sequence classification."""

import random

import torch

from torch_ref.models.seq_classify import SeqClassifyCNN, evaluate, train_epoch


def test_loss_decreases() -> None:
    torch.manual_seed(42)
    random.seed(42)
    model = SeqClassifyCNN().float()  # F32 (2026-06-04 dtype flip)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for _ in range(200):
        loss = train_epoch(model, optimizer)
        losses.append(loss)

    assert losses[-1] < losses[0], f"Loss did not decrease: first={losses[0]}, last={losses[-1]}"


def test_accuracy_above_chance() -> None:
    torch.manual_seed(42)
    random.seed(42)
    model = SeqClassifyCNN().float()  # F32 (2026-06-04 dtype flip)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(300):
        train_epoch(model, optimizer)

    accuracy = evaluate(model, 300)
    assert accuracy > 0.5, f"Accuracy {accuracy:.3f} not above chance (0.33)"
