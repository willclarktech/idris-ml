"""Correctness tests for character-level GPT."""

import random

import torch

from torch_ref.models.gpt import (
    CORPUS_INDICES,
    evaluate_bpc,
    generate_text,
    train_gpt,
)

GPT_KWARGS = dict(
    seed=42,
    seq_len=32,
    d_model=32,
    num_heads=4,
    num_blocks=1,
    batch_size=16,
)


class TestGpt:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over training."""
        _, history = train_gpt(epochs=200, **GPT_KWARGS)
        early_avg = sum(history[:50]) / 50
        late_avg = sum(history[-50:]) / 50
        assert late_avg < early_avg, f"Expected loss to decrease: {early_avg:.3f} -> {late_avg:.3f}"

    def test_generates_text(self) -> None:
        """Trained model should generate non-trivial text."""
        model, _ = train_gpt(epochs=500, **GPT_KWARGS)
        text = generate_text(model, "the ", length=50, temperature=0.8)
        assert len(text) == 54  # seed (4) + generated (50)
        assert " " in text[4:]  # should contain real words

    def test_bpc_reasonable(self) -> None:
        """BPC should be below random baseline (log2(36) = 5.17)."""
        torch.manual_seed(42)
        random.seed(42)
        model, _ = train_gpt(epochs=500, **GPT_KWARGS)
        bpc = evaluate_bpc(model, CORPUS_INDICES, seq_len=32, n_samples=20)
        assert bpc < 4.0, f"BPC {bpc:.3f} >= 4.0 (random = 5.17)"
