"""Correctness tests for NTM copy model."""

import math
import random

import pytest
import torch

from bench.data.copy_task import generate_copy_batch
from bench.models.ntm_copy import NtmCopyConfig, NtmCopyModel, train_ntm_copy_step
from bench.training.curriculum import Stage, run_curriculum
from bench.training.losses import nll_loss


class TestNtmCopyQuick:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over 500 epochs with fixed data."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmCopyConfig()
        model = NtmCopyModel(cfg)

        # Fixed short sequences
        data = generate_copy_batch(8, 1, 3, cfg.w)

        losses: list[float] = []
        loss_val = 0.0
        for i in range(500):
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            loss_val = train_ntm_copy_step(model, data, nll_loss, optimizer)
            if i % 100 == 0:
                losses.append(loss_val)

        losses.append(loss_val)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_forward_shape(self) -> None:
        """Verify output dimensions."""
        cfg = NtmCopyConfig()
        model = NtmCopyModel(cfg)

        model.reset_state()
        x = torch.zeros(cfg.w)
        x[1] = 1.0
        output = model(x)
        assert output.shape == (cfg.w,)
        # LogSoftmax output should be <= 0
        assert (output <= 0).all()


@pytest.mark.slow
class TestNtmCopySlow:
    def test_curriculum_convergence(self) -> None:
        """Full curriculum training, accuracy > 90%."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmCopyConfig(epochs=3000, patience=150, chunk_size=50)
        model = NtmCopyModel(cfg)

        bs, w = cfg.batch_size, cfg.w
        stages = [
            Stage("Stage 1 (len 1-3)", 0.15, lambda: generate_copy_batch(bs, 1, 3, w)),
            Stage("Stage 2 (len 1-5)", 0.10, lambda: generate_copy_batch(bs, 1, 5, w)),
            Stage("Stage 3 (len 1-8)", 0.0, lambda: generate_copy_batch(bs, 1, 8, w)),
        ]

        def schedule_fn(epoch: int) -> float:
            warmup = int(0.25 * cfg.epochs)
            if epoch < warmup:
                lr_start = cfg.lr / 25.0
                return lr_start + (cfg.lr - lr_start) * epoch / max(warmup, 1)
            lr_end = cfg.lr / cfg.div_final
            progress = (epoch - warmup) / max(cfg.epochs - warmup, 1)
            return lr_end + (cfg.lr - lr_end) * 0.5 * (1 + math.cos(math.pi * progress))

        def optimizer_factory(m: NtmCopyModel, lr: float) -> torch.optim.Optimizer:
            return torch.optim.Adam(
                m.parameters(), lr=lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
            )

        done, _ = run_curriculum(
            model=model,
            loss_fn=nll_loss,
            stages=stages,
            total_epochs=cfg.epochs,
            patience=cfg.patience,
            chunk_size=cfg.chunk_size,
            optimizer_factory=optimizer_factory,
            schedule_fn=schedule_fn,
            post_step_fn=model.project_addressing,
            train_step_fn=train_ntm_copy_step,
        )

        # Evaluate accuracy
        test_data = generate_copy_batch(20, 1, 8, cfg.w)
        correct = 0
        total = 0
        with torch.no_grad():
            for xs, ys in test_data:
                model.reset_state()
                for x, y in zip(xs, ys, strict=True):
                    pred = model(x)
                    if pred.argmax() == y.argmax():
                        correct += 1
                    total += 1

        acc = correct / total
        print(f"Accuracy: {acc:.2%} ({correct}/{total})")
        assert acc > 0.9, f"Accuracy {acc:.2%} < 90%"
