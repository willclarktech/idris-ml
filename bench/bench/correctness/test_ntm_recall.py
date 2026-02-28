"""Correctness tests for NTM associative recall model."""

import random

import pytest
import torch

from bench.data.recall_task import generate_recall_batch
from bench.models.ntm_recall import NtmRecallConfig, NtmRecallModel, train_ntm_recall_step
from bench.training.losses import weighted_nll_loss


class TestNtmRecallQuick:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over 200 epochs with fixed K=1 data."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmRecallConfig()
        model = NtmRecallModel(cfg)

        # Fixed simple data (K=1 pair)
        data = generate_recall_batch(8, 1, 1, cfg.w)

        initial_loss = None
        final_loss = None
        for i in range(200):
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            loss_val = train_ntm_recall_step(model, data, weighted_nll_loss, optimizer)
            if i == 0:
                initial_loss = loss_val
            final_loss = loss_val

        assert final_loss is not None and initial_loss is not None
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_forward_shape(self) -> None:
        """Verify output dimensions."""
        cfg = NtmRecallConfig()
        model = NtmRecallModel(cfg)

        model.reset_state()
        model.ntm.controller.reset_state()
        x = torch.zeros(cfg.w)
        x[1] = 1.0
        output = model(x)
        assert output.shape == (cfg.w,)
        # LogSoftmax output should be <= 0
        assert (output <= 0).all()


@pytest.mark.slow
class TestNtmRecallSlow:
    def test_k1_convergence(self) -> None:
        """K=1 pairs should converge with enough training."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmRecallConfig(epochs=2000, patience=500, chunk_size=25)
        model = NtmRecallModel(cfg)

        losses = []
        for epoch in range(cfg.epochs):
            if epoch % cfg.chunk_size == 0:
                data = generate_recall_batch(cfg.batch_size, 1, 1, cfg.w)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
            )
            loss_val = train_ntm_recall_step(model, data, weighted_nll_loss, optimizer)
            losses.append(loss_val)

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        print(f"K=1 final loss: {losses[-1]:.6f}")
