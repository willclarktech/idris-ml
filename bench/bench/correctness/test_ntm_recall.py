"""Correctness tests for NTM associative recall model."""

import random

import pytest
import torch

from bench.data.recall_task import associative_recall_point, generate_recall_batch
from bench.models.ntm_recall import NtmRecallConfig, NtmRecallModel, train_ntm_recall_step
from bench.training.losses import weighted_nll_loss


class TestNtmRecallQuick:
    def test_encoding_length(self) -> None:
        """Sequence length should be 3K+1 (immediate output, no blank query steps)."""
        w = 8
        for k in [1, 2, 3]:
            pairs = [(i + 1, i + 2) for i in range(k)]
            query_order = [p[0] for p in pairs]
            xs, ys = associative_recall_point(pairs, query_order, w)
            assert len(xs) == 3 * k + 1, f"K={k}: expected {3 * k + 1}, got {len(xs)}"
            assert len(ys) == 3 * k + 1

    def test_query_targets_immediate(self) -> None:
        """Query targets should appear at same timestep as query key (not delayed)."""
        w = 8
        pairs = [(1, 3), (2, 4)]
        query_order = [2, 1]
        xs, ys = associative_recall_point(pairs, query_order, w)
        # Store: k1 v1 k2 v2 (4 steps) + delim (1 step) = 5 steps
        # Query: q1=2 q2=1 (2 steps), targets: v2=4 v1=3
        assert len(xs) == 7  # 3*2+1
        assert int(ys[5].argmax()) == 4  # t5: query=2, answer=4
        assert int(ys[6].argmax()) == 3  # t6: query=1, answer=3

    def test_loss_decreases(self) -> None:
        """Loss should decrease over 200 epochs with fixed K=1 data (RNN)."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmRecallConfig(controller="rnn")
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
        """Verify output dimensions (RNN, read output mode)."""
        cfg = NtmRecallConfig(controller="rnn")
        model = NtmRecallModel(cfg)

        model.reset_state()
        x = torch.zeros(cfg.w)
        x[1] = 1.0
        output = model(x)
        assert output.shape == (cfg.w,)
        # LogSoftmax output should be <= 0
        assert (output <= 0).all()

    def test_forward_shape_lstm(self) -> None:
        """Verify output dimensions with LSTM controller (read output mode)."""
        cfg = NtmRecallConfig(controller="lstm")
        model = NtmRecallModel(cfg)

        model.reset_state()
        x = torch.zeros(cfg.w)
        x[1] = 1.0
        output = model(x)
        assert output.shape == (cfg.w,)
        assert (output <= 0).all()

    def test_loss_decreases_lstm(self) -> None:
        """Loss should decrease over 200 epochs with LSTM + K=1 data."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmRecallConfig(controller="lstm")
        model = NtmRecallModel(cfg)

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

    def test_read_output_mode(self) -> None:
        """Verify read output mode: output uses read vector + controller hidden."""
        cfg = NtmRecallConfig(controller="lstm", output_mode="read")
        model = NtmRecallModel(cfg)

        model.reset_state()
        x = torch.zeros(cfg.w)
        x[1] = 1.0
        output = model(x)
        assert output.shape == (cfg.w,)
        assert (output <= 0).all()

        # NTMLayer should have output_fc in read mode
        assert hasattr(model.ntm, "output_fc")
        assert model.ntm.output_mode == "read"

    def test_controller_output_mode(self) -> None:
        """Verify controller output mode (idris-ml compatible)."""
        cfg = NtmRecallConfig(controller="lstm", output_mode="controller")
        model = NtmRecallModel(cfg)

        model.reset_state()
        x = torch.zeros(cfg.w)
        x[1] = 1.0
        output = model(x)
        assert output.shape == (cfg.w,)
        assert (output <= 0).all()
        assert model.ntm.output_mode == "controller"

    def test_value_clipping(self) -> None:
        """Verify value clipping mode works without errors."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmRecallConfig(controller="lstm", clip_mode="value", clip_value=10.0)
        model = NtmRecallModel(cfg)
        data = generate_recall_batch(4, 1, 1, cfg.w)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loss = train_ntm_recall_step(model, data, weighted_nll_loss, optimizer)
        assert loss > 0

    def test_rmsprop_optimizer(self) -> None:
        """Verify RMSprop optimizer trains without errors."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmRecallConfig(controller="lstm", optimizer="rmsprop")
        model = NtmRecallModel(cfg)
        data = generate_recall_batch(4, 1, 1, cfg.w)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.95, momentum=0.9)
        loss = train_ntm_recall_step(model, data, weighted_nll_loss, optimizer)
        assert loss > 0


@pytest.mark.slow
class TestNtmRecallSlow:
    def test_k1_convergence(self) -> None:
        """K=1 pairs should converge with enough training."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmRecallConfig(epochs=2000, patience=500, chunk_size=25)
        model = NtmRecallModel(cfg)

        losses = []
        data = generate_recall_batch(cfg.batch_size, 1, 1, cfg.w)
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
