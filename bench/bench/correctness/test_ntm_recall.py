"""Correctness tests for NTM associative recall model (reference architecture)."""

import random

import torch

from bench.data.recall_task import generate_recall_sequence
from bench.models.ntm_recall import NtmRecallConfig, NtmRecallModel, train_ntm_recall_step


class TestNtmRecallQuick:
    def test_forward_shape(self) -> None:
        """Output should be (seq_width,) sigmoid values in [0,1]."""
        cfg = NtmRecallConfig()
        model = NtmRecallModel(cfg)
        model.reset_state()

        x = torch.zeros(cfg.seq_width + 2)  # input width includes 2 delim channels
        x[0] = 1.0
        output = model(x)
        assert output.shape == (cfg.seq_width,)
        assert (output >= 0).all() and (output <= 1).all()

    def test_loss_decreases(self) -> None:
        """Loss should decrease over 200 training steps."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmRecallConfig()
        model = NtmRecallModel(cfg)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.95, momentum=0.9)

        losses: list[float] = []
        final_loss = 0.0
        for i in range(200):
            input_seq, target_seq = generate_recall_sequence(
                num_items=2, seq_len=cfg.seq_len, seq_width=cfg.seq_width
            )
            final_loss = train_ntm_recall_step(model, input_seq, target_seq, optimizer)
            if i < 10:
                losses.append(final_loss)

        early_avg = sum(losses) / len(losses)
        assert final_loss < early_avg, f"Loss did not decrease: {early_avg:.4f} -> {final_loss:.4f}"

    def test_sequence_structure(self) -> None:
        """Verify recall sequence dimensions for 3 items, seq_len=3, seq_width=6."""
        input_seq, target_seq = generate_recall_sequence(num_items=3, seq_len=3, seq_width=6)
        # 3 items: each has 1 delim + 3 data = 4 rows -> 12 rows
        # Query phase: 1 query_delim + 3 query data + 1 query_delim = 5 rows
        # Total: 12 + 5 = 17 timesteps
        assert input_seq.shape == (17, 8)  # 6 data + 2 delim channels
        assert target_seq.shape == (3, 6)  # seq_len=3 output vectors

    def test_variable_items(self) -> None:
        """Sequences with different item counts have correct dimensions."""
        for num_items in [2, 4, 6]:
            input_seq, target_seq = generate_recall_sequence(
                num_items=num_items, seq_len=3, seq_width=6
            )
            # Items: num_items * (1 delim + 3 data) = num_items * 4
            # Query: 1 + 3 + 1 = 5
            expected_total = num_items * 4 + 5
            assert input_seq.shape[0] == expected_total, (
                f"num_items={num_items}: expected {expected_total}, got {input_seq.shape[0]}"
            )
            assert input_seq.shape[1] == 8
            assert target_seq.shape == (3, 6)
