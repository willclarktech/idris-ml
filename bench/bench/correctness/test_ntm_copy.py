"""Correctness tests for NTM copy model (reference architecture)."""

import random

import torch

from bench.data.copy_task import generate_copy_sequence
from bench.models.ntm_copy import NtmCopyConfig, NtmCopyModel, train_ntm_copy_step


class TestNtmCopyQuick:
    def test_forward_shape(self) -> None:
        """Output should be (seq_width,) sigmoid values in [0,1]."""
        cfg = NtmCopyConfig()
        model = NtmCopyModel(cfg)
        model.reset_state()

        x = torch.zeros(cfg.seq_width + 1)  # input width includes delimiter channel
        x[0] = 1.0
        output = model(x)
        assert output.shape == (cfg.seq_width,)
        assert (output >= 0).all() and (output <= 1).all()

    def test_loss_decreases(self) -> None:
        """Loss should decrease over 200 training steps."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmCopyConfig()
        model = NtmCopyModel(cfg)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.95, momentum=0.9)

        losses: list[float] = []
        final_loss = 0.0
        for i in range(200):
            seq_len = random.randint(1, 5)
            input_seq, target_seq = generate_copy_sequence(seq_len, cfg.seq_width)
            final_loss = train_ntm_copy_step(model, input_seq, target_seq, optimizer)
            if i < 10:
                losses.append(final_loss)

        early_avg = sum(losses) / len(losses)
        assert final_loss < early_avg, f"Loss did not decrease: {early_avg:.4f} -> {final_loss:.4f}"

    def test_input_output_dimensions(self) -> None:
        """Verify copy sequence dimensions."""
        input_seq, target_seq = generate_copy_sequence(seq_len=5, seq_width=8)
        assert input_seq.shape == (6, 9)  # 5 data + 1 delimiter, 8 bits + 1 delim channel
        assert target_seq.shape == (5, 8)  # 5 data vectors, 8 bits
