"""Correctness tests for NTM associative recall task."""

import math
import random
from typing import cast

import torch

from torch_ref.data.recall_task import generate_recall_sequence
from torch_ref.diagnostics.ntm_diagnostics import (
    compute_summary,
    instrumented_forward_recall,
)
from torch_ref.models.ntm import NtmConfig, NtmModel, train_ntm_step

SEQ_WIDTH = 6
SEQ_LEN = 3


def _recall_config(**kwargs: object) -> NtmConfig:
    """Create NtmConfig for recall task (input_width=seq_width+2)."""
    return NtmConfig(input_width=SEQ_WIDTH + 2, output_width=SEQ_WIDTH, **kwargs)  # type: ignore[arg-type]


class TestNtmRecallQuick:
    def test_forward_shape(self) -> None:
        """Output should be (seq_width,) raw logits (unbounded)."""
        cfg = _recall_config()
        model = NtmModel(cfg)
        model.reset_state()

        x = torch.zeros(cfg.input_width)
        x[0] = 1.0
        output = model(x)
        assert output.shape == (cfg.output_width,)
        assert output.isfinite().all()

    def test_loss_decreases(self) -> None:
        """Loss should decrease over 1000 training steps."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        random.seed(42)

        cfg = _recall_config()
        model = NtmModel(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses: list[float] = []
        final_loss = 0.0
        for i in range(1000):
            input_seq, target_seq = generate_recall_sequence(
                num_items=2, seq_len=SEQ_LEN, seq_width=SEQ_WIDTH
            )
            final_loss, _ = train_ntm_step(model, input_seq, target_seq, optimizer)
            if i < 20:
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


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def _train_small_recall(steps: int, seed: int = 42) -> tuple[NtmModel, list[float]]:
    """Train a small recall model, returning model and loss history."""
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
    random.seed(seed)

    cfg = _recall_config()
    model = NtmModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for _ in range(steps):
        input_seq, target_seq = generate_recall_sequence(
            num_items=2, seq_len=SEQ_LEN, seq_width=SEQ_WIDTH
        )
        loss, _ = train_ntm_step(model, input_seq, target_seq, optimizer)
        losses.append(loss)

    return model, losses


def _bit_accuracy(model: NtmModel, n_seqs: int = 10) -> float:
    """Compute bit accuracy on random 2-item sequences."""
    cfg = model.cfg
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_seqs):
            input_seq, target_seq = generate_recall_sequence(
                num_items=2, seq_len=SEQ_LEN, seq_width=SEQ_WIDTH
            )
            model.reset_state()
            for t in range(input_seq.shape[0]):
                model(input_seq[t])
            zero_input = torch.zeros(cfg.input_width)
            for t in range(target_seq.shape[0]):
                out = torch.sigmoid(model(zero_input))
                pred_bits = (out > 0.5).float()
                correct += int((pred_bits == target_seq[t]).sum().item())
                total += target_seq.shape[1]
    return correct / total if total > 0 else 0.0


class TestNtmRecallConvergence:
    """Tiny 2-item recall converges in ~3000 steps."""

    def test_tiny_recall_converges(self) -> None:
        model, losses = _train_small_recall(3000)

        # Use tail average to smooth single-sequence noise
        # Random baseline is ~0.69 (BCE); model should be clearly below
        tail_avg = sum(losses[-50:]) / 50
        assert tail_avg < 0.60, f"Tail-avg loss {tail_avg:.4f} should be < 0.60"

        acc = _bit_accuracy(model, n_seqs=10)
        assert acc > 0.55, f"Bit accuracy {acc:.2%} should be > 55%"


class TestNtmRecallGradientFlow:
    """Gradients flow through all parameter groups."""

    def test_all_params_have_gradients(self) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        random.seed(42)

        cfg = _recall_config()
        model = NtmModel(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        input_seq, target_seq = generate_recall_sequence(
            num_items=2, seq_len=SEQ_LEN, seq_width=SEQ_WIDTH
        )
        train_ntm_step(model, input_seq, target_seq, optimizer)

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            # Tensor.norm carries untyped dim/dtype params in torch's stubs
            grad_norm = cast(
                "float",
                param.grad.norm().item(),  # pyright: ignore[reportUnknownMemberType]
            )
            assert not math.isnan(grad_norm), f"NaN gradient for {name}"


class TestNtmRecallMemoryState:
    """Memory state is non-trivial after encoding."""

    def test_memory_modified_after_encoding(self) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        random.seed(42)

        cfg = _recall_config()
        model = NtmModel(cfg)
        model.reset_state()

        input_seq, _ = generate_recall_sequence(num_items=2, seq_len=SEQ_LEN, seq_width=SEQ_WIDTH)

        with torch.no_grad():
            for t in range(input_seq.shape[0]):
                model(input_seq[t])

        memory = model.ntm.memory.detach()

        mem_std = memory.std().item()
        assert mem_std > 0.001, f"Memory std {mem_std:.6f} too low (still near init?)"

        # Check memory differs from init
        init_memory = torch.full_like(memory, 1e-6)
        diff = (memory - init_memory).abs().max().item()
        assert diff > 1e-4, f"Memory unchanged from init: max diff {diff:.6f}"


class TestNtmRecallDiagnostics:
    """After training, diagnostics show no degenerate strategies."""

    def test_no_degenerate_strategies(self) -> None:
        model, _ = _train_small_recall(3000)

        torch.manual_seed(99)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        random.seed(99)

        input_seq, target_seq = generate_recall_sequence(
            num_items=2, seq_len=SEQ_LEN, seq_width=SEQ_WIDTH
        )

        timesteps = instrumented_forward_recall(model, input_seq, target_seq)

        encode_len = input_seq.shape[0]
        summary = compute_summary(timesteps, seq_len=encode_len)

        # Memory slots modified
        assert summary.slots_used >= 1, f"No memory slots used ({summary.slots_used})"

        # No NaN in head params
        for ts in timesteps:
            assert not math.isnan(ts.read_beta), f"NaN read_beta at t={ts.timestep}"
            assert not math.isnan(ts.read_g), f"NaN read_g at t={ts.timestep}"
            assert not math.isnan(ts.read_gamma), f"NaN read_gamma at t={ts.timestep}"
            assert not math.isnan(ts.write_beta), f"NaN write_beta at t={ts.timestep}"
            assert not math.isnan(ts.write_g), f"NaN write_g at t={ts.timestep}"
            assert not math.isnan(ts.write_gamma), f"NaN write_gamma at t={ts.timestep}"
