"""Correctness tests for NTM associative recall model (reference architecture)."""

import math
import random

import torch

from bench.data.recall_task import generate_recall_sequence
from bench.diagnostics.ntm_diagnostics import (
    compute_summary,
    instrumented_forward_recall,
)
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
        """Loss should decrease over 500 training steps."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = NtmRecallConfig()
        model = NtmRecallModel(cfg)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.95, momentum=0.9)

        losses: list[float] = []
        final_loss = 0.0
        for i in range(500):
            input_seq, target_seq = generate_recall_sequence(
                num_items=2, seq_len=cfg.seq_len, seq_width=cfg.seq_width
            )
            final_loss = train_ntm_recall_step(model, input_seq, target_seq, optimizer)
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
# Validation tests: fast checks that core NTM recall machinery works
# ---------------------------------------------------------------------------


def _small_recall_config() -> NtmRecallConfig:
    """Default-sized config restricted to 2-item sequences for fast tests."""
    return NtmRecallConfig(
        seq_width=6,
        seq_len=3,
        min_items=2,
        max_items=2,
    )


def _train_small_recall(steps: int, seed: int = 42) -> tuple[NtmRecallModel, list[float]]:
    """Train a small recall model, returning model and loss history."""
    torch.manual_seed(seed)
    random.seed(seed)

    cfg = _small_recall_config()
    model = NtmRecallModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for _ in range(steps):
        input_seq, target_seq = generate_recall_sequence(
            num_items=2, seq_len=cfg.seq_len, seq_width=cfg.seq_width
        )
        loss = train_ntm_recall_step(model, input_seq, target_seq, optimizer)
        losses.append(loss)

    return model, losses


def _bit_accuracy(model: NtmRecallModel, n_seqs: int = 10) -> float:
    """Compute bit accuracy on random 2-item sequences."""
    cfg = model.cfg
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_seqs):
            input_seq, target_seq = generate_recall_sequence(
                num_items=2, seq_len=cfg.seq_len, seq_width=cfg.seq_width
            )
            model.reset_state()
            for t in range(input_seq.shape[0]):
                model(input_seq[t])
            zero_input = torch.zeros(cfg.seq_width + 2)
            for t in range(target_seq.shape[0]):
                out = model(zero_input)
                pred_bits = (out > 0.5).float()
                correct += int((pred_bits == target_seq[t]).sum().item())
                total += target_seq.shape[1]
    return correct / total if total > 0 else 0.0


class TestNtmRecallConvergence:
    """Test 1: Tiny 2-item recall converges in ~2000 steps."""

    def test_tiny_recall_converges(self) -> None:
        model, losses = _train_small_recall(2000)

        # Use tail average to smooth single-sequence noise
        # Random baseline is ~0.69 (BCE); model should be clearly below
        tail_avg = sum(losses[-50:]) / 50
        assert tail_avg < 0.55, f"Tail-avg loss {tail_avg:.4f} should be < 0.55"

        acc = _bit_accuracy(model, n_seqs=10)
        assert acc > 0.60, f"Bit accuracy {acc:.2%} should be > 60%"


class TestNtmRecallGradientFlow:
    """Test 2: Gradients flow through all parameter groups."""

    def test_all_params_have_gradients(self) -> None:
        torch.manual_seed(42)
        random.seed(42)

        cfg = _small_recall_config()
        model = NtmRecallModel(cfg)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.95, momentum=0.9)

        input_seq, target_seq = generate_recall_sequence(
            num_items=2, seq_len=cfg.seq_len, seq_width=cfg.seq_width
        )
        train_ntm_recall_step(model, input_seq, target_seq, optimizer)

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            grad_norm = param.grad.norm().item()
            assert grad_norm > 0, f"Zero gradient for {name}"
            assert not math.isnan(grad_norm), f"NaN gradient for {name}"


class TestNtmRecallMemoryState:
    """Test 3: Memory state is non-trivial after encoding."""

    def test_memory_modified_after_encoding(self) -> None:
        torch.manual_seed(42)
        random.seed(42)

        cfg = _small_recall_config()
        model = NtmRecallModel(cfg)
        model.reset_state()

        input_seq, _ = generate_recall_sequence(
            num_items=2, seq_len=cfg.seq_len, seq_width=cfg.seq_width
        )

        with torch.no_grad():
            for t in range(input_seq.shape[0]):
                model(input_seq[t])

        memory = model.ntm.memory.detach()

        mem_std = memory.std().item()
        assert mem_std > 0.01, f"Memory std {mem_std:.6f} too low (still near init?)"

        row_norms = memory.norm(dim=-1)
        active_rows = int((row_norms > 0.1).sum().item())
        assert active_rows >= 2, (
            f"Only {active_rows} memory rows active (norm > 0.1), expected >= 2"
        )


class TestNtmRecallDiagnostics:
    """Test 4: After training, diagnostics show no degenerate strategies."""

    def test_no_degenerate_strategies(self) -> None:
        model, _ = _train_small_recall(2000)

        torch.manual_seed(99)
        random.seed(99)

        cfg = model.cfg
        input_seq, target_seq = generate_recall_sequence(
            num_items=2, seq_len=cfg.seq_len, seq_width=cfg.seq_width
        )

        timesteps = instrumented_forward_recall(model, input_seq, target_seq)

        # Encoding length: 2 items * (1 delim + 3 data) = 8, plus query phase
        # delims + data before output
        encode_len = input_seq.shape[0]
        summary = compute_summary(timesteps, seq_len=encode_len)

        # Write addressing not collapsed: at least 2 distinct slots during encoding
        write_argmaxes_encode = summary.write_argmaxes[:encode_len]
        distinct_write_slots = len(set(write_argmaxes_encode))
        assert distinct_write_slots >= 2, (
            f"Write addressing collapsed: only {distinct_write_slots} distinct slots "
            f"during encoding (argmaxes: {write_argmaxes_encode})"
        )

        # Read addressing not frozen: peak mass > 0.15 during output
        # (averaged over all timesteps, encoding phase dilutes this)
        assert summary.read_addr_peak_mass > 0.15, (
            f"Read addressing too diffuse: peak mass {summary.read_addr_peak_mass:.4f} <= 0.15"
        )

        # Content addressing active: read g > 0.1 during output
        assert summary.read_g_output > 0.1, (
            f"Read gate too low during output: g={summary.read_g_output:.4f} <= 0.1, "
            "model may ignore content addressing"
        )

        # Write g active during encoding
        assert summary.write_g_input > 0.1, (
            f"Write gate too low during encoding: g={summary.write_g_input:.4f} <= 0.1"
        )

        # Memory slots actually used
        assert summary.slots_used >= 2, (
            f"Only {summary.slots_used} memory slots used, expected >= 2"
        )

        # No NaN in head params
        for ts in timesteps:
            assert not math.isnan(ts.read_beta), f"NaN read_beta at t={ts.timestep}"
            assert not math.isnan(ts.read_g), f"NaN read_g at t={ts.timestep}"
            assert not math.isnan(ts.read_gamma), f"NaN read_gamma at t={ts.timestep}"
            assert not math.isnan(ts.write_beta), f"NaN write_beta at t={ts.timestep}"
            assert not math.isnan(ts.write_g), f"NaN write_g at t={ts.timestep}"
            assert not math.isnan(ts.write_gamma), f"NaN write_gamma at t={ts.timestep}"
