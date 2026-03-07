"""Unit tests for NTM addressing and memory operations."""

import torch

from bench.ntm.addressing import content_address, cosine_similarity, focus, interpolate, shift
from bench.ntm.memory import (
    forward_read_head,
    forward_write_head,
    read_op,
    write_memory,
)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        a = torch.tensor([1.0, 2.0, 3.0])
        result = cosine_similarity(a, a)
        assert abs(result.item() - 1.0) < 1e-5

    def test_orthogonal_vectors(self) -> None:
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        result = cosine_similarity(a, b)
        assert abs(result.item()) < 1e-5

    def test_opposite_vectors(self) -> None:
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        result = cosine_similarity(a, b)
        assert abs(result.item() + 1.0) < 1e-5


class TestContentAddress:
    def test_strong_match(self) -> None:
        memory = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        key = torch.tensor([1.0, 0.0])
        beta = torch.tensor(10.0)
        w = content_address(beta, memory, key)
        assert w.shape == (3,)
        # First row should have highest weight
        assert w[0].item() > w[1].item()
        assert w[0].item() > w[2].item()
        # Should sum to 1
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_low_beta_uniform(self) -> None:
        memory = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        key = torch.tensor([1.0, 0.0])
        beta = torch.tensor(0.001)
        w = content_address(beta, memory, key)
        # Low beta → nearly uniform
        assert abs(w[0].item() - w[1].item()) < 0.1


class TestInterpolate:
    def test_g_zero(self) -> None:
        content = torch.tensor([1.0, 0.0, 0.0])
        location = torch.tensor([0.0, 0.0, 1.0])
        result = interpolate(torch.tensor(0.0), content, location)
        assert torch.allclose(result, location)

    def test_g_one(self) -> None:
        content = torch.tensor([1.0, 0.0, 0.0])
        location = torch.tensor([0.0, 0.0, 1.0])
        result = interpolate(torch.tensor(1.0), content, location)
        assert torch.allclose(result, content)


class TestShift:
    def test_stay(self) -> None:
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # kernel that strongly favors "stay" (middle element)
        kernel = torch.tensor([-10.0, 10.0, -10.0])
        result = shift(weights, kernel)
        assert torch.allclose(result, weights, atol=1e-3)

    def test_shift_right(self) -> None:
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # kernel that strongly favors shift right (sr = kernel[2])
        kernel = torch.tensor([-10.0, -10.0, 10.0])
        result = shift(weights, kernel)
        expected = torch.tensor([0.0, 1.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-3)


class TestFocus:
    def test_sharpening(self) -> None:
        weights = torch.tensor([0.6, 0.3, 0.1])
        gamma_low = torch.tensor(1.0)
        gamma_high = torch.tensor(5.0)
        result_low = focus(gamma_low, weights)
        result_high = focus(gamma_high, weights)
        # Higher gamma should sharpen: max weight should increase
        assert result_high[0].item() > result_low[0].item()


class TestMemoryOps:
    def test_read_op(self) -> None:
        weights = torch.tensor([1.0, 0.0, 0.0])
        memory = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = read_op(weights, memory)
        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(result, expected)

    def test_write_memory(self) -> None:
        """Interpolation write: w*data + (1-w)*mem."""
        memory = torch.ones(3, 2)
        weights = torch.tensor([1.0, 0.0, 0.0])
        add = torch.tensor([5.0, 6.0])
        result = write_memory(memory, weights, add)
        # First row: 1.0 * [5, 6] + 0.0 * [1, 1] = [5, 6]
        assert torch.allclose(result[0], torch.tensor([5.0, 6.0]))
        # Other rows: 0.0 * [5, 6] + 1.0 * [1, 1] = [1, 1]
        assert torch.allclose(result[1], torch.tensor([1.0, 1.0]))


class TestForwardHeads:
    def test_read_head_output_shape(self) -> None:
        n, w = 4, 3
        memory = torch.full((n, w), 1e-6)
        addr = torch.zeros(n)
        addr[0] = 1.0
        # head_input size: w + 3 + 3 = 9
        head_input = torch.randn(w + 6)
        new_addr, output = forward_read_head(memory, addr, head_input, w)
        assert new_addr.shape == (n,)
        assert output.shape == (w,)
        assert abs(new_addr.sum().item() - 1.0) < 1e-4

    def test_write_head_output_shape(self) -> None:
        n, w = 4, 3
        memory = torch.full((n, w), 1e-6)
        addr = torch.zeros(n)
        addr[0] = 1.0
        # head_input size: (w + 3 + 3) + w = 12 (no erase vector)
        head_input = torch.randn(w + 6 + w)
        new_addr, new_memory = forward_write_head(memory, addr, head_input, w)
        assert new_addr.shape == (n,)
        assert new_memory.shape == (n, w)
