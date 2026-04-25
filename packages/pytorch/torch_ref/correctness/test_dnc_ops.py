"""Unit tests for DNC addressing and memory operations."""

import torch

from torch_ref.dnc.addressing import (
    allocation_weighting,
    content_address,
    read_weighting,
    update_link_matrix,
    update_usage,
    write_weighting,
)
from torch_ref.dnc.memory import erase_add_write, read_op


class TestContentAddress:
    def test_strong_match(self) -> None:
        memory = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        key = torch.tensor([1.0, 0.0])
        beta = torch.tensor(10.0)
        w = content_address(beta, memory, key)
        assert w.shape == (3,)
        assert w[0].item() > w[1].item()
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_low_beta_uniform(self) -> None:
        memory = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        key = torch.tensor([1.0, 0.0])
        beta = torch.tensor(0.001)
        w = content_address(beta, memory, key)
        assert abs(w[0].item() - w[1].item()) < 0.1


class TestUsageUpdate:
    def test_write_increases_usage(self) -> None:
        prev_usage = torch.zeros(4)
        prev_write = torch.tensor([1.0, 0.0, 0.0, 0.0])
        free_gates = torch.tensor([0.0])  # no freeing
        prev_reads = [torch.zeros(4)]
        usage = update_usage(prev_usage, prev_write, free_gates, prev_reads)
        assert usage[0].item() > 0.9  # written slot is now used
        assert usage[1].item() < 0.1  # unwritten slots stay unused

    def test_free_gate_reduces_usage(self) -> None:
        prev_usage = torch.ones(4)
        prev_write = torch.zeros(4)
        free_gates = torch.tensor([1.0])  # fully free
        prev_reads = [torch.tensor([1.0, 0.0, 0.0, 0.0])]  # read slot 0
        usage = update_usage(prev_usage, prev_write, free_gates, prev_reads)
        # Slot 0 was read with free_gate=1 -> retention = 0 -> usage drops
        assert usage[0].item() < 0.1
        # Other slots: retention = 1 (not read) -> usage stays
        assert usage[1].item() > 0.9


class TestAllocationWeighting:
    def test_empty_memory(self) -> None:
        usage = torch.zeros(4)
        alloc = allocation_weighting(usage)
        assert alloc.shape == (4,)
        # All slots empty: first sorted slot gets weight 1.0
        assert alloc.sum().item() < 1.0 + 1e-5
        assert alloc.max().item() > 0.9

    def test_full_memory(self) -> None:
        usage = torch.ones(4)
        alloc = allocation_weighting(usage)
        # All slots full: allocation weights should be ~0
        assert alloc.sum().item() < 0.1

    def test_partial_usage(self) -> None:
        usage = torch.tensor([1.0, 0.0, 0.5, 0.0])
        alloc = allocation_weighting(usage)
        assert alloc.shape == (4,)
        # First empty slot (sorted ascending) gets all allocation.
        # Cumprod: [1, 0, 0, 0] because first sorted entry is 0.
        # Only slot 1 (first in sort order) gets weight 1.0.
        assert alloc[1].item() > 0.9
        assert alloc[0].item() < 0.01  # fully used slot gets nothing

    def test_graded_usage(self) -> None:
        usage = torch.tensor([0.9, 0.1, 0.5, 0.3])
        alloc = allocation_weighting(usage)
        assert alloc.shape == (4,)
        # Least used slot (1, usage=0.1) should get highest allocation
        assert alloc[1].item() > alloc[0].item()


class TestWriteWeighting:
    def test_allocation_mode(self) -> None:
        content_w = torch.tensor([0.0, 0.0, 1.0, 0.0])
        alloc_w = torch.tensor([1.0, 0.0, 0.0, 0.0])
        write_gate = torch.tensor(1.0)
        alloc_gate = torch.tensor(1.0)  # fully allocation
        w = write_weighting(content_w, alloc_w, write_gate, alloc_gate)
        assert torch.allclose(w, alloc_w)

    def test_content_mode(self) -> None:
        content_w = torch.tensor([0.0, 0.0, 1.0, 0.0])
        alloc_w = torch.tensor([1.0, 0.0, 0.0, 0.0])
        write_gate = torch.tensor(1.0)
        alloc_gate = torch.tensor(0.0)  # fully content
        w = write_weighting(content_w, alloc_w, write_gate, alloc_gate)
        assert torch.allclose(w, content_w)

    def test_write_gate_zero(self) -> None:
        content_w = torch.ones(4) * 0.25
        alloc_w = torch.ones(4) * 0.25
        write_gate = torch.tensor(0.0)  # no write
        alloc_gate = torch.tensor(0.5)
        w = write_weighting(content_w, alloc_w, write_gate, alloc_gate)
        assert torch.allclose(w, torch.zeros(4))


class TestLinkMatrix:
    def test_initial_state(self) -> None:
        link = torch.zeros(4, 4)
        precedence = torch.zeros(4)
        write_w = torch.tensor([1.0, 0.0, 0.0, 0.0])
        new_link, new_prec = update_link_matrix(link, precedence, write_w)
        assert new_link.shape == (4, 4)
        assert new_prec.shape == (4,)
        # Diagonal should be zero
        assert torch.allclose(new_link.diag(), torch.zeros(4))
        # Precedence should be write_w (since prev_precedence was zero)
        assert torch.allclose(new_prec, write_w)

    def test_sequential_writes(self) -> None:
        n = 4
        link = torch.zeros(n, n)
        precedence = torch.zeros(n)
        # Write to slot 0
        w0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        link, precedence = update_link_matrix(link, precedence, w0)
        # Write to slot 1
        w1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        link, precedence = update_link_matrix(link, precedence, w1)
        # link[1, 0] should be high (slot 1 was written after slot 0)
        assert link[1, 0].item() > 0.5
        # link[0, 1] should be low
        assert link[0, 1].item() < 0.1


class TestReadWeighting:
    def test_content_mode(self) -> None:
        n = 4
        link = torch.zeros(n, n)
        prev_rw = torch.tensor([1.0, 0.0, 0.0, 0.0])
        content_w = torch.tensor([0.0, 0.0, 1.0, 0.0])
        # mode = [back, content, forward] with strong content
        mode_params = torch.tensor([-10.0, 10.0, -10.0])
        rw = read_weighting(link, prev_rw, content_w, mode_params)
        assert rw.shape == (n,)
        # Should be close to content_w
        assert rw[2].item() > 0.9

    def test_forward_mode(self) -> None:
        n = 4
        # Link: slot 1 follows slot 0
        link = torch.zeros(n, n)
        link[1, 0] = 1.0
        prev_rw = torch.tensor([1.0, 0.0, 0.0, 0.0])
        content_w = torch.zeros(n)
        mode_params = torch.tensor([-10.0, -10.0, 10.0])  # strong forward
        rw = read_weighting(link, prev_rw, content_w, mode_params)
        # Forward from slot 0 -> link @ prev_rw = link[:, 0] = [0, 1, 0, 0]
        assert rw[1].item() > 0.9


class TestMemoryOps:
    def test_read_op(self) -> None:
        weights = torch.tensor([1.0, 0.0, 0.0])
        memory = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = read_op(weights, memory)
        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(result, expected)

    def test_erase_add_write(self) -> None:
        memory = torch.ones(3, 2)
        w = torch.tensor([1.0, 0.0, 0.0])
        erase = torch.ones(2)  # erase everything at slot 0
        add = torch.tensor([5.0, 6.0])
        result = erase_add_write(memory, w, erase, add)
        # Slot 0: 1*(1-1*1) + 1*5 = 5, 1*(1-1*1) + 1*6 = 6
        assert torch.allclose(result[0], torch.tensor([5.0, 6.0]))
        # Other slots: 1*(1-0) + 0 = 1
        assert torch.allclose(result[1], torch.tensor([1.0, 1.0]))

    def test_partial_erase(self) -> None:
        memory = torch.ones(2, 2) * 10.0
        w = torch.tensor([0.5, 0.0])
        erase = torch.tensor([1.0, 0.0])  # erase only first element
        add = torch.zeros(2)
        result = erase_add_write(memory, w, erase, add)
        # Slot 0: [10*(1-0.5*1), 10*(1-0.5*0)] = [5, 10]
        assert abs(result[0, 0].item() - 5.0) < 1e-5
        assert abs(result[0, 1].item() - 10.0) < 1e-5


class TestGradientFlow:
    """Verify gradients flow through DNC operations."""

    def test_allocation_gradient(self) -> None:
        usage = torch.tensor([0.1, 0.9, 0.5, 0.2], requires_grad=True)
        alloc = allocation_weighting(usage)
        loss = alloc.sum()
        loss.backward()
        assert usage.grad is not None
        assert not torch.any(torch.isnan(usage.grad))

    def test_link_matrix_gradient(self) -> None:
        link = torch.zeros(4, 4, requires_grad=True)
        prec = torch.zeros(4, requires_grad=True)
        ww = torch.tensor([0.5, 0.5, 0.0, 0.0], requires_grad=True)
        new_link, new_prec = update_link_matrix(link, prec, ww)
        loss = new_link.sum() + new_prec.sum()
        loss.backward()
        assert ww.grad is not None
        assert not torch.any(torch.isnan(ww.grad))

    def test_full_forward_gradient(self) -> None:
        """End-to-end gradient through content address + write + read."""
        n, m = 4, 3
        memory = torch.randn(n, m, requires_grad=True)
        key = torch.randn(m, requires_grad=True)
        beta = torch.tensor(5.0, requires_grad=True)

        # Content address -> write -> read
        cw = content_address(beta, memory, key)
        erase = torch.sigmoid(torch.randn(m, requires_grad=True))
        add = torch.randn(m, requires_grad=True)
        ww = cw * 0.5  # scale write weights

        new_mem = erase_add_write(memory, ww, erase, add)
        output = read_op(cw, new_mem)
        loss = output.sum()
        loss.backward()

        assert memory.grad is not None
        assert key.grad is not None
        assert not torch.any(torch.isnan(memory.grad))
