"""NTM layer module matching idris-ml's Layer.idr NtmLayer.

NOTE: Controller output is clamped to [-20, 20] matching applyLayerVar's
clampVar. Addressing weights are projected onto the probability simplex
after optimizer steps, matching syncLayerBuffers/projectWeights.
"""

import torch
import torch.nn as nn
from torch import Tensor

from bench.ntm.memory import forward_read_head, forward_write_head, tanh_bound


def _read_head_input_width(w: int) -> int:
    """w + ShiftKernelSize + 3"""
    return w + 3 + 3


def _write_head_input_width(w: int) -> int:
    """ReadHeadInputWidth + 2*w (erase + add)"""
    return _read_head_input_width(w) + 2 * w


def ntm_input_width(w: int) -> int:
    """NtmInputWidth = w + w (input + previous read output)"""
    return 2 * w


def ntm_output_width(n: int, w: int) -> int:
    """NtmOutputWidth = ReadHeadInputWidth + WriteHeadInputWidth + w"""
    return _read_head_input_width(w) + _write_head_input_width(w) + w


class NTMLayer(nn.Module):
    """Neural Turing Machine layer matching idris-ml's NtmLayer."""

    def __init__(self, controller: nn.Module, n: int, w: int) -> None:
        super().__init__()
        self.controller = controller
        self.n = n
        self.w = w

        # Memory initialized to constant 1e-6 (Collier & Beel)
        self.register_buffer("_init_memory", torch.full((n, w), 1e-6))
        self.memory: Tensor = torch.full((n, w), 1e-6)

        # Hot-start addressing: [1, 0, ..., 0]
        init_addr = torch.zeros(n)
        init_addr[0] = 1.0
        self.read_addressing = nn.Parameter(init_addr.clone())
        self.write_addressing = nn.Parameter(init_addr.clone())

        # Initial read head output (zeros)
        self.read_head_output = nn.Parameter(torch.zeros(w))

        # Dimension calculations
        self.read_head_width = _read_head_input_width(w)
        self.write_head_width = _write_head_input_width(w)

    def reset_state(self) -> None:
        """Reset memory and head state between sequences."""
        # pyright's torch stubs don't recognize .clone() on register_buffer tensors
        self.memory = self._init_memory.clone()  # type: ignore[reportCallIssue]
        self._current_read_addr = self.read_addressing.clone()
        self._current_write_addr = self.write_addressing.clone()
        self._current_read_output = self.read_head_output.clone()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for one timestep.

        x: (w,) input vector
        Returns: (w,) output vector (new data, before logSoftmax)
        """
        # Concatenate previous read output with input
        controller_input = torch.cat([self._current_read_output, x])

        # Controller forward pass
        # pyright doesn't see __call__ on nn.Module-typed fields
        raw_output: Tensor = self.controller(controller_input)  # type: ignore[operator]

        # NOTE: Clamp controller output to [-20, 20] matching applyLayerVar
        raw_output = raw_output.clamp(-20, 20)

        # Split controller output into head inputs + new data
        read_input = raw_output[: self.read_head_width]
        write_end = self.read_head_width + self.write_head_width
        write_input = raw_output[self.read_head_width : write_end]
        new_data = raw_output[self.read_head_width + self.write_head_width :]

        # Read head
        new_read_addr, read_output, _ = forward_read_head(
            self.memory, self._current_read_addr, read_input, self.w
        )

        # Write head
        new_write_addr, new_memory = forward_write_head(
            self.memory, self._current_write_addr, write_input, self.w
        )

        # Tanh memory bounding (Collier & Beel)
        new_memory = tanh_bound(new_memory)

        # Update state
        self.memory = new_memory
        self._current_read_addr = new_read_addr
        self._current_write_addr = new_write_addr
        self._current_read_output = read_output

        # Stash diagnostics (detached, no gradient impact)
        self._diag = {
            "read_addr": new_read_addr.detach().clone(),
            "write_addr": new_write_addr.detach().clone(),
            "memory": new_memory.detach().clone(),
            "read_output": read_output.detach().clone(),
            "controller_output": raw_output.detach().clone(),
        }

        return new_data

    def project_addressing(self, eps: float = 1e-6) -> None:
        """Project addressing weights onto probability simplex.

        Matches Layer.idr syncLayerBuffers/projectWeights: clamp to [eps, inf),
        then renormalize. Called after optimizer step.
        """
        with torch.no_grad():
            for param in [self.read_addressing, self.write_addressing]:
                param.clamp_(min=eps)
                param.div_(param.sum())
