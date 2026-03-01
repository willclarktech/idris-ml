"""NTM layer matching loudinthecloud/pytorch-ntm reference architecture.

The controller (LSTM or other) outputs a hidden state. Separate linear layers
map this hidden state to read head params, write head params, and output.
This decouples memory width (m) from input/output width.
"""

import torch
import torch.nn as nn
from torch import Tensor

from bench.ntm.memory import forward_read_head, forward_write_head, tanh_bound


def _read_head_params_width(m: int) -> int:
    """key(m) + shift(3) + beta(1) + g(1) + gamma(1) = m + 6"""
    return m + 6


def _write_head_params_width(m: int) -> int:
    """key(m) + shift(3) + beta(1) + g(1) + gamma(1) + erase(m) + add(m) = 3m + 6"""
    return 3 * m + 6


class NTMLayer(nn.Module):
    """Neural Turing Machine layer matching loudinthecloud/pytorch-ntm.

    Head parameters are produced by separate linear layers from the controller
    hidden state, rather than slicing a monolithic controller output vector.

    Args:
        controller: Recurrent controller (e.g. LSTM). Must expose .last_hidden
                    property and .reset_state() method.
        n: Number of memory rows (slots).
        m: Memory width per row.
        num_inputs: External input width (task-specific).
        num_outputs: Output width (task-specific).
        controller_hidden_size: Hidden dimension of the controller.
    """

    def __init__(
        self,
        controller: nn.Module,
        n: int,
        m: int,
        num_inputs: int,
        num_outputs: int,
        controller_hidden_size: int,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.n = n
        self.m = m
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_hidden_size = controller_hidden_size

        # Head FC layers (from controller hidden → head params)
        read_params_w = _read_head_params_width(m)
        write_params_w = _write_head_params_width(m)

        self.read_fc = nn.Linear(controller_hidden_size, read_params_w)
        self.write_fc = nn.Linear(controller_hidden_size, write_params_w)

        # Output FC: controller hidden + read vector → output
        self.output_fc = nn.Linear(controller_hidden_size + m, num_outputs)

        # Initialize all FCs
        for fc in [self.read_fc, self.write_fc, self.output_fc]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

        # Memory initialized to constant 1e-6 (Collier & Beel)
        self.register_buffer("_init_memory", torch.full((n, m), 1e-6))
        self.memory: Tensor = torch.full((n, m), 1e-6)

    def reset_state(self) -> None:
        """Reset memory and head state between sequences.

        Addressing weights start as zeros (not learnable) — matching vlgiitr
        reference. The controller must learn to address correctly through its
        own outputs rather than relying on a learned addressing prior.
        """
        self.memory = self._init_memory.clone()  # type: ignore[reportCallIssue]
        self._current_read_addr = torch.zeros(self.n)
        self._current_write_addr = torch.zeros(self.n)
        # Read head output: kaiming init (matching vlgiitr reference)
        read_out = torch.empty(1, self.m)
        nn.init.kaiming_uniform_(read_out)
        self._current_read_output = read_out.squeeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for one timestep.

        x: (num_inputs,) input vector
        Returns: (num_outputs,) output vector
        """
        # Concatenate previous read output with input
        controller_input = torch.cat([self._current_read_output, x])

        # Controller forward pass
        _: Tensor = self.controller(controller_input)  # type: ignore[operator]
        controller_hidden: Tensor = self.controller.last_hidden  # type: ignore[union-attr]

        # Head params from separate FCs
        read_params: Tensor = self.read_fc(controller_hidden)
        write_params: Tensor = self.write_fc(controller_hidden)

        # Read head
        new_read_addr, read_output, _ = forward_read_head(
            self.memory, self._current_read_addr, read_params, self.m
        )

        # Write head
        new_write_addr, new_memory = forward_write_head(
            self.memory, self._current_write_addr, write_params, self.m
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
            "read_params": read_params.detach().clone(),
            "write_params": write_params.detach().clone(),
        }

        # Output from controller hidden + read vector
        output = self.output_fc(torch.cat([controller_hidden, read_output]))
        return output
