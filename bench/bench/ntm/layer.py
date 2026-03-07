"""NTM layer: controller + memory + read/write heads.

Head parameters are produced by separate linear layers from the controller
cell state. Memory uses interpolation write (no erase vector). Memory and
controller state are learned via FC+sigmoid from a dummy input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bench.ntm.addressing import addressing_params_width
from bench.ntm.memory import forward_read_head, forward_write_head


class NTMLayer(nn.Module):
    """Neural Turing Machine layer.

    Args:
        controller: Recurrent controller (e.g. LSTM). Must expose .last_hidden,
                    .last_cell properties and .reset_state() method.
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

        # Head FC layers (from controller cell state → head params)
        read_params_w = addressing_params_width(m)
        write_params_w = addressing_params_width(m) + m  # addressing + add vector

        self.read_fc = nn.Linear(controller_hidden_size, read_params_w)
        self.write_fc = nn.Linear(controller_hidden_size, write_params_w)

        # Output FC: controller hidden + read vector → output
        self.output_fc = nn.Linear(controller_hidden_size + m, num_outputs)

        # Head FCs: xavier gain=1.4, normal bias
        for fc in [self.read_fc, self.write_fc]:
            nn.init.xavier_uniform_(fc.weight, gain=1.4)
            nn.init.normal_(fc.bias, std=0.01)
        # Output FC: kaiming
        nn.init.kaiming_uniform_(self.output_fc.weight)
        nn.init.normal_(self.output_fc.bias, std=0.01)

        # Learned memory initialization (FC+sigmoid from dummy)
        self.memory_bias_fc = nn.Linear(1, n * m)
        self.memory: Tensor = torch.full((n, m), 1e-6)

    def reset_state(self) -> None:
        """Reset memory and head state between sequences."""
        # Learned memory init
        dummy = torch.tensor([[0.0]])
        memory_bias = F.sigmoid(self.memory_bias_fc(dummy))
        self.memory = memory_bias.view(self.n, self.m)

        self._current_read_addr = torch.zeros(self.n)
        self._current_write_addr = torch.zeros(self.n)

        # Read head output: kaiming init
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

        # Head params from cell state
        head_fc_input: Tensor = self.controller.last_cell  # type: ignore[union-attr]
        read_params: Tensor = self.read_fc(head_fc_input)
        write_params: Tensor = self.write_fc(head_fc_input)

        # Read head
        new_read_addr, read_output, _ = forward_read_head(
            self.memory, self._current_read_addr, read_params, self.m
        )

        # Write head (interpolation, no erase)
        new_write_addr, new_memory = forward_write_head(
            self.memory, self._current_write_addr, write_params, self.m
        )

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
