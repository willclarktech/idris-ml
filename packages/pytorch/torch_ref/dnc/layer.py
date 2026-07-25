"""DNC layer: LSTM controller + external memory with temporal links.

Graves et al. 2016, "Hybrid computing using a neural network with dynamic
external memory". Extends NTM with:
- Dynamic memory allocation (usage-based)
- Temporal link matrix (forward/backward traversal)
- Multiple read heads with 3-way mode mixture
- Erase+add write (instead of interpolation write)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.dnc.addressing import (
    allocation_weighting,
    content_address,
    read_weighting,
    update_link_matrix,
    update_usage,
    write_weighting,
)
from torch_ref.dnc.memory import erase_add_write, read_op
from torch_ref.ntm.controller import LSTMController


class DNCLayer(nn.Module):
    """Differentiable Neural Computer layer.

    Args:
        n: Number of memory slots.
        m: Memory width per slot.
        num_reads: Number of read heads (R).
        num_inputs: External input width.
        num_outputs: Output width.
        controller_hidden_size: LSTM hidden dimension (h).
    """

    def __init__(
        self,
        n: int,
        m: int,
        num_reads: int,
        num_inputs: int,
        num_outputs: int,
        controller_hidden_size: int,
    ) -> None:
        super().__init__()
        self.n = n
        self.m = m
        self.num_reads = num_reads
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.h = controller_hidden_size

        # Controller: input = R read outputs + external input
        controller_input_size = num_reads * m + num_inputs
        self.controller = LSTMController(controller_input_size, controller_hidden_size)

        # Interface vector FCs (from controller hidden -> head params)
        # Write head params
        self.write_key_fc = nn.Linear(controller_hidden_size, m)
        self.write_beta_fc = nn.Linear(controller_hidden_size, 1)
        self.erase_fc = nn.Linear(controller_hidden_size, m)
        self.add_fc = nn.Linear(controller_hidden_size, m)
        self.free_gates_fc = nn.Linear(controller_hidden_size, num_reads)
        self.alloc_gate_fc = nn.Linear(controller_hidden_size, 1)
        self.write_gate_fc = nn.Linear(controller_hidden_size, 1)

        # Read head params (R heads)
        self.read_keys_fc = nn.Linear(controller_hidden_size, num_reads * m)
        self.read_betas_fc = nn.Linear(controller_hidden_size, num_reads)
        self.read_modes_fc = nn.Linear(controller_hidden_size, num_reads * 3)

        # Output: controller hidden + R read outputs -> output
        self.output_fc = nn.Linear(controller_hidden_size + num_reads * m, num_outputs)

        # Initialize weights
        for fc in [
            self.write_key_fc,
            self.write_beta_fc,
            self.erase_fc,
            self.add_fc,
            self.free_gates_fc,
            self.alloc_gate_fc,
            self.write_gate_fc,
            self.read_keys_fc,
            self.read_betas_fc,
            self.read_modes_fc,
        ]:
            nn.init.xavier_uniform_(fc.weight, gain=1.4)
            nn.init.normal_(fc.bias, std=0.01)
        # Output FC: He-uniform (kaiming_uniform_ default a=0) + normal bias.
        # NOT the shared dense contract: narrowing this to U(+-1/sqrt(fan_in))
        # measurably slowed recall convergence (tail-avg loss 0.66 against a
        # 0.60 bar), and NTM init is the tuned, stability-sensitive part of the
        # architecture. Idris `Nn.ntm` matches this bound.
        nn.init.kaiming_uniform_(self.output_fc.weight)
        nn.init.normal_(self.output_fc.bias, std=0.01)

        # Learned memory initialization
        self.memory_init = nn.Parameter(torch.empty(n * m))
        nn.init.xavier_uniform_(self.memory_init.data.view(n, m))

        # Fixed read output init (kaiming, non-learnable). Registered as
        # a buffer so .to(device) moves it alongside the parameters.
        read_out = torch.empty(num_reads, m)
        nn.init.kaiming_uniform_(read_out)
        self.register_buffer("_init_read_outputs", read_out)
        self._init_read_outputs: Tensor

    def reset_state(self) -> None:
        """Reset all state between sequences."""
        self.controller.reset_state()

        # Memory
        self.memory = torch.sigmoid(self.memory_init).view(self.n, self.m)

        # Addressing state (allocated on the active param device)
        device = self.memory_init.device
        self.usage = torch.zeros(self.n, device=device)
        self.write_weights = torch.zeros(self.n, device=device)
        self.read_weights = [torch.zeros(self.n, device=device) for _ in range(self.num_reads)]
        self.read_outputs = [self._init_read_outputs[i].clone() for i in range(self.num_reads)]

        # Temporal link state
        self.link_matrix = torch.zeros(self.n, self.n, device=device)
        self.precedence = torch.zeros(self.n, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """Forward one timestep.

        x: [num_inputs]
        Returns: [num_outputs]
        """
        # 1. Controller input: concat all read outputs + input
        controller_input = torch.cat(self.read_outputs + [x])
        self.controller(controller_input)
        hidden = self.controller.last_hidden

        # 2. Interface vector: all head params from controller hidden
        write_key = self.write_key_fc(hidden)
        write_beta = F.softplus(self.write_beta_fc(hidden).squeeze(-1))
        erase_vector = torch.sigmoid(self.erase_fc(hidden))
        add_vector = self.add_fc(hidden)
        free_gates = torch.sigmoid(self.free_gates_fc(hidden))
        alloc_gate = torch.sigmoid(self.alloc_gate_fc(hidden).squeeze(-1))
        write_gate = torch.sigmoid(self.write_gate_fc(hidden).squeeze(-1))

        read_keys = self.read_keys_fc(hidden).view(self.num_reads, self.m)
        read_betas = F.softplus(self.read_betas_fc(hidden))
        read_modes = self.read_modes_fc(hidden).view(self.num_reads, 3)

        # 3. Usage update
        self.usage = update_usage(
            self.usage,
            self.write_weights,
            free_gates,
            self.read_weights,
        )

        # 4. Allocation weighting
        alloc_w = allocation_weighting(self.usage)

        # 5. Write content addressing
        content_w = content_address(write_beta, self.memory, write_key)

        # 6. Write weighting
        self.write_weights = write_weighting(content_w, alloc_w, write_gate, alloc_gate)

        # 7. Memory write (erase + add)
        self.memory = erase_add_write(self.memory, self.write_weights, erase_vector, add_vector)

        # 8. Link matrix + precedence update
        self.link_matrix, self.precedence = update_link_matrix(
            self.link_matrix,
            self.precedence,
            self.write_weights,
        )

        # 9. Read heads
        new_read_weights: list[Tensor] = []
        new_read_outputs: list[Tensor] = []
        for i in range(self.num_reads):
            # Content addressing for this read head
            rc_w = content_address(read_betas[i], self.memory, read_keys[i])

            # Mode mixture: backward + content + forward
            rw = read_weighting(
                self.link_matrix,
                self.read_weights[i],
                rc_w,
                read_modes[i],
            )
            new_read_weights.append(rw)

            # Read from memory
            ro = read_op(rw, self.memory)
            new_read_outputs.append(ro)

        self.read_weights = new_read_weights
        self.read_outputs = new_read_outputs

        # 10. Output
        output = self.output_fc(torch.cat([hidden] + self.read_outputs))
        return output
