"""Init-manifest dump, the reference half of the alignment gate.

`maybe_dump_init(model)` writes the model's `state_dict` to safetensors and
exits when `IDRISML_DUMP_INIT` names a path. Call it straight after model
construction, before the optimizer touches anything, so the file holds the
*initial* weights. `scripts/check-init-manifest.py` runs both sides that way
and diffs the shapes and init moments.

Same env-var contract as Idris `Ml.Checkpoint.maybeDumpInit`, so one runner
drives both sides. Exiting rather than returning keeps the dump honest — a run
that continued to train could report a manifest taken after the first step.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

# safetensors ships no py.typed marker, so pyright sees the signature as
# partially unknown. The call below is fully annotated on our side.
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

if TYPE_CHECKING:
    import torch.nn as nn

ENV_VAR = "IDRISML_DUMP_INIT"


def maybe_dump_init(model: nn.Module) -> None:
    """Dump `model`'s initial state to safetensors and exit, if asked to."""
    path = os.environ.get(ENV_VAR)
    if not path:
        return
    # `.contiguous()`: safetensors rejects views, and a few reference models
    # hold transposed or sliced parameters.
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, path)
    print(f"{ENV_VAR}: wrote {path}")
    sys.exit(0)
