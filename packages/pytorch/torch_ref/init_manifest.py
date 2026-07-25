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
    from torch import Tensor

ENV_VAR = "IDRISML_DUMP_INIT"


def maybe_dump_init(*models: nn.Module) -> None:
    """Dump the models' initial state to safetensors and exit, if asked to.

    Pass every network the example builds, in the order the Idris side
    constructs them: its registry is one flat list across all of them, and the
    gate compares the ordered shape sequence. An actor/critic pair dumped in
    the wrong order reads as a shape mismatch.

    `named_parameters()`, not `state_dict()`: the latter also carries buffers
    (MultiHeadTransformer's causal mask, its positional encoding), which the
    Idris param registry does not hold and which are not init at all.

    Keys are prefixed by model index so two networks with the same layer names
    do not collide. Names are not compared by the gate — the Idris registry
    derives its own from the init scope — so the prefix costs nothing.
    """
    path = os.environ.get(ENV_VAR)
    if not path:
        return
    tensors: dict[str, Tensor] = {}
    for i, model in enumerate(models):
        for k, v in model.named_parameters():
            # `.clone()`, not just `.contiguous()`: safetensors refuses to write
            # tensors that share storage, and MultiHeadTransformer hands the
            # same causal-mask buffer to every block.
            tensors[f"{i}.{k}"] = v.detach().cpu().contiguous().clone()
    save_file(tensors, path)
    print(f"{ENV_VAR}: wrote {path} ({len(tensors)} tensors from {len(models)} model(s))")
    sys.exit(0)


DATA_ENV_VAR = "IDRISML_DUMP_DATA"


def maybe_dump_batch(x: Tensor, y: Tensor) -> None:
    """Print one batch's shape and moments and exit, if asked to.

    The reference half of the data-generator gate. Values cannot be compared
    across the two sides (the RNGs differ), so this reports what can be:
    shape, mean, std, min, max. `Example.SeqClassify` generated noise-free
    waveforms while this side added N(0, 0.1) per timestep — the two trained
    on different difficulties for months, and no init or metric check could
    see it.
    """
    if not os.environ.get(DATA_ENV_VAR):
        return
    for tag, t in (("input", x), ("target", y)):
        f = t.detach().cpu().double().flatten()
        # Lag-1 difference std: see Ml.Checkpoint.tensorMoments for why the
        # batch mean/std alone cannot see additive noise on a structured
        # signal.
        d1 = float((f[1:] - f[:-1]).std(unbiased=False)) if f.numel() > 1 else 0.0
        print(
            f"DATA_MANIFEST\t{tag}\t{list(t.shape)}\t{float(f.mean())}\t"
            f"{float(f.std(unbiased=False))}\t{d1}\t{float(f.min())}\t{float(f.max())}"
        )
    sys.exit(0)
