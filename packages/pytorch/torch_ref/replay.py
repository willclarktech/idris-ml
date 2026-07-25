"""Writer for the `Ml.Rng.loadReplay` recording format.

One draw per line, `<channel> <value>`; blank lines and `#` lines are
ignored by the reader. Channels interleave freely — only the order within
a channel matters, and it is the consumption order of the run being
recorded. `repr()` prints the shortest decimal that round-trips to the
same IEEE double, so the reader recovers each draw bit-exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def write_replay(
    path: str,
    *,
    env: Iterable[float] = (),
    choices: Iterable[int] = (),
    uniforms: Iterable[float] = (),
    normals: Iterable[float] = (),
) -> None:
    """Write a replay file: `env` feeds the environment `Source` (reset
    draws), `choices` the categorical decisions, `uniforms`/`normals` the
    remaining `Rng` channels."""
    lines = ["# idris-ml replay (Ml.Rng.loadReplay): one <channel> <value> per line"]
    lines += [f"env {float(v)!r}" for v in env]
    lines += [f"choice {int(c)}" for c in choices]
    lines += [f"uniform {float(v)!r}" for v in uniforms]
    lines += [f"normal {float(v)!r}" for v in normals]
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"replay: wrote {path} ({len(lines) - 1} draws)")
