"""Per-block divergence report between the Idris-side --bisect-blocks
dump and the HF oracle's per-block hidden-state captures.

Inputs:
  - models/bitnet-2b-4t-bisect/<label>.safetensors  (HF oracle, F32)
  - models/idris-bisect/<label>.txt                  (Idris dump,
                                                       one float per
                                                       line, in row-
                                                       major order)

Labels: "embedding", "block_00".."block_29", "final_norm", optionally
"logits" (the latter matches bitnet-2b-4t-oracle.safetensors content).

Output: a table with per-label stats — the first label whose max-rel-
diff exceeds the threshold is the divergence point.

Usage (from repo root):
    uv run --directory packages/pytorch python \\
        packages/idris-transformers/scripts/compare_bitnet_blocks.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.torch import load_file  # pyright: ignore[reportUnknownVariableType]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
ORACLE_DIR = REPO_ROOT / "models" / "bitnet-2b-4t-bisect"
IDRIS_DIR = REPO_ROOT / "models" / "idris-bisect"

# Labels in the order the Idris example emits them.
LABELS = ["embedding"] + [f"block_{i:02d}" for i in range(30)] + ["final_norm", "logits"]


def load_idris(label: str) -> np.ndarray | None:
    p = IDRIS_DIR / f"{label}.txt"
    if not p.is_file():
        return None
    vals: list[float] = []
    with p.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("[") or s.startswith("param_load") or s.startswith(" "):
                # diagnostic / stage line, skip
                continue
            try:
                vals.append(float(s))
            except ValueError:
                continue
    return np.array(vals, dtype=np.float64)


def load_oracle(label: str) -> np.ndarray | None:
    p = ORACLE_DIR / f"{label}.safetensors"
    if not p.is_file():
        return None
    t = load_file(str(p))["output"]
    return t.to(dtype=__import__("torch").float64).cpu().numpy().reshape(-1)


def fmt_row(
    label: str,
    n_idris: int | None,
    n_oracle: int | None,
    max_abs: float | None,
    max_rel: float | None,
    verdict: str,
) -> str:
    nstr = f"{n_idris}/{n_oracle}" if n_idris is not None and n_oracle is not None else "—"
    abs_s = f"{max_abs:.3e}" if max_abs is not None else "   —    "
    rel_s = f"{max_rel:.3e}" if max_rel is not None else "   —    "
    return f"  {label:14}  count={nstr:14}  max_abs={abs_s}  max_rel={rel_s}  {verdict}"


def main() -> int:
    print("Per-block divergence report")
    print("=" * 80)
    print(f"  oracle dir: {ORACLE_DIR}")
    print(f"  idris dir:  {IDRIS_DIR}")
    print()

    first_diverge: str | None = None
    threshold = 0.10  # max-rel-diff that counts as a real divergence (vs BF16 round-off)

    rows: list[str] = []
    for label in LABELS:
        idris = load_idris(label)
        oracle = load_oracle(label)

        if idris is None or oracle is None:
            verdict = "MISSING"
            rows.append(
                fmt_row(
                    label,
                    len(idris) if idris is not None else None,
                    len(oracle) if oracle is not None else None,
                    None,
                    None,
                    verdict,
                )
            )
            continue

        if len(idris) != len(oracle):
            verdict = "COUNT MISMATCH"
            rows.append(fmt_row(label, len(idris), len(oracle), None, None, verdict))
            continue

        diff = np.abs(idris - oracle)
        denom = np.maximum(np.abs(oracle), 1e-6)
        rel = diff / denom

        max_abs = float(diff.max())
        max_rel = float(rel.max())

        if max_rel > threshold:
            verdict = "DIVERGED"
            if first_diverge is None:
                first_diverge = label
        else:
            verdict = "ok"

        rows.append(fmt_row(label, len(idris), len(oracle), max_abs, max_rel, verdict))

    print("\n".join(rows))
    print()
    if first_diverge is not None:
        print(f"First divergence: {first_diverge}  (max_rel_diff > {threshold})")
        return 1
    else:
        all_ok = all("ok" in r or "MISSING" in r for r in rows)
        if all_ok:
            print("All labels within tolerance (or missing).")
            return 0
        print("Some labels failed; see above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
