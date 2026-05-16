#!/usr/bin/env python3
"""
One-shot generator: split mlx tensor_backward's switch into per-op
replay functions sitting next to their forward kernels.

Reads packages/backends/backend_mlx/training/backward.cpp, extracts each
case body, and appends:
    static void mlx_replay_<lower>(std::vector<mx::array>& pool, TapeEntry& e)
    + MLX_REGISTER_REPLAY(OP_*, mlx_replay_<lower>)
to the owning forward-kernel .cpp file (per the OP_TO_FILE map below).

Shared-body cases (e.g. `case OP_MM: case OP_BMM: case OP_BMM_3X3: pool[out] =
mx::matmul(a, b); break;`) collapse into one replay function with multiple
REGISTER macros pointing at it.

Run once, eyeball the diff, build tri-link green, delete this script.
"""
from __future__ import annotations
import re
from pathlib import Path

REPO     = Path(__file__).resolve().parent.parent
MLX_ROOT = REPO / "packages/backends/backend_mlx"
BACKWARD = MLX_ROOT / "training/backward.cpp"

# OP_* -> path relative to MLX_ROOT. Resolved from `grep tape_append(OP_`.
OP_TO_FILE = {
    "OP_ADD":             "core/elementwise/add.cpp",
    "OP_SUB":             "core/elementwise/sub.cpp",
    "OP_MUL":             "core/elementwise/mul.cpp",
    "OP_DIV":             "core/elementwise/div.cpp",
    "OP_NEG":             "core/elementwise/neg.cpp",
    "OP_ABS":             "core/elementwise/abs.cpp",
    "OP_EXP":             "core/elementwise/exp.cpp",
    "OP_LOG":             "core/elementwise/log.cpp",
    "OP_SQRT":            "core/elementwise/sqrt.cpp",
    "OP_POW":             "core/elementwise/pow.cpp",
    "OP_SIGMOID":         "core/elementwise/sigmoid.cpp",
    "OP_TANH":            "core/elementwise/tanh.cpp",
    "OP_SOFTPLUS":        "core/elementwise/softplus.cpp",
    "OP_ADD_SCALAR":      "core/scalar/add_scalar.cpp",
    "OP_MUL_SCALAR":      "core/scalar/mul_scalar.cpp",
    "OP_CLAMP_MIN":       "core/scalar/clamp_min.cpp",
    "OP_CAST_DTYPE":      "core/lifecycle/cast.cpp",
    "OP_GELU":            "nn/activation/gelu.cpp",
    "OP_LEAKY_RELU":      "nn/activation/leaky_relu.cpp",
    "OP_SILU":            "nn/activation/silu.cpp",
    "OP_COSINE_SIM":      "nn/attention/cosine_similarity.cpp",
    "OP_EMBEDDING":       "nn/attention/embedding.cpp",
    "OP_MASKED_FILL":     "nn/mask/masked_fill.cpp",
    "OP_BATCH_NORM":      "nn/norm/batch_norm.cpp",
    "OP_DROPOUT":         "nn/norm/dropout.cpp",
    "OP_LAYER_NORM_2D":   "nn/norm/layer_norm.cpp",
    "OP_GRU_CELL":        "nn/recurrent/gru_cell.cpp",
    "OP_SOFTMAX_2D":      "nn/softmax/softmax.cpp",
    "OP_SOFTMAX_3D":      "nn/softmax/softmax.cpp",
    "OP_LOG_SOFTMAX_2D":  "nn/softmax/log_softmax.cpp",
    "OP_CAT":             "linear/concat/cat.cpp",
    "OP_CAT_MULTI":       "linear/concat/cat.cpp",
    "OP_CONCAT_2D_AXIS1": "linear/concat/concat_2d_axis1.cpp",
    "OP_STACK":           "linear/concat/stack.cpp",
    "OP_GATHER":          "linear/index/gather.cpp",
    "OP_SCATTER_ADD":     "linear/index/scatter_add.cpp",
    "OP_LINEAR_2D":       "linear/linalg/linear.cpp",
    "OP_MM":              "linear/linalg/matmul.cpp",
    "OP_BMM":             "linear/linalg/mm.cpp",
    "OP_BMM_3X3":         "linear/linalg/mm.cpp",
    "OP_MV":              "linear/linalg/mv.cpp",
    "OP_OUTER":           "linear/linalg/outer.cpp",
    "OP_TILE_2D":         "linear/linalg/tile.cpp",
    "OP_TRANSPOSE_2D":    "linear/linalg/transpose.cpp",
    "OP_TRANSPOSE_LAST2": "linear/linalg/transpose.cpp",
    "OP_MEAN":            "linear/reduction/mean.cpp",
    "OP_SUM":             "linear/reduction/sum.cpp",
    "OP_SUM_DIM":         "linear/reduction/sum.cpp",
    "OP_NARROW":          "linear/shape/narrow.cpp",
    "OP_RESHAPE":         "linear/shape/reshape.cpp",
    "OP_SELECT":          "linear/shape/select.cpp",
    "OP_CUMPROD":         "linear/sort/cumprod.cpp",
    "OP_AVG_POOL1D":      "conv/avg_pool1d.cpp",
    "OP_AVG_POOL2D":      "conv/avg_pool2d.cpp",
    "OP_CONV1D":          "conv/conv1d.cpp",
    "OP_CONV1D_CIRC":     "conv/conv1d_circular.cpp",
    "OP_CONV2D":          "conv/conv2d.cpp",
    "OP_CONV2D_BATCHED":  "conv/conv2d.cpp",
    "OP_MAX_POOL1D":      "conv/max_pool1d.cpp",
    "OP_MAX_POOL2D":      "conv/max_pool2d.cpp",
    "OP_MAX_POOL2D_BATCHED": "conv/max_pool2d_batched.cpp",
}


def extract_switch_body(text: str) -> str:
    """Return the source between `switch (e.op) {` and its matching `}`."""
    marker = "switch (e.op) {"
    start = text.index(marker) + len(marker)
    depth = 1
    i = start
    while depth > 0 and i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def parse_cases(switch_body: str) -> list[tuple[list[str], str]]:
    """Walk the switch body and yield (op_tags_sharing_body, body_src) groups.

    Tracks brace depth so we don't misparse a `case OP_X:` token appearing
    inside a nested string/comment/block; the cases live at depth 0.
    """
    case_re = re.compile(r"\bcase\s+(OP_[A-Z0-9_]+)\s*:")
    raw = []  # list of (op_tag, body_string)
    matches = list(case_re.finditer(switch_body))
    for idx, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(switch_body)
        raw.append((m.group(1), switch_body[body_start:body_end]))

    # Collapse empty-body cases (the `case OP_X: case OP_Y: <body>` pattern)
    # into the next non-empty group.
    groups: list[tuple[list[str], str]] = []
    pending: list[str] = []
    for op_tag, body in raw:
        if body.strip() == "":
            pending.append(op_tag)
            continue
        pending.append(op_tag)
        groups.append((pending, body))
        pending = []
    if pending:
        raise RuntimeError(f"trailing empty cases with no body: {pending}")
    return groups


def clean_body(body: str) -> str:
    """Strip leading whitespace, outer braces if present, and the trailing
    `break;` / `default: break;` from a case body."""
    s = body.strip()
    # Strip a trailing `default: break;` that may follow the last real case.
    s = re.sub(r"\bdefault\s*:\s*break\s*;\s*$", "", s).rstrip()
    # If the body is wrapped in a single outer brace pair, strip it.
    if s.startswith("{") and s.endswith("}"):
        # Be sure the brace pair matches (not e.g. `{n}` from an mx::zeros literal).
        depth = 0
        first_close = -1
        for i, c in enumerate(s):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    first_close = i
                    break
        if first_close == len(s) - 1:
            s = s[1:-1].strip()
    # Trim trailing `break;`
    s = re.sub(r"\s*break\s*;\s*$", "", s)
    return s


SCALAR_HELPERS = re.compile(r"\b(scalar_like|zero_like|one_like|half_like|kF32_ZERO)\b")


def replay_source(ops: list[str], raw_body: str) -> tuple[str, str]:
    """Return (source_block, fn_name) for an op group."""
    primary = ops[0]
    fn_name = "mlx_replay_" + primary.removeprefix("OP_").lower()
    body = clean_body(raw_body)
    indented = "\n".join(("    " + line) if line else "" for line in body.split("\n"))
    src = (
        f"static void {fn_name}(std::vector<mx::array>& pool, TapeEntry& e) {{\n"
        f"    int out = e.result->pool_idx;\n"
        f"    [[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();\n"
        f"    [[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();\n"
        f"{indented}\n"
        f"}}\n"
    )
    registers = "\n".join(f"MLX_REGISTER_REPLAY({op}, {fn_name})" for op in ops)
    return src + registers + "\n", fn_name


def patch_file(rel_path: str, blocks: list[tuple[str, list[str]]]) -> None:
    """Insert the dispatch include + (conditional) precision include after the
    last existing #include, then append each block at end of file."""
    path = MLX_ROOT / rel_path
    text = path.read_text()
    depth = rel_path.count("/")
    dispatch_inc = f'#include "{"../" * depth}training/autograd/op_dispatch.h"'
    precision_inc = f'#include "{"../" * depth}precision.h"'

    needs_precision = any(SCALAR_HELPERS.search(b) for b, _ in blocks)

    inc_lines = re.findall(r"^#include .*$", text, flags=re.MULTILINE)
    if not inc_lines:
        raise RuntimeError(f"{rel_path}: no existing #include found")
    last_inc = inc_lines[-1]
    last_pos = text.rfind(last_inc) + len(last_inc)

    additions = []
    if dispatch_inc not in text:
        additions.append(dispatch_inc)
    if needs_precision and precision_inc not in text:
        additions.append(precision_inc)
    if additions:
        text = text[:last_pos] + "\n" + "\n".join(additions) + text[last_pos:]

    text = text.rstrip() + "\n\n" + "\n".join(b for b, _ in blocks)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text)


def main() -> None:
    text = BACKWARD.read_text()
    switch_body = extract_switch_body(text)
    groups = parse_cases(switch_body)

    file_blocks: dict[str, list[tuple[str, list[str]]]] = {}
    skipped = []
    for ops, body in groups:
        if ops == ["OP_CONST"] and clean_body(body) == "":
            # Leaf marker — no replay needed.
            skipped.append(ops)
            continue
        if not any(op in OP_TO_FILE for op in ops):
            print(f"WARN: no file mapping for {ops}")
            continue
        primary = next(op for op in ops if op in OP_TO_FILE)
        rel = OP_TO_FILE[primary]
        src, _ = replay_source(ops, body)
        file_blocks.setdefault(rel, []).append((src, ops))

    for rel, blocks in sorted(file_blocks.items()):
        op_count = sum(len(ops) for _, ops in blocks)
        print(f"  {rel:48s}  +{op_count} op(s)")
        patch_file(rel, blocks)

    print(f"\nTotal files patched: {len(file_blocks)}")
    print(f"Total ops registered: {sum(len(ops) for blocks in file_blocks.values() for _, ops in blocks)}")
    print(f"Skipped (no replay needed): {skipped}")


if __name__ == "__main__":
    main()
