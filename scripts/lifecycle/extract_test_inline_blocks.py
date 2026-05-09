#!/usr/bin/env python3
"""Phase 0.2: extract T16-T33 inline scope blocks from main() in
packages/backends/test_backend.c into named static functions.

For each block of the form:
    /* T<NN>: <title> [...] */
[#if defined(BACKEND_FOO)]
    {
        <body>
    }
[#endif]

we:
  1. derive a snake_case function name from the explicit mapping below
  2. extract the body into `static void test_<name>(void) { <body> }`
     placed before `int main(void)`. The function is wrapped in the
     same #if guard if present (so torch-only blocks don't try to link
     against tape-only symbols on a tape build).
  3. replace the original block in main() with `test_<name>();` (also
     wrapped in the same #if guard); the /* T<NN>: ... */ comment
     above the call is preserved as in-context documentation.

Idempotent: detects already-extracted blocks (`test_<name>();` present)
and skips. Writes test_backend.c in place.

Run from repo root: `python3 scripts/lifecycle/extract_test_inline_blocks.py`
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "packages" / "backends" / "test_backend.c"

# Explicit T-marker -> function-name mapping. Disambiguates the reused
# T20 / T24 markers and produces predictable names.
# Format: (T-marker prefix on the /* T... line, function suffix)
MARKERS: list[tuple[str, str]] = [
    ("T16: Embedding",                            "embedding"),
    ("T17: Gather/Scatter",                       "gather_scatter"),
    ("T18: Argsort + Cumprod",                    "argsort_cumprod"),
    ("T19: LeakyReLU + SiLU activations",         "leaky_relu_silu_softplus"),
    ("T20: Per-param LR overrides",               "per_param_lr"),
    ("T21: min/max reductions",                   "min_max_reductions"),
    ("T22: squeeze",                              "squeeze"),
    ("T23: sum_dim with backward",                "sum_dim_backward"),
    ("T24a: stack with backward",                 "stack_backward"),
    ("T24b: cat with backward",                   "cat_backward"),
    ("T24c: batch — convenience wrapper",         "batch_convenience"),
    ("T24d: cat_from_array",                      "cat_from_array"),
    ("T24e: MSE loss with backward",              "mse_loss_backward"),
    ("T24f: cross-entropy loss with backward",    "cross_entropy_loss_backward"),
    ("T24g: LSTM gates (void-output variant)",    "lstm_gates_void_output"),
    ("T24h: LSTM cell",                           "lstm_cell"),
    ("T25: grad/detach/with_grad",                "grad_detach_with_grad"),
    ("T24: unbatch",                              "unbatch"),
    ("T20: Inference-only dtype scaffolding",     "inference_dtype_scaffolding_torch"),
    ("T27: unified dtag-dispatch create/cast",    "unified_dtag_dispatch"),
    ("T28: tape dtype storage scaffolding",       "tape_dtype_storage"),
    ("T29: F32 gradcheck oracle vs F64",          "tape_f32_gradcheck_oracle"),
    ("T30: F32 non-elementwise coverage",         "tape_f32_non_elementwise_coverage"),
    ("T31: inference-only dtype matrix",          "tape_inference_dtype_matrix"),
    ("T32: F32 cast → tensor_to_doubles",         "tape_f32_cast_readout_agreement"),
    ("T33: RuntimeDType tag layout",              "runtime_dtype_tag_layout"),
]


def find_marker_line(lines: list[str], prefix: str, start: int) -> int:
    """Find the 1-indexed line containing the marker prefix, searching from `start`."""
    needle = "/* " + prefix
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    raise SystemExit(f"Marker not found: {prefix!r} from line {start}")


def find_comment_end(lines: list[str], comment_start: int) -> int:
    """Return last index (0-based) of the /* ... */ block starting at comment_start."""
    for j in range(comment_start, len(lines)):
        if "*/" in lines[j]:
            return j
    raise SystemExit(f"Unterminated comment starting at line {comment_start+1}")


def find_open_brace(lines: list[str], comment_end: int) -> tuple[int, str | None]:
    """Skip over an optional `#if defined(...)` and find the `{` line.
    Returns (brace_line_index, ifdef_text_or_None)."""
    i = comment_end + 1
    ifdef = None
    while i < len(lines):
        s = lines[i].strip()
        if s == "":
            i += 1
            continue
        if s.startswith("#if "):
            ifdef = lines[i].rstrip("\n")
            i += 1
            continue
        if s == "{":
            return i, ifdef
        # Otherwise something unexpected — likely we walked into the next block.
        return -1, None
    return -1, None


def find_matching_close_brace(lines: list[str], open_idx: int) -> int:
    """Brace counter ignoring contents of //, /* */, "..." and '...' literals."""
    depth = 0
    in_block_comment = False
    in_line_comment = False
    in_string = None  # '"' or "'"
    for i in range(open_idx, len(lines)):
        line = lines[i]
        j = 0
        in_line_comment = False
        while j < len(line):
            c = line[j]
            nxt = line[j+1] if j+1 < len(line) else ""
            if in_block_comment:
                if c == "*" and nxt == "/":
                    in_block_comment = False
                    j += 2
                    continue
                j += 1
                continue
            if in_line_comment:
                break
            if in_string:
                if c == "\\" and j+1 < len(line):
                    j += 2
                    continue
                if c == in_string:
                    in_string = None
                j += 1
                continue
            if c == "/" and nxt == "*":
                in_block_comment = True
                j += 2
                continue
            if c == "/" and nxt == "/":
                in_line_comment = True
                break
            if c == '"' or c == "'":
                in_string = c
                j += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
            j += 1
    raise SystemExit(f"Unbalanced brace from line {open_idx+1}")


def find_optional_endif(lines: list[str], close_idx: int) -> int:
    """If the next non-blank line is `#endif`, return its index; else return close_idx."""
    i = close_idx + 1
    while i < len(lines):
        s = lines[i].strip()
        if s == "":
            i += 1
            continue
        if s == "#endif":
            return i
        return close_idx
    return close_idx


def dedent_block(body: str) -> str:
    """Re-indent: existing body is indented to 8 spaces inside main's brace+block.
    Drop 4 spaces so the contents fit under a function indented at 4."""
    out_lines = []
    for ln in body.splitlines():
        if ln.startswith("    "):
            out_lines.append(ln[4:])
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)


def main() -> int:
    if not FILE.exists():
        print(f"Not found: {FILE}", file=sys.stderr)
        return 2
    text = FILE.read_text()
    lines = text.splitlines(keepends=True)

    # Find the start of `int main(`
    main_idx = next((i for i, l in enumerate(lines) if l.startswith("int main(")), None)
    if main_idx is None:
        print("int main( not found", file=sys.stderr)
        return 2

    # Find the matching `}` for main so we can locate the insertion point for new fns
    main_open = next((i for i in range(main_idx, len(lines)) if lines[i].rstrip().endswith("{")), None)
    if main_open is None:
        print("opening { for main not found", file=sys.stderr)
        return 2

    # Collect block spans in original order
    blocks = []
    cursor = main_open + 1
    for prefix, fn_suffix in MARKERS:
        comment_line = find_marker_line(lines, prefix, cursor)
        comment_end = find_comment_end(lines, comment_line)
        brace_idx, ifdef = find_open_brace(lines, comment_end)
        if brace_idx < 0:
            print(f"Open brace not found after marker {prefix!r}", file=sys.stderr)
            return 2
        close_idx = find_matching_close_brace(lines, brace_idx)
        end_idx = find_optional_endif(lines, close_idx)
        blocks.append({
            "prefix": prefix,
            "fn_suffix": fn_suffix,
            "comment_start": comment_line,
            "comment_end": comment_end,
            "brace_open": brace_idx,
            "brace_close": close_idx,
            "block_end": end_idx,   # may equal close_idx or #endif line
            "ifdef": ifdef,         # full line text incl '#if defined(...)'
        })
        cursor = end_idx + 1

    # Idempotency check
    name = f"test_{blocks[0]['fn_suffix']}"
    if any(f"static void {name}(void)" in l for l in lines):
        print(f"Already extracted ({name} present). Nothing to do.")
        return 0

    # Build the function definitions
    new_fn_text_parts: list[str] = []
    for b in blocks:
        # Body lines = brace_open+1 .. brace_close-1
        body_raw = "".join(lines[b["brace_open"]+1 : b["brace_close"]])
        body = dedent_block(body_raw).rstrip() + "\n"
        # Drop trailing blank lines
        while body.endswith("\n\n"):
            body = body[:-1]

        fn_lines: list[str] = []
        if b["ifdef"]:
            fn_lines.append(b["ifdef"] + "\n")
        fn_lines.append(f"static void test_{b['fn_suffix']}(void) {{\n")
        fn_lines.append(body if body.endswith("\n") else body + "\n")
        fn_lines.append("}\n")
        if b["ifdef"]:
            fn_lines.append("#endif\n")
        fn_lines.append("\n")
        new_fn_text_parts.append("".join(fn_lines))

    new_fns_text = "".join(new_fn_text_parts)

    # Build replacement text inside main() for each block.
    # We replace from `comment_start` through `block_end` inclusive with:
    #   <preserved comment lines (the /* T<NN>: ... */)>
    #   [#if defined(...)]
    #       test_<name>();
    #   [#endif]
    # Replacements are applied bottom-up so earlier line indices stay valid.

    # New text built by emitting lines around the spans.
    blocks_sorted = sorted(blocks, key=lambda b: b["comment_start"])

    out_lines: list[str] = []
    write_idx = 0
    for b in blocks_sorted:
        # Emit unchanged lines up to comment_start
        out_lines.extend(lines[write_idx : b["comment_start"]])
        # Preserve original comment lines
        out_lines.extend(lines[b["comment_start"] : b["comment_end"]+1])
        # Replacement
        if b["ifdef"]:
            out_lines.append(b["ifdef"] + "\n")
        out_lines.append(f"    test_{b['fn_suffix']}();\n")
        if b["ifdef"]:
            out_lines.append("#endif\n")
        write_idx = b["block_end"] + 1
    # Tail
    out_lines.extend(lines[write_idx:])

    # Insert generated function definitions just before `int main(`
    new_lines: list[str] = []
    inserted = False
    for ln in out_lines:
        if not inserted and ln.startswith("int main("):
            new_lines.append(new_fns_text)
            inserted = True
        new_lines.append(ln)

    FILE.write_text("".join(new_lines))
    print(f"Extracted {len(blocks)} blocks → static functions in {FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
