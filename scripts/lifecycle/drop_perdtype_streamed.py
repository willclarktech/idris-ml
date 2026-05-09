#!/usr/bin/env python3
"""One-shot: delete the superseded per-dtype *_streamed symbol blocks now
that the unified tensor_create_<shape>_streamed(..., int dtag) symbols carry
everything. Deletes by start/end marker substrings; prints the removed range
head+tail for review. Keeps create_*_dt helpers and the non-streamed
per-dtype creators (used by the unified switch)."""
import sys

# (path, start_substr, end_substr, end_inclusive)
BLOCKS = [
    # backend.h: the per-dtype + inference *_streamed declaration block.
    ("packages/backends/backend.h",
     "L60 dtype-cascade stream wrappers ----",
     "TensorHandle tensor_cast_dtype_bool_streamed(TensorHandle src, int stream_tag);",
     True),
    # torch: f32/f64 streamed wrappers (keep is_floating + create_*_dt after).
    ("packages/backends/backend_torch.cpp",
     "L60 dtype-cascade stream wrappers (no-op stream on torch)",
     "Inference-only dtype scaffolding (BF16, F16, Int, Bool)",
     False),
    # torch: the inference IDRISML_DEFINE_DTYPE_STREAMED macro + invocations.
    ("packages/backends/backend_torch.cpp",
     "// Each pasted name (e.g. tensor_create_1d_bf16_streamed)",
     "#undef IDRISML_DEFINE_DTYPE_STREAMED",
     True),
    # mlx: the *_streamed forwarding wrappers (keep *_mlx_streamed internals).
    ("packages/backends/backend_mlx.cpp",
     "L60 dtype-cascade stream wrappers (forwards to *_mlx_streamed)",
     "Unified dtag-dispatch create/cast entry points",
     False),
    # tape: the *_streamed trampolines (keep non-streamed f32/f64 + abort stubs).
    ("packages/backends/backend_tape.c",
     "L60 dtype-cascade stream wrappers (no-op stream on tape)",
     "Unified dtag-dispatch create/cast entry points",
     False),
]


def find_line(lines, substr, start=0):
    for i in range(start, len(lines)):
        if substr in lines[i]:
            return i
    raise SystemExit(f"marker not found: {substr!r}")


def main():
    # group edits per file, apply bottom-up so line indices stay valid
    from collections import defaultdict
    per_file = defaultdict(list)
    for path, s, e, inc in BLOCKS:
        per_file[path].append((s, e, inc))

    for path, blocks in per_file.items():
        with open(path) as f:
            lines = f.readlines()
        ranges = []
        for s, e, inc in blocks:
            si = find_line(lines, s)
            ei = find_line(lines, e, si)
            end = ei if inc else ei - 1
            # trim a single trailing blank line inside the removed region
            while end > si and lines[end].strip() == "":
                end -= 1
            ranges.append((si, end))
        ranges.sort(reverse=True)
        for si, end in ranges:
            print(f"{path}: removing lines {si+1}-{end+1} "
                  f"({end-si+1} lines)\n    head: {lines[si].rstrip()}\n"
                  f"    tail: {lines[end].rstrip()}")
            del lines[si:end+1]
        with open(path, "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    main()
