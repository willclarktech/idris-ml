#!/usr/bin/env python3
"""Remap dtag literals at the trailing position of streamed-create / cast /
one_hot calls to the new kind-major layout. Tracks paren depth so nested
calls like `heap_copy(xv, 3)` don't confuse the dtag-arg identification."""

import re
import sys

OLD_TO_NEW = {
    0:  14,  # F32
    1:  15,  # F64
    2:  17,  # BF16
    3:  13,  # F16
    4:  8,   # I8
    5:  9,   # I16
    6:  10,  # I32
    7:  11,  # I64
    8:  4,   # U8
    9:  1,   # Bool
}

CALL_NAMES = re.compile(
    r'\btensor_(?:create_\w*_streamed|cast_dtype_streamed|one_hot)\('
)

def find_streamed_calls(text):
    """Yield (dtag_abs_start, dtag_abs_end, old_dtag) for each call whose
    last positional arg is an integer literal in OLD_TO_NEW."""
    for m in CALL_NAMES.finditer(text):
        open_idx = m.end() - 1
        depth = 1
        i = open_idx + 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            continue
        close_idx = i
        args = text[open_idx + 1:close_idx]
        depth = 0
        last_comma = -1
        for j, ch in enumerate(args):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                last_comma = j
        if last_comma == -1:
            continue
        dtag_part = args[last_comma + 1:]
        m2 = re.match(r'(\s*)(\d+)(\s*)$', dtag_part)
        if not m2:
            continue
        prefix, dtag_str, suffix = m2.groups()
        dtag = int(dtag_str)
        if dtag not in OLD_TO_NEW:
            continue
        dtag_abs_start = open_idx + 1 + last_comma + 1 + len(prefix)
        dtag_abs_end = dtag_abs_start + len(dtag_str)
        yield (dtag_abs_start, dtag_abs_end, dtag)

def remap(text):
    edits = list(find_streamed_calls(text))
    edits.sort()
    result = []
    cursor = 0
    for start, end, old in edits:
        result.append(text[cursor:start])
        result.append(str(OLD_TO_NEW[old]))
        cursor = end
    result.append(text[cursor:])
    return ''.join(result), len(edits)

if __name__ == '__main__':
    for path in sys.argv[1:]:
        with open(path) as f:
            text = f.read()
        new, n = remap(text)
        with open(path, 'w') as f:
            f.write(new)
        print(f"{path}: {n} dtag literals remapped")
