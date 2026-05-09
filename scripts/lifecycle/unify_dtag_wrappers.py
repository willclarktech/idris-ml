#!/usr/bin/env python3
"""One-shot: collapse the per-dtype dtag dispatch inside the create/cast
Scheme wrappers (Device/{Tape,Torch,Mlx}.idr) into a single pass-through
call to the unified tensor_create_<shape>_streamed(..., int dtag) symbol.

Each wrapper has the form
    ... (let ((raw_r <DISPATCH>)) (let ((wr (vector ...))) ...))
where <DISPATCH> is either
    (if (= dtag 0) (<f32-call>) (<f64-call>))                 [tape/mlx]
    (cond ((= dtag 1) (<f64-call>)) ... (else (<f32-call>)))  [torch]
and every dispatch contains exactly one *_f32_streamed_<back> call. We
derive the unified symbol (drop the `_f32` infix), append `int` to the
foreign-procedure signature and `dtag` to the args, and replace the whole
<DISPATCH> with that single call. The guardian/retain/wrap boilerplate is
left untouched.
"""
import re
import sys

FILES = [
    "packages/idris-ml/src/Device/Tape.idr",
    "packages/idris-ml/src/Device/Torch.idr",
    "packages/idris-ml/src/Device/Mlx.idr",
]

# matches `(foreign-procedure \"<base>_f32_streamed_<back>\"` inside the file
F32_FP = re.compile(r'\(foreign-procedure \\"([a-z0-9_]+)_f32_streamed_(tape|torch|mlx)\\"')
# parse a full f32 call: ((foreign-procedure \"NAME\" (SIG) void*) ARGS)
CALL = re.compile(
    r'^\(\(foreign-procedure \\"(?P<name>[^"\\]+)\\" \((?P<sig>[^)]*)\) void\*\) (?P<args>.*)\)$'
)


def match_balanced(s, i):
    """s[i] == '(' ; return index just past the matching ')'."""
    assert s[i] == '(', f"expected ( at {i}, got {s[i]!r}"
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise ValueError("unbalanced parens")


def transform_line(line):
    """Return (new_line, n_changed)."""
    changed = 0
    out = line
    while True:
        anchor = out.find('(raw_r ')
        # find an anchor whose dispatch still contains an f32 streamed call
        search_from = 0
        ds = de = None
        while True:
            a = out.find('(raw_r ', search_from)
            if a < 0:
                break
            dstart = a + len('(raw_r ')
            if dstart >= len(out) or out[dstart] != '(':
                search_from = a + 1
                continue
            dend = match_balanced(out, dstart)
            dispatch = out[dstart:dend]
            if '_f32_streamed_' in dispatch:
                ds, de = dstart, dend
                break
            search_from = a + 1
        if ds is None:
            break

        dispatch = out[ds:de]
        m = F32_FP.search(dispatch)
        if not m:
            break
        fp_idx = m.start()
        call_start = fp_idx - 1
        assert dispatch[call_start] == '(', dispatch[:80]
        call_end = match_balanced(dispatch, call_start)
        call = dispatch[call_start:call_end]
        cm = CALL.match(call)
        if not cm:
            raise SystemExit(f"could not parse f32 call:\n{call}")
        name = cm.group('name')             # e.g. tensor_create_1d_f32_streamed_tape
        sig = cm.group('sig')               # e.g. int void* int int
        args = cm.group('args')             # e.g. n data rg stream
        unified_name = name.replace('_f32_streamed_', '_streamed_', 1)
        unified_sig = (sig + ' int').strip()
        unified_args = (args + ' dtag').strip()
        unified_call = (
            f'((foreign-procedure \\"{unified_name}\\" ({unified_sig}) void*) {unified_args})'
        )
        out = out[:ds] + unified_call + out[de:]
        changed += 1
    return out, changed


def main():
    total = 0
    for path in FILES:
        with open(path) as f:
            lines = f.readlines()
        n_file = 0
        for i, line in enumerate(lines):
            if '%foreign "scheme:' not in line or '(raw_r ' not in line:
                continue
            new, n = transform_line(line)
            if n:
                lines[i] = new
                n_file += n
        with open(path, 'w') as f:
            f.writelines(lines)
        print(f"{path}: {n_file} dispatch(es) unified")
        total += n_file
    print(f"total: {total}")


if __name__ == '__main__':
    main()
