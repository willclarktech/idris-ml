"""Parsers for the Idris reachability gap-finder.

`reach-gap-probe.py` computes which top-level Idris definitions are never
reachable from any test/example entry point. It needs two halves:

  - `parse_reachable(cases_text)`
        The set of definition FQNs that appear in an `idris2 --dumpcases`
        dump. The dump is TREE-SHAKEN — only definitions reachable from
        the compiled `main` get a line — so this IS the reachable set for
        that entry point. Format per line: `Fully.Qualified.Name = [args]:
        (body)`. Names are normalized (see `normalize_fqn`).

  - `scan_universe(src_root)`
        Every top-level definition declared in the `.idr` source under
        `src_root`, as normalized `Module.name` FQNs mapped to the source
        file. Anchored on column-0 type signatures + the in-file `module`
        declaration; nested scopes (data/record/interface bodies, `where`
        locals, `namespace` blocks) are indented and therefore skipped,
        so constructors / fields / methods — which never get their own
        dump line — are excluded for free.

`normalize_fqn` strips parentheses from operator name segments so the
two halves agree: the dump renders operators inconsistently (`Array.+`
bare but `Array.(++)` / `Floating.(^)` parenthesized), while source always
writes `(op)`. Stripping parens on both sides canonicalizes them.

Accuracy limits (see docs/develop/reachability-policy.md): defs without a
type signature, nested-namespace defs, and interface methods are not in
the universe; inlined/erased defs may be absent from a dump. These are
handled by the EXCL sidecar + documented as known gaps.
"""

from __future__ import annotations

import re
from pathlib import Path

# A definition line in a --dumpcases dump: `<fqn> = ...`. The FQN is every
# non-space, non-`=` character before the ` = `.
_DUMP_LINE_RE = re.compile(r"^([^\s=]+) = ")

# Column-0 top-level type signature. Optional leading modifiers (on the
# same line — they are usually on their own line above, but tolerate
# inline), then an identifier or a parenthesized operator, then a single
# `:` that is not `::`/`:=`.
_MODIFIER = r"(?:(?:public\s+)?export\s+|private\s+|%[A-Za-z]\w*\s+|total\s+|covering\s+|partial\s+)*"
_SIG_RE = re.compile(rf"^{_MODIFIER}([a-z_][A-Za-z0-9'_]*|\([^)]+\))\s*:(?![:=])")

# MULTILINE: the `module` decl is rarely the literal first byte — most
# files open with a license/doc comment block, so anchor to any line start.
_MODULE_RE = re.compile(r"^module\s+(\S+)", re.MULTILINE)


def normalize_fqn(fqn: str) -> str:
    """Canonicalize an FQN by stripping parens from operator segments.

    `Array.(++)` -> `Array.++`, `(>>=)` -> `>>=`, `Array.+` -> `Array.+`.
    Operators are the only parenthesized part of a name, so a blanket
    paren strip is safe and makes the dump's inconsistent operator
    rendering agree with the source's always-parenthesized form.
    """
    return fqn.replace("(", "").replace(")", "")


def parse_reachable(cases_text: str) -> set[str]:
    """Normalized FQNs of every definition in a --dumpcases dump."""
    out: set[str] = set()
    for line in cases_text.splitlines():
        m = _DUMP_LINE_RE.match(line)
        if m:
            out.add(normalize_fqn(m.group(1)))
    return out


def _strip_block_comments(text: str) -> str:
    """Remove `{- ... -}` (nesting-aware) so commented code can't look
    like a top-level signature. Line comments (`--`) are handled per-line
    in the scanner."""
    out: list[str] = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        two = text[i : i + 2]
        if two == "{-":
            depth += 1
            i += 2
        elif two == "-}" and depth > 0:
            depth -= 1
            i += 2
        else:
            # Preserve newlines even inside comments so line structure
            # (and column-0 anchoring) survives for the post-strip scan.
            ch = text[i]
            if depth == 0 or ch == "\n":
                out.append(ch)
            i += 1
    return "".join(out)


def scan_universe_text(module_text: str) -> set[str]:
    """Top-level def names (paren-normalized, no module prefix) declared
    in one module's source text. Column-0 signatures only."""
    names: set[str] = set()
    for raw in _strip_block_comments(module_text).splitlines():
        # Column-0 only: a leading space/tab means a nested scope
        # (data/record/interface body, `where`, `namespace`, multi-line
        # sig continuation) — skip it.
        if not raw or raw[0].isspace():
            continue
        if raw.lstrip().startswith("--"):
            continue
        m = _SIG_RE.match(raw)
        if m:
            names.add(normalize_fqn(m.group(1)))
    return names


def scan_universe(src_root: Path, *, exclude_dirs: tuple[str, ...] = ("Test",)) -> dict[str, str]:
    """Map normalized `Module.name` FQN -> source file (relative to
    `src_root`) for every top-level def under `src_root`, skipping any
    `.idr` whose path crosses one of `exclude_dirs` and `.idr.in`
    templates."""
    excluded = set(exclude_dirs)
    universe: dict[str, str] = {}
    for path in sorted(src_root.rglob("*.idr")):
        rel = path.relative_to(src_root)
        if excluded.intersection(rel.parts):
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        mod_match = _MODULE_RE.search(text)
        if not mod_match:
            continue
        module = mod_match.group(1)
        for name in scan_universe_text(text):
            universe[f"{module}.{name}"] = str(rel)
    return universe
