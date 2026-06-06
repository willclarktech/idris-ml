#!/usr/bin/env python3
"""Verify Idris example defaults match the paired torch_ref/scripts/<x>.py defaults.

For each Idris/Python pair, extract per-flag default values from both sides and
diff them. Drift between paired example/ref defaults silently desyncs the
benchmark — this check catches it.

Idris parse: regex over `record Config`, `defaultConfig = MkConfig <vals>`, and
the `Arg "--flag" (\\v, c => { field := ... } c)` lines.
Python parse: ast.parse, walk for `parser.add_argument(...)` calls.

Exit 0 = clean, 1 = drift, 2 = parse failure.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mapping table: short name -> (idris file, python file, optional overrides).
# `idris_only` / `python_only` declare flags that legitimately exist on only
# one side. Anything not declared falls through as drift.
#
# Common pattern for python-only `--lr-find`: that example doesn't have the
# lr_find machinery wired on the Idris side. Mostly the supervised/RNN-family
# ones. Adding it is a separate small task per example, tracked elsewhere.
EXAMPLES: list[dict] = [
    {
        "name": "supervised",
        "idris": "packages/idris-ml-examples/src/Example/Supervised.idr",
        "python": "packages/pytorch/torch_ref/scripts/supervised.py",
        "python_only": ["--lr-find"],
    },
    {
        "name": "rnn",
        "idris": "packages/idris-ml-examples/src/Example/Rnn.idr",
        "python": "packages/pytorch/torch_ref/scripts/rnn.py",
        "idris_only": ["--patience"],  # idris-side windowed-avg ES; py runs fixed-epoch
        "python_only": ["--lr-find"],
    },
    {
        "name": "lstm",
        "idris": "packages/idris-ml-examples/src/Example/Lstm.idr",
        "python": "packages/pytorch/torch_ref/scripts/lstm.py",
        "python_only": ["--lr-find"],
    },
    {
        "name": "gru",
        "idris": "packages/idris-ml-examples/src/Example/Gru.idr",
        "python": "packages/pytorch/torch_ref/scripts/gru.py",
        "python_only": ["--lr-find"],
    },
    {
        "name": "mnist",
        "idris": "packages/idris-ml-examples/src/Example/Mnist.idr",
        "python": "packages/pytorch/torch_ref/scripts/mnist.py",
        "idris_only": ["--data"],  # idris loads from local path; py uses torchvision
        "python_only": ["--batch-size"],
    },  # py exposes batch knob; idris bakes it
    {
        "name": "seq-classify",
        "idris": "packages/idris-ml-examples/src/Example/SeqClassify.idr",
        "python": "packages/pytorch/torch_ref/scripts/seq_classify.py",
        "idris_only": ["--patience"],
    },
    {
        "name": "transformer",
        "idris": "packages/idris-ml-examples/src/Example/Transformer.idr",
        "python": "packages/pytorch/torch_ref/scripts/transformer.py",
        "python_only": ["--blocks"],
    },  # py parameterises blocks; idris bakes it
    {
        "name": "gpt",
        "idris": "packages/idris-ml-examples/src/Example/Gpt.idr",
        "python": "packages/pytorch/torch_ref/scripts/gpt.py",
    },
    # NTM/DNC family: alpha/eps/momentum are RMSprop tuning. Idris exposes
    # them as CLI flags; Python bakes them into `torch.optim.RMSprop(...)`.
    # Same values used on both sides (verified at call site).
    {
        "name": "ntm-copy",
        "idris": "packages/idris-ml-examples/src/Example/NtmCopy.idr",
        "python": "packages/pytorch/torch_ref/scripts/ntm_copy.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "ntm-recall",
        "idris": "packages/idris-ml-examples/src/Example/NtmAssociativeRecall.idr",
        "python": "packages/pytorch/torch_ref/scripts/ntm_recall.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "dnc-copy",
        "idris": "packages/idris-ml-examples/src/Example/DncCopy.idr",
        "python": "packages/pytorch/torch_ref/scripts/dnc_copy.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "dnc-recall",
        "idris": "packages/idris-ml-examples/src/Example/DncAssociativeRecall.idr",
        "python": "packages/pytorch/torch_ref/scripts/dnc_recall.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "reinforce",
        "idris": "packages/idris-ml-examples/src/Example/Reinforce.idr",
        "python": "packages/pytorch/torch_ref/scripts/reinforce.py",
        "idris_only": ["--batched"],
    },  # Job 4 Phase B; py doesn't have it
    {
        "name": "a2c",
        "idris": "packages/idris-ml-examples/src/Example/A2c.idr",
        "python": "packages/pytorch/torch_ref/scripts/a2c.py",
        "python_only": ["--rollout"],
    },  # rollout len exposed py-side; baked idris-side
    {
        "name": "ppo",
        "idris": "packages/idris-ml-examples/src/Example/Ppo.idr",
        "python": "packages/pytorch/torch_ref/scripts/ppo.py",
        "idris_only": ["--value-coef"],
        "python_only": ["--batch-size", "--max-ep-len", "--rollout"],
    },
    {
        "name": "dqn",
        "idris": "packages/idris-ml-examples/src/Example/Dqn.idr",
        "python": "packages/pytorch/torch_ref/scripts/dqn.py",
        "idris_only": ["--eps-start", "--eps-end", "--eps-decay"],
    },
    {
        "name": "mountain-car",
        "idris": "packages/idris-ml-examples/src/Example/MountainCar.idr",
        "python": "packages/pytorch/torch_ref/scripts/mountain_car.py",
    },
    {
        "name": "mountain-car-cont",
        "idris": "packages/idris-ml-examples/src/Example/MountainCarCont.idr",
        "python": "packages/pytorch/torch_ref/scripts/mountain_car_cont.py",
        "idris_only": ["--clip", "--es-threshold", "--es-window", "--es-patience"],
    },
]


@dataclass
class FlagInfo:
    flag: str
    default: object
    source_line: int


@dataclass
class ExampleReport:
    name: str
    idris_flags: dict[str, FlagInfo]
    python_flags: dict[str, FlagInfo]
    value_mismatches: list[tuple[str, object, object]]  # (flag, idris_val, python_val)
    idris_only: list[str]
    python_only: list[str]


def _parse_idris_literal(tok: str) -> object:
    """Convert an Idris literal token to a Python value.

    Numbers: 0.03, 1.0e-8, 3.0e-4, 1000 → float / int
    Negative numbers wrapped in parens: (-85.0) → float
    Booleans: True / False → bool
    Strings: "data/mnist" → str
    """
    tok = tok.strip()
    # Idris uses (-N) for negative literals to avoid ambiguity with subtraction.
    if tok.startswith("(") and tok.endswith(")"):
        tok = tok[1:-1].strip()
    if tok == "True":
        return True
    if tok == "False":
        return False
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1]
    # Numeric: try int first, then float.
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    raise ValueError(f"unrecognised Idris literal: {tok!r}")


def _tokenise_mkconfig_values(rest: str) -> list[str]:
    """Split the args of `MkConfig <v> <v> ...` respecting quoted strings.

    Idris-side values are always whitespace-separated atomic tokens here —
    no compound expressions are allowed in the MkConfig literal by convention.
    """
    tokens: list[str] = []
    i = 0
    n = len(rest)
    while i < n:
        while i < n and rest[i].isspace():
            i += 1
        if i >= n:
            break
        if rest[i] == '"':
            j = i + 1
            while j < n and rest[j] != '"':
                if rest[j] == "\\":
                    j += 2
                else:
                    j += 1
            tokens.append(rest[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < n and not rest[j].isspace():
                j += 1
            tokens.append(rest[i:j])
            i = j
    return tokens


def parse_idris(path: Path) -> dict[str, FlagInfo]:
    """Extract {flag: FlagInfo} from an Idris example file.

    Process:
      1. Parse `record Config where ... constructor MkConfig\\n  <f1> : <t1>\\n  <f2> : <t2>...`
         to get the ordered field list.
      2. Parse `defaultConfig = MkConfig <v1> <v2> ...` to get positional values.
      3. Parse each `Arg "--flag" (\\v, c => { <field> := ... } c)` to get
         (flag, field) pairs.
      4. Compose: {flag: default-value-of-mapped-field}.
    """
    text = path.read_text()

    # 1. Record fields. Walk the record body line-by-line starting from the
    # `constructor MkConfig` line. Accumulate field names from indented
    # `<ident> : <type>` lines; skip indented `|||` docstring lines and blank
    # lines (which Idris allows between fields). Stop at the first non-indented
    # line so we don't slurp the next top-level binding.
    record_header = re.search(
        r"record\s+Config\s+where\s*\n[ \t]+constructor\s+MkConfig\s*\n",
        text,
    )
    if not record_header:
        raise ValueError(f"{path}: could not find `record Config where ... constructor MkConfig`")
    fields: list[str] = []
    body_start = record_header.end()
    for line in text[body_start:].splitlines():
        if not line.strip():
            continue  # blank — keep scanning
        if not line[0].isspace():
            break  # next top-level binding — end of record body
        stripped = line.lstrip()
        if stripped.startswith("|||") or stripped.startswith("--"):
            continue  # docstring / line comment between fields
        m = re.match(r"([a-zA-Z][a-zA-Z0-9_]*)\s*:\s*", stripped)
        if m:
            fields.append(m.group(1))

    # 2. defaultConfig values. The MkConfig literal can span multiple physical
    # lines (long arg lists are typically broken with continuation indent).
    # Grab everything from `MkConfig` up to the next blank line or top-level
    # binding (a non-indented line starting with a letter).
    dc_match = re.search(
        r"^defaultConfig\s*=\s*MkConfig\s+([\s\S]*?)(?=\n\n|\n[a-zA-Z])",
        text,
        re.MULTILINE,
    )
    if not dc_match:
        raise ValueError(f"{path}: could not find `defaultConfig = MkConfig ...`")
    raw_values = dc_match.group(1).strip()
    tokens = _tokenise_mkconfig_values(raw_values)
    if len(tokens) != len(fields):
        raise ValueError(
            f"{path}: MkConfig has {len(tokens)} values but record has {len(fields)} fields"
        )
    field_defaults: dict[str, object] = {}
    for fld, tok in zip(fields, tokens, strict=True):
        field_defaults[fld] = _parse_idris_literal(tok)

    # 3. Arg specs: capture (flag, field) pairs.
    flag_to_field: dict[str, str] = {}
    spec_pattern = re.compile(
        r'Arg\s+"(--[a-zA-Z][a-zA-Z0-9_\-]*)"\s+\(\\v,\s*c\s*=>\s*\{\s*([a-zA-Z][a-zA-Z0-9_]*)\s*:='
    )
    for m in spec_pattern.finditer(text):
        flag, field = m.group(1), m.group(2)
        flag_to_field[flag] = field

    # Compose: {flag: default-value}.
    result: dict[str, FlagInfo] = {}
    spec_lines = {}
    for m in spec_pattern.finditer(text):
        spec_lines[m.group(1)] = text[: m.start()].count("\n") + 1
    for flag, field in flag_to_field.items():
        if field not in field_defaults:
            raise ValueError(f"{path}: spec flag {flag} maps to unknown field {field}")
        result[flag] = FlagInfo(
            flag=flag,
            default=field_defaults[field],
            source_line=spec_lines.get(flag, 0),
        )
    return result


def parse_python(path: Path) -> dict[str, FlagInfo]:
    """Extract {flag: FlagInfo} from a torch_ref Python script.

    Walks the AST looking for `parser.add_argument(...)` calls. Handles:
      - positional first arg = flag (e.g. "--lr")
      - default=<literal>
      - action="store_true"  → default False
      - action="store_false" → default True
      - no default + no action → flag is positional/required, skip
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    result: dict[str, FlagInfo] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        flag = first.value
        if not flag.startswith("--"):
            continue  # positional arg; not a default-bearing CLI flag

        default_value: object | None = None
        action: str | None = None
        for kw in node.keywords:
            if kw.arg == "default":
                if isinstance(kw.value, ast.Constant):
                    default_value = kw.value.value
                elif (
                    isinstance(kw.value, ast.UnaryOp)
                    and isinstance(kw.value.op, ast.USub)
                    and isinstance(kw.value.operand, ast.Constant)
                ):
                    default_value = -kw.value.operand.value
            elif kw.arg == "action" and isinstance(kw.value, ast.Constant):
                action = kw.value.value

        if default_value is None and action == "store_true":
            default_value = False
        elif default_value is None and action == "store_false":
            default_value = True

        if default_value is None:
            # No explicit default; argparse uses None. Skip — Idris always has
            # a concrete default, so reporting "None vs 42" is noise.
            continue
        result[flag] = FlagInfo(flag=flag, default=default_value, source_line=node.lineno)
    return result


def _values_match(a: object, b: object) -> bool:
    """Compare two default values across language boundaries.

    Handles int/float widening (3 vs 3.0 = match), float epsilon (1e-12),
    and otherwise == equality.
    """
    if type(a) is bool or type(b) is bool:
        return a == b and type(a) is type(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-12
    return a == b


def compare_example(spec: dict) -> ExampleReport:
    idris_path = REPO_ROOT / spec["idris"]
    python_path = REPO_ROOT / spec["python"]
    idris_flags = parse_idris(idris_path)
    python_flags = parse_python(python_path)

    expected_idris_only = set(spec.get("idris_only", []))
    expected_python_only = set(spec.get("python_only", []))

    value_mismatches: list[tuple[str, object, object]] = []
    for flag, idris_info in idris_flags.items():
        if flag in python_flags and not _values_match(
            idris_info.default, python_flags[flag].default
        ):
            value_mismatches.append((flag, idris_info.default, python_flags[flag].default))

    actual_idris_only = sorted(set(idris_flags) - set(python_flags) - expected_idris_only)
    actual_python_only = sorted(set(python_flags) - set(idris_flags) - expected_python_only)

    return ExampleReport(
        name=spec["name"],
        idris_flags=idris_flags,
        python_flags=python_flags,
        value_mismatches=value_mismatches,
        idris_only=actual_idris_only,
        python_only=actual_python_only,
    )


def format_value(v: object) -> str:
    if isinstance(v, float):
        if v == 0:
            return "0.0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.6g}"
        return f"{v:g}"
    return repr(v) if isinstance(v, str) else str(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on idris-only / python-only flags (not just value mismatches).",
    )
    p.add_argument(
        "--example",
        action="append",
        help="Only check the named example (repeatable). Default: all.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report instead of human-readable text.",
    )
    args = p.parse_args()

    selected = EXAMPLES
    if args.example:
        wanted = set(args.example)
        selected = [e for e in EXAMPLES if e["name"] in wanted]
        missing = wanted - {e["name"] for e in selected}
        if missing:
            print(f"unknown example(s): {sorted(missing)}", file=sys.stderr)
            return 2

    reports: list[ExampleReport] = []
    parse_errors: list[tuple[str, str]] = []
    for spec in selected:
        try:
            reports.append(compare_example(spec))
        except Exception as e:
            parse_errors.append((spec["name"], str(e)))

    if args.json:
        out = {
            "parse_errors": [{"example": n, "error": e} for n, e in parse_errors],
            "examples": [
                {
                    "name": r.name,
                    "value_mismatches": [
                        {"flag": f, "idris": i, "python": py} for f, i, py in r.value_mismatches
                    ],
                    "idris_only": r.idris_only,
                    "python_only": r.python_only,
                }
                for r in reports
            ],
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        any_drift = False
        any_only = False
        for r in reports:
            status_parts = []
            if r.value_mismatches:
                status_parts.append(f"{len(r.value_mismatches)} mismatch")
                any_drift = True
            if r.idris_only:
                status_parts.append(f"{len(r.idris_only)} idris-only")
                any_only = True
            if r.python_only:
                status_parts.append(f"{len(r.python_only)} python-only")
                any_only = True
            status = "OK" if not status_parts else " · ".join(status_parts)
            print(f"{r.name:<20} [{status}]")
            for flag, iv, pv in r.value_mismatches:
                print(f"    DRIFT  {flag}  idris={format_value(iv)}  python={format_value(pv)}")
            for flag in r.idris_only:
                print(
                    f"    idris-only  {flag}  (idris={format_value(r.idris_flags[flag].default)})"
                )
            for flag in r.python_only:
                print(
                    f"    python-only {flag}  (python={format_value(r.python_flags[flag].default)})"
                )

        for name, err in parse_errors:
            print(f"{name:<20} [PARSE ERROR] {err}")

        print()
        ok = (not any_drift) and (not parse_errors) and (not (args.strict and any_only))
        if ok:
            print(f"All {len(reports)} paired examples have matching defaults.")
        else:
            bits = []
            if any_drift:
                bits.append("value mismatches present")
            if parse_errors:
                bits.append(f"{len(parse_errors)} parse errors")
            if args.strict and any_only:
                bits.append("unmatched flags (--strict)")
            print("FAIL: " + ", ".join(bits))

    if parse_errors:
        return 2
    if any(r.value_mismatches for r in reports):
        return 1
    if args.strict and any(r.idris_only or r.python_only for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
