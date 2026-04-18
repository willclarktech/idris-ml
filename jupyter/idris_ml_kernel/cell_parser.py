"""Classify notebook cell content and transform for the Idris 2 REPL."""

# REPL commands that take an expression argument and may span multiple lines.
_EXPR_COMMANDS = (":exec ", ":t ", ":type ", ":doc ", ":search ")


def looks_like_definition(line: str) -> bool:
    """Heuristic: type signature (name : Type) or function clause (name args = body)."""
    # Type signature: "name : Type -> Type" but not "(expr : thing)"
    if " : " in line and not line.startswith("("):
        return True
    # Function clause: "name args = body" but not "x == y"
    if " = " in line and line[0].isalpha() and "==" not in line:
        return True
    return False


def parse_cell(code: str) -> list[str]:
    """Parse a notebook cell into a list of REPL commands.

    Routes each line:
      - REPL commands starting with ':' are passed through
      - Multi-line :exec/:t/:doc commands are joined (continuation lines
        that don't start with ':' are appended to the previous command)
      - Lines that look like definitions get ':let ' prefixed
      - Everything else is sent as a bare expression
    """
    lines = [line for line in code.strip().split("\n") if line.strip()]
    commands: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if any(s.startswith(prefix) for prefix in _EXPR_COMMANDS):
            # Join continuation lines that don't start with ':'
            parts = [s]
            while i + 1 < len(lines) and not lines[i + 1].strip().startswith(":"):
                i += 1
                parts.append(lines[i].strip())
            commands.append(" ".join(parts))
        elif s.startswith(":"):
            commands.append(s)
        elif looks_like_definition(s):
            commands.append(f":let {s}")
        else:
            commands.append(s)
        i += 1
    return commands
