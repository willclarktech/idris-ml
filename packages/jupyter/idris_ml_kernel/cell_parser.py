"""Classify notebook cell content and transform for the Idris 2 REPL."""

# REPL commands that take an expression argument and may span multiple lines.
_EXPR_COMMANDS = (":exec ", ":t ", ":type ", ":doc ", ":search ")


def looks_like_definition(line: str) -> bool:
    """Heuristic: type signature (name : Type) or function clause (name args = body)."""
    # Type signature: "name : Type -> Type" but not "(expr : thing)"
    if " : " in line and not line.startswith("("):
        return True
    # Function clause: "name args = body" but not "x == y"
    return " = " in line and line[0].isalpha() and "==" not in line


def _unclosed_brackets(text: str) -> bool:
    """True if text has more opening brackets than closing ones."""
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    return depth > 0


def parse_cell(code: str) -> list[str]:
    """Parse a notebook cell into a list of REPL commands.

    Routes each line:
      - REPL commands starting with ':' are passed through
      - Multi-line commands are joined when brackets are unclosed or
        continuation lines don't start with ':'
      - Lines that look like definitions get ':let ' prefixed
      - Everything else is sent as a bare expression
    """
    lines = [line for line in code.strip().split("\n") if line.strip()]
    commands: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()

        if any(s.startswith(prefix) for prefix in _EXPR_COMMANDS):
            # :exec/:t/:doc — join continuation lines
            parts = [s]
            while i + 1 < len(lines) and not lines[i + 1].strip().startswith(":"):
                i += 1
                parts.append(lines[i].strip())
            commands.append(" ".join(parts))
        elif s.startswith(":"):
            commands.append(s)
        elif looks_like_definition(s):
            # Join continuation lines while brackets are unclosed
            parts = [s]
            while i + 1 < len(lines) and _unclosed_brackets(" ".join(parts)):
                i += 1
                parts.append(lines[i].strip())
            commands.append(f":let {' '.join(parts)}")
        else:
            # Bare expression — also join on unclosed brackets
            parts = [s]
            while i + 1 < len(lines) and _unclosed_brackets(" ".join(parts)):
                i += 1
                parts.append(lines[i].strip())
            commands.append(" ".join(parts))
        i += 1
    return commands
