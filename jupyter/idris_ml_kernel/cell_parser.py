"""Classify notebook cell content and transform for the Idris 2 REPL."""


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
      - Lines starting with ':' are REPL commands (passthrough)
      - Lines that look like definitions get ':let ' prefixed
      - Everything else is sent as a bare expression
    """
    lines = [line for line in code.strip().split("\n") if line.strip()]
    commands = []
    for line in lines:
        s = line.strip()
        if s.startswith(":"):
            commands.append(s)
        elif looks_like_definition(s):
            commands.append(f":let {s}")
        else:
            commands.append(s)
    return commands
