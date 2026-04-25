"""Jupyter kernel for idris-ml wrapping the Idris 2 REPL."""

import os
import re
from pathlib import Path

import pexpect
from ipykernel.kernelbase import Kernel

from .cell_parser import parse_cell
from .repl import Idris2REPL


def _find_project_root() -> Path:
    """Find the idris-ml project root from IDRIS_ML_ROOT env or by walking up."""
    env_root = os.environ.get("IDRIS_ML_ROOT")
    if env_root:
        return Path(env_root)
    # Walk up from this file looking for packages/idris-ml/idris-ml.ipkg
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "packages" / "idris-ml" / "idris-ml.ipkg").exists():
            return p
        p = p.parent
    raise FileNotFoundError(
        "Cannot find idris-ml project root. Set IDRIS_ML_ROOT environment variable."
    )


def _extract_word(code: str, cursor_pos: int) -> str:
    """Extract the word under/before the cursor for inspection."""
    left = code[:cursor_pos]
    match = re.search(r"[A-Za-z_][A-Za-z0-9_.']*$", left)
    return match.group(0) if match else ""


class Idris2Kernel(Kernel):
    implementation = "idris-ml"
    implementation_version = "0.1.0"
    language = "idris2"
    language_version = "0.8.0"
    language_info = {
        "name": "idris2",
        "mimetype": "text/x-idris2",
        "file_extension": ".idr",
        "codemirror_mode": "haskell",
    }
    banner = "Idris 2 (idris-ml) \u2014 Deep learning with dependent types"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = _find_project_root()
        self.repl = Idris2REPL(project_root=root)

    def do_execute(
        self, code, silent, store_history=True, user_expressions=None, allow_stdin=False
    ):
        commands = parse_cell(code)
        if not commands:
            return {
                "status": "ok",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

        output_parts = []
        has_error = False

        for cmd in commands:
            # Track session state for restart recovery
            if cmd.startswith(":module "):
                mod = cmd.split(maxsplit=1)[1]
                if mod not in self.repl.modules:
                    self.repl.modules.append(mod)
            elif cmd.startswith(":let "):
                self.repl.lets.append(cmd)

            try:
                result = self.repl.send(cmd)
            except (pexpect.EOF, pexpect.TIMEOUT):
                self.repl.restart()
                result = "REPL restarted (crash recovery). Re-run this cell."
                has_error = True

            if result:
                if "Error:" in result or "Exception:" in result:
                    has_error = True
                output_parts.append(result)

        text = "\n".join(output_parts)

        if not silent and text:
            stream_name = "stderr" if has_error else "stdout"
            self.send_response(
                self.iopub_socket, "stream", {"name": stream_name, "text": text + "\n"}
            )

        if has_error:
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "ename": "IdrisError",
                "evalue": text,
                "traceback": [text],
            }

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    def do_inspect(self, code, cursor_pos, detail_level=0, omit_sections=()):
        word = _extract_word(code, cursor_pos)
        if not word:
            return {"status": "ok", "found": False, "data": {}, "metadata": {}}

        try:
            type_info = self.repl.send(f":t {word}", timeout=5)
            doc_info = self.repl.send(f":doc {word}", timeout=5)
        except (pexpect.EOF, pexpect.TIMEOUT):
            return {"status": "ok", "found": False, "data": {}, "metadata": {}}

        combined = type_info
        if doc_info and doc_info != type_info:
            combined = f"{type_info}\n\n{doc_info}"

        return {
            "status": "ok",
            "found": bool(combined),
            "data": {"text/plain": combined},
            "metadata": {},
        }

    def do_is_complete(self, code):
        stripped = code.rstrip()
        if stripped.endswith(("=", "where", "do", "of", "\\", ",")):
            return {"status": "incomplete", "indent": "  "}
        return {"status": "complete"}

    def do_shutdown(self, restart):
        self.repl.close()
        return {"status": "ok", "restart": restart}
