"""Persistent Idris 2 REPL wrapper via pexpect."""

from __future__ import annotations

import contextlib
import platform
import shutil
import termios
from typing import TYPE_CHECKING

import pexpect

if TYPE_CHECKING:
    from pathlib import Path


class Idris2REPL:
    """Manage a persistent idris2 REPL subprocess with FFI dylib support."""

    # The REPL prompt looks like "Notebook.Prelude> " at line start.
    # Must not match layer names in output like "relu> " mid-line.
    # The module name always contains a dot (Notebook.Prelude, Layer.Core, etc.)
    # or is "Main" (bare REPL before :module).
    PROMPT_RE = r"(\[scheme\] )?(Main|[A-Za-z][A-Za-z0-9]*\.[A-Za-z0-9.]*)> "

    # .dylib on macOS, .so on Linux
    _LIB_EXT = ".dylib" if platform.system() == "Darwin" else ".so"

    def __init__(self, project_root: Path, timeout: int = 60):
        self.root = project_root
        self.timeout = timeout
        self.modules: list[str] = []
        self.lets: list[str] = []
        self._ensure_dylib()
        self._spawn()

    def _lib_name(self) -> str:
        return f"libidrisml{self._LIB_EXT}"

    def _ensure_dylib(self) -> None:
        """Copy the backend dylib where :exec's temp Chez directory expects it."""
        tmpchez = self.root / "build" / "exec" / "_tmpchez_app"
        tmpchez.mkdir(parents=True, exist_ok=True)
        dylib = self.root / "build" / self._lib_name()
        dest = tmpchez / self._lib_name()
        if dylib.exists():
            shutil.copy2(dylib, dest)

    def _spawn(self) -> None:
        """Start the idris2 REPL with Notebook.Prelude loaded via installed packages."""
        # Set IDRIS2_PACKAGE_PATH so idris2 finds locally-installed packages
        pkg_path = str(self.root / ".idris2" / "idris2-0.8.0")
        import os

        env = os.environ.copy()
        env["IDRIS2_PACKAGE_PATH"] = pkg_path

        self.child = pexpect.spawn(
            "idris2",
            [
                "-p",
                "contrib",
                "-p",
                "idris-ml",
                "-p",
                "idris-ml-notebook",
                "--no-banner",
                "--no-colour",
            ],
            cwd=str(self.root),
            timeout=self.timeout,
            encoding="utf-8",
            echo=False,
            env=env,
            dimensions=(24, 10000),  # wide terminal to prevent line wrapping
        )
        self.child.expect(self.PROMPT_RE, timeout=self.timeout)
        # Disable canonical mode to remove PTY line-length limit (1024 bytes on macOS).
        attrs = termios.tcgetattr(self.child.child_fd)
        attrs[3] &= ~termios.ICANON
        termios.tcsetattr(self.child.child_fd, termios.TCSANOW, attrs)
        # Load the notebook prelude module
        self.send(":module Notebook.Prelude")

    def send(self, cmd: str, timeout: int | None = None) -> str:
        """Send a command and return the output (text between send and next prompt)."""
        t = timeout if timeout is not None else self.timeout
        self.child.sendline(cmd)
        self.child.expect(self.PROMPT_RE, timeout=t)
        output = self.child.before or ""
        # Strip echoed command from front (pexpect may echo even with echo=False)
        lines = output.split("\n")
        if lines and lines[0].strip() == cmd.strip():
            lines = lines[1:]
        return "\n".join(lines).strip()

    def is_alive(self) -> bool:
        return self.child.isalive()

    def restart(self) -> None:
        """Kill and respawn, replaying session state."""
        with contextlib.suppress(Exception):
            self.child.close(force=True)
        self._ensure_dylib()
        self._spawn()
        for mod in self.modules:
            self.send(f":module {mod}")
        for let_cmd in self.lets:
            self.send(let_cmd)

    def close(self) -> None:
        """Shut down the REPL."""
        with contextlib.suppress(Exception):
            self.child.sendline(":q")
            self.child.expect(pexpect.EOF, timeout=5)
        with contextlib.suppress(Exception):
            self.child.close(force=True)
