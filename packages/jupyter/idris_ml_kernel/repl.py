"""Persistent Idris 2 REPL wrapper via pexpect."""

from __future__ import annotations

import contextlib
import platform
import shutil
import subprocess
import termios
from pathlib import Path

import pexpect


def _idris2_bin() -> str:
    """Resolve the idris2 executable.

    Prefer pack's RAW binary (`pack app-path idris2`) over a bare `idris2`
    on PATH. In this repo's pack setup the PATH `idris2` is a wrapper that
    `export`s IDRIS2_PACKAGE_PATH="$(pack package-path)" before exec'ing the
    real binary — clobbering the value we (and make) set to include the
    local install prefix where idris-ml lives. The result is the kernel
    dying with "Can't find package idris-ml (any)" even though idris-ml is
    installed (CI run 28329359748). The raw binary honours the inherited
    IDRIS2_PACKAGE_PATH. Fall back to a PATH `idris2`, then the bare name.
    """
    with contextlib.suppress(Exception):
        out = subprocess.run(
            ["pack", "app-path", "idris2"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if out:
            return out
    found = shutil.which("idris2")
    if found:
        return found
    return "idris2"


class Idris2REPL:
    """Manage a persistent idris2 REPL subprocess with FFI dylib support."""

    # The REPL prompt looks like "Notebook.Prelude> " at line start.
    # Must not match layer names in output like "relu> " mid-line.
    # The module name always contains a dot (Notebook.Prelude, Ml.Tensor, etc.)
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

    def _build_dir(self) -> Path:
        """Resolve the active per-set build tree (`build/<BUILD_KEY>/`).

        Builds live under per-set trees since the BUILD_KEY refactor
        (c5c78ee9, 2026-05-17) — there is no top-level `build/exec` or
        `.idris2/` prefix anymore. Resolution order:
          1. `IDRIS_ML_BUILD_DIR` env var — set by the Make recipes
             (test-e2e-jupyter etc.) so tests pin the exact set Make
             just built.
          2. Newest `build/*/libidrisml.*` tree — the standalone-kernel
             path (Jupyter launched from the kernelspec, no Make env).
        """
        import os

        env_dir = os.environ.get("IDRIS_ML_BUILD_DIR")
        if env_dir:
            return Path(env_dir)
        candidates = sorted(
            (p.parent for p in self.root.glob(f"build/*/{self._lib_name()}")),
            key=lambda p: (p / self._lib_name()).stat().st_mtime,
        )
        if not candidates:
            raise RuntimeError(
                f"no build/<set>/{self._lib_name()} found under {self.root} — "
                "run `make install` first (or set IDRIS_ML_BUILD_DIR)"
            )
        return candidates[-1]

    def _ensure_dylib(self) -> None:
        """Copy the backend dylib where :exec's temp Chez directory expects it.

        The destination is NOT per-set: idris2's `:exec` writes its temp
        Chez app to `<cwd>/build/exec/_tmpchez_app/` unconditionally (the
        REPL runs with cwd = repo root), so the dylib must land there.
        Only the *source* lives in the per-set tree.
        """
        tmpchez = self.root / "build" / "exec" / "_tmpchez_app"
        tmpchez.mkdir(parents=True, exist_ok=True)
        dylib = self._build_dir() / self._lib_name()
        dest = tmpchez / self._lib_name()
        if dylib.exists():
            shutil.copy2(dylib, dest)

    def _spawn(self) -> None:
        """Start the idris2 REPL with Notebook.Prelude loaded via installed packages."""
        import os

        env = os.environ.copy()
        # Make's config.mk exports the correct per-set IDRIS2_PACKAGE_PATH
        # to recipe subprocesses — respect it when present; otherwise
        # derive it from the resolved build tree (standalone kernel).
        if "IDRIS2_PACKAGE_PATH" not in env:
            pkg_path = self._build_dir() / "idris2-prefix" / "idris2-0.8.0"
            paths = [str(pkg_path)]
            # Append pack's collection so base/contrib/linear resolve: we use
            # the raw idris2 binary (see _idris2_bin), bypassing the PATH
            # wrapper that would otherwise set this. Make-driven runs already
            # export the combined local:pack path, so this branch only fires
            # for a standalone kernel launched straight from the kernelspec.
            with contextlib.suppress(Exception):
                pp = subprocess.run(
                    ["pack", "package-path"], capture_output=True, text=True, check=True
                ).stdout.strip()
                if pp:
                    paths.append(pp)
            env["IDRIS2_PACKAGE_PATH"] = ":".join(paths)

        # Declared spawn[str]: the stubs can't infer the str specialization
        # from encoding="utf-8" (the constructor isn't overloaded on it).
        self.child: pexpect.spawn[str] = pexpect.spawn(
            _idris2_bin(),
            [
                "-p",
                "contrib",
                # `linear` carries Control.Linear.LIO — the Nn/Fit surface
                # (and thus Notebook.Prelude) imports it transitively, so the
                # prelude fails to load without it.
                "-p",
                "linear",
                "-p",
                "idris-ml",
                "-p",
                "idris-ml-notebook",
                # HF model loading (models/bert.ipynb does an explicit
                # `:module Transformers.Bert`); installed by install-notebook's
                # install-transformers dependency. elab-util is its transitive
                # dependency (Transformers.* fails to load without it).
                "-p",
                "elab-util",
                "-p",
                "idris-transformers",
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
