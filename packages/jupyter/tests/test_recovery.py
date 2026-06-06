"""Test REPL restart and session state recovery."""

from pathlib import Path

from idris_ml_kernel.repl import Idris2REPL

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_restart_replays_state():
    """After restart, tracked :module and :let commands are replayed."""
    repl = Idris2REPL(project_root=PROJECT_ROOT)
    try:
        # Set up state
        repl.send(":module Layer.Core")
        repl.modules.append("Layer.Core")

        repl.send(":let recoveryTest : Int")
        repl.send(":let recoveryTest = 99")
        repl.lets.extend([":let recoveryTest : Int", ":let recoveryTest = 99"])

        # Verify state exists before restart
        result = repl.send(":t recoveryTest")
        assert "Int" in result

        # Force restart
        repl.restart()

        # Verify state was replayed
        result = repl.send(":t recoveryTest")
        assert "Int" in result
    finally:
        repl.close()
