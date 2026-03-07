"""Side-by-side comparison of Idris and PyTorch benchmarks.

Runs both Idris bench (via build/exec/bench) and PyTorch benchmark,
parses output, and prints a comparison table with ratios.
"""

import re
import subprocess
import sys
from pathlib import Path

from torch_ref.benchmark import bench_ntm, bench_ntm_copy, bench_rnn, bench_supervised

# Repo root is pytorch/../
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_idris_output(output: str) -> dict[str, tuple[float, float, float]]:
    """Parse Idris bench output into {model: (ms, loss, rss_mb)} dict."""
    results: dict[str, tuple[float, float, float]] = {}
    lines = output.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"(.+?)\s+\(\d+ epochs?\):\s+([\d.]+)\s+ms", line)
        if m:
            name = m.group(1).strip()
            ms = float(m.group(2))
            loss = 0.0
            rss = 0.0
            # Scan following lines for loss and RSS
            for j in range(i + 1, min(i + 3, len(lines))):
                lm = re.search(r"Final loss:\s+([\d.e+-]+)", lines[j])
                if lm:
                    loss = float(lm.group(1))
                rm = re.search(r"Peak RSS:\s+(\d+)\s+MB", lines[j])
                if rm:
                    rss = float(rm.group(1))
            results[name] = (ms, loss, rss)
        i += 1
    return results


def run_idris_bench() -> dict[str, tuple[float, float, float]] | None:
    """Run Idris bench and parse output."""
    bench_bin = _REPO_ROOT / "build" / "exec" / "bench"
    try:
        result = subprocess.run(
            [str(bench_bin)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=_REPO_ROOT,
        )
        if result.returncode != 0:
            print(f"Idris bench failed: {result.stderr}", file=sys.stderr)
            return None
        return parse_idris_output(result.stdout)
    except FileNotFoundError:
        print("Idris bench not found. Run 'make bench' first.", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("Idris bench timed out.", file=sys.stderr)
        return None


def main() -> None:
    print("Running PyTorch benchmarks...")
    py_supervised = bench_supervised()
    py_rnn = bench_rnn()
    py_ntm = bench_ntm()
    py_ntm_copy = bench_ntm_copy()

    py_results = {
        "Supervised": py_supervised,
        "RNN": py_rnn,
        "NTM": py_ntm,
        "NTM-copy": py_ntm_copy,
    }

    print("\nRunning Idris benchmarks...")
    idris_results = run_idris_bench()

    print("\n" + "=" * 80)
    print("Benchmark Comparison: Idris vs PyTorch")
    print("=" * 80)

    cols = [
        "Model",
        "Idris (ms)",
        "PyTorch (ms)",
        "Ratio",
        "Idris Loss",
        "PyTorch Loss",
        "Idris RSS",
        "PyTorch RSS",
    ]
    header = (
        f"{cols[0]:<15} {cols[1]:>12} {cols[2]:>14} {cols[3]:>8}"
        f" {cols[4]:>12} {cols[5]:>14} {cols[6]:>10} {cols[7]:>12}"
    )
    print(header)
    print("-" * len(header))

    for name in ["Supervised", "RNN", "NTM", "NTM-copy"]:
        py_ms, py_loss, py_rss = py_results[name]

        if idris_results and name in idris_results:
            idris_ms, idris_loss, idris_rss = idris_results[name]
            ratio = idris_ms / py_ms if py_ms > 0 else 0
            print(
                f"{name:<15} {idris_ms:>12.1f} {py_ms:>14.1f}"
                f" {ratio:>7.2f}x {idris_loss:>12.6f} {py_loss:>14.6f}"
                f" {idris_rss:>9.0f}MB {py_rss:>11.0f}MB"
            )
        else:
            na = "N/A"
            print(
                f"{name:<15} {na:>12} {py_ms:>14.1f} {na:>8}"
                f" {na:>12} {py_loss:>14.6f} {na:>10} {py_rss:>11.0f}MB"
            )


if __name__ == "__main__":
    main()
