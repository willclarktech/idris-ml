"""Side-by-side comparison of idris-ml and PyTorch benchmarks.

Runs both idris-ml bench (via build/exec/bench) and PyTorch benchmark,
parses output, and prints a comparison table with ratios.
"""

import re
import subprocess
import sys

from bench.benchmark import bench_ntm, bench_rnn, bench_supervised


def parse_idris_output(output: str) -> dict[str, tuple[float, float]]:
    """Parse idris-ml bench output into {model: (ms, loss)} dict."""
    results: dict[str, tuple[float, float]] = {}
    lines = output.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"(\w[\w ]+?)\s+\(\d+ epochs?\):\s+([\d.]+)\s+ms", line)
        if m:
            name = m.group(1).strip()
            ms = float(m.group(2))
            # Next line should have final loss
            if i + 1 < len(lines):
                lm = re.search(r"Final loss:\s+([\d.e+-]+)", lines[i + 1])
                loss = float(lm.group(1)) if lm else 0.0
            else:
                loss = 0.0
            results[name] = (ms, loss)
        i += 1
    return results


def run_idris_bench() -> dict[str, tuple[float, float]] | None:
    """Run idris-ml bench and parse output."""
    try:
        result = subprocess.run(
            ["./build/exec/bench"],
            capture_output=True,
            text=True,
            timeout=300,
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

    py_results = {
        "Supervised": py_supervised,
        "RNN": py_rnn,
        "NTM": py_ntm,
    }

    print("\nRunning Idris benchmarks...")
    idris_results = run_idris_bench()

    print("\n" + "=" * 70)
    print("Benchmark Comparison: idris-ml vs PyTorch")
    print("=" * 70)

    header = f"{'Model':<15} {'Idris (ms)':>12} {'PyTorch (ms)':>14} {'Ratio':>8} {'Idris Loss':>12} {'PyTorch Loss':>14}"
    print(header)
    print("-" * len(header))

    idris_map = {
        "Supervised": "Supervised",
        "RNN": "RNN",
        "NTM": "NTM",
    }

    for name in ["Supervised", "RNN", "NTM"]:
        py_ms, py_loss = py_results[name]

        if idris_results:
            idris_key = idris_map[name]
            if idris_key in idris_results:
                idris_ms, idris_loss = idris_results[idris_key]
                ratio = idris_ms / py_ms if py_ms > 0 else 0
                print(
                    f"{name:<15} {idris_ms:>12.1f} {py_ms:>14.1f} {ratio:>7.2f}x {idris_loss:>12.6f} {py_loss:>14.6f}"
                )
            else:
                print(f"{name:<15} {'N/A':>12} {py_ms:>14.1f} {'N/A':>8} {'N/A':>12} {py_loss:>14.6f}")
        else:
            print(f"{name:<15} {'N/A':>12} {py_ms:>14.1f} {'N/A':>8} {'N/A':>12} {py_loss:>14.6f}")


if __name__ == "__main__":
    main()
