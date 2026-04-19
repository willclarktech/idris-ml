"""Compare C backend operator benchmarks against PyTorch.

Runs both suites, parses output, prints a comparison table.

Usage:
    cd pytorch && uv run python -m torch_ref.compare_ops [--backend tape|mlx|torch]
"""

import re
import subprocess
import sys

from torch_ref.bench_ops import main as pytorch_main


def parse_bench_output(output: str) -> dict[str, float]:
    """Parse benchmark output lines into {label: ms} dict."""
    results: dict[str, float] = {}
    pattern = re.compile(r"^(.+?):\t([\d.]+) ms")
    for line in output.splitlines():
        m = pattern.match(line)
        if m:
            label = m.group(1).strip()
            ms = float(m.group(2))
            results[label] = ms
    return results


def run_c_bench() -> dict[str, float]:
    """Run the C bench_ops binary and parse output."""
    import os

    # Find project root (parent of pytorch/)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bench_bin = os.path.join(root, "build", "bench_ops")
    result = subprocess.run(
        [bench_bin],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=root,
    )
    return parse_bench_output(result.stdout)


def run_pytorch_bench() -> dict[str, float]:
    """Run PyTorch benchmarks, capturing stdout."""
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        pytorch_main()
    return parse_bench_output(f.getvalue())


def main() -> None:
    print("Running C backend benchmarks...")
    c_results = run_c_bench()

    print("Running PyTorch benchmarks...")
    pt_results = run_pytorch_bench()

    # Collect all labels in order of appearance
    all_labels = list(dict.fromkeys(list(c_results.keys()) + list(pt_results.keys())))

    print()
    print("=" * 80)
    print("Operator Benchmark Comparison: C Backend vs PyTorch")
    print("=" * 80)
    print()

    # Print header
    hdr = f"{'Operation':<35} {'Backend (ms)':>12} {'PyTorch (ms)':>12} {'Ratio':>8}"
    print(hdr)
    print("-" * len(hdr))

    current_section = ""
    for label in all_labels:
        # Detect section from label prefix
        section = ""
        if "matmul" in label and "vec" not in label:
            section = "Matrix multiply"
        elif "matvec" in label:
            section = "Matrix-vector"
        elif "add+mul" in label:
            section = "Element-wise"
        elif "softmax" in label:
            section = "Softmax"
        elif "conv2d" in label:
            section = "Conv2d"
        elif "train_step" in label:
            section = "Training step"

        if section and section != current_section:
            if current_section:
                print()
            current_section = section

        c_ms = c_results.get(label)
        pt_ms = pt_results.get(label)

        c_str = f"{c_ms:.3f}" if c_ms is not None else "---"
        pt_str = f"{pt_ms:.3f}" if pt_ms is not None else "---"

        if c_ms is not None and pt_ms is not None and pt_ms > 0:
            ratio = c_ms / pt_ms
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "---"

        print(f"{label:<35} {c_str:>12} {pt_str:>12} {ratio_str:>8}")

    print()
    print("Ratio = Backend / PyTorch (lower is better for backend)")
    print("Note: C backend benchmarks bypass Idris/Chez Scheme overhead.")
    print("End-to-end training adds ~50ms/epoch Chez runtime overhead.")
    print("Run `make bench-compare` for full training loop comparison.")


if __name__ == "__main__":
    main()
