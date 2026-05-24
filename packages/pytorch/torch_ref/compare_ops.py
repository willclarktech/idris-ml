"""Compare C backend operator benchmarks against PyTorch.

Discovers all available backend binaries (bench_ops_tape, bench_ops_mlx,
bench_ops_torch) and runs them alongside PyTorch for comparison.

Usage:
    make bench-ops-compare          # build all backends + compare
    cd pytorch && uv run python -m torch_ref.compare_ops   # compare only
"""

import contextlib
import io
import os
import platform
import re
import subprocess

import torch

from torch_ref.bench_ops import main as pytorch_main

BACKENDS = ["tape", "mlx", "torch"]


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


def find_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_c_bench(backend: str) -> dict[str, float] | None:
    """Run the C bench_ops binary for a specific backend. Returns None if unavailable."""
    root = find_project_root()
    bench_bin = os.path.join(root, "build", f"bench_ops_{backend}")
    if not os.path.exists(bench_bin):
        return None
    try:
        result = subprocess.run(
            [bench_bin],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=root,
        )
        parsed = parse_bench_output(result.stdout)
        if result.returncode != 0 and not parsed:
            print(f"  WARNING: bench_ops_{backend} crashed with no output")
            return None
        if result.returncode != 0:
            print(f"  ({len(parsed)} benchmarks captured before exit)")
        return parsed if parsed else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"  WARNING: bench_ops_{backend} timed out or not found")
        return None


def run_pytorch_bench() -> dict[str, float]:
    """Run PyTorch benchmarks, capturing stdout."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        pytorch_main()
    return parse_bench_output(f.getvalue())


def classify_op(label: str) -> str:
    if "matmul" in label and "vec" not in label:
        return "Matrix multiply"
    if "matvec" in label:
        return "Matrix-vector"
    if "add+mul" in label:
        return "Element-wise"
    if "softmax" in label:
        return "Softmax"
    if "conv2d" in label:
        return "Conv2d"
    if "train_step" in label:
        return "Training step"
    return ""


def fmt_ms(ms: float | None) -> str:
    return f"{ms:.3f}" if ms is not None else "---"


def fmt_ratio(backend_ms: float | None, pt_ms: float | None) -> str:
    if backend_ms is not None and pt_ms is not None and pt_ms > 0:
        return f"{backend_ms / pt_ms:.2f}x"
    return "---"


def print_system_info(available: dict[str, dict[str, float]]) -> None:
    """Print system information header."""
    print()
    print("=" * 90)
    print("System Information")
    print("=" * 90)
    mac_ver = platform.mac_ver()[0]
    if platform.system() == "Darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        cpu = platform.processor()
    mem_gb = ""
    if platform.system() == "Darwin":
        mem_bytes = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        with contextlib.suppress(ValueError):
            mem_gb = f"{int(mem_bytes) / (1024**3):.0f} GB"
    print(f"  Platform:  {platform.system()} {platform.release()} ({platform.machine()})")
    if mac_ver:
        print(f"  macOS:     {mac_ver}")
    print(f"  CPU:       {cpu}")
    if mem_gb:
        print(f"  Memory:    {mem_gb}")
    print(f"  PyTorch:   {torch.__version__}")
    print("  Precision: float32 (single)")
    print(f"  Backends:  {', '.join(available.keys())}")


def main() -> None:
    # Discover available backends
    available: dict[str, dict[str, float]] = {}
    for backend in BACKENDS:
        print(f"Running {backend} backend benchmarks...")
        results = run_c_bench(backend)
        if results:
            available[backend] = results
            print(f"  {len(results)} benchmarks collected")
        else:
            print("  skipped (not built)")

    print("Running PyTorch benchmarks...")
    pt_results = run_pytorch_bench()

    if not available:
        print("\nNo backend benchmarks available. Run: make bench-ops-compare")
        return

    print_system_info(available)

    # Collect all labels in order
    all_labels: list[str] = []
    for results in [*available.values(), pt_results]:
        for label in results:
            if label not in all_labels:
                all_labels.append(label)

    # Build header
    backend_names = list(available.keys())
    print()
    print("=" * 90)
    print("Operator Benchmark Comparison: C Backends vs PyTorch")
    print("=" * 90)
    print()

    op_w = 28
    col_w = 11
    ratio_w = 7

    hdr = f"{'Operation':<{op_w}}"
    for name in backend_names:
        hdr += f" {name + ' (ms)':>{col_w}}"
    hdr += f" {'PyTorch':>{col_w}}"
    for name in backend_names:
        hdr += f" {name[:5]:>{ratio_w}}"
    print(hdr)
    print("-" * len(hdr))

    current_section = ""
    for label in all_labels:
        section = classify_op(label)
        if section and section != current_section:
            if current_section:
                print()
            current_section = section

        line = f"{label:<{op_w}}"
        for name in backend_names:
            line += f" {fmt_ms(available[name].get(label)):>{col_w}}"
        line += f" {fmt_ms(pt_results.get(label)):>{col_w}}"
        for name in backend_names:
            line += f" {fmt_ratio(available[name].get(label), pt_results.get(label)):>{ratio_w}}"
        print(line)

    # Per-backend category summaries
    cat_map: dict[str, callable] = {
        "BLAS (matmul)": lambda lbl: "matmul" in lbl and "vec" not in lbl,
        "BLAS (matvec)": lambda lbl: "matvec" in lbl,
        "Element-wise": lambda lbl: "add+mul" in lbl,
        "Softmax": lambda lbl: "softmax" in lbl,
        "Conv2d": lambda lbl: "conv2d" in lbl,
        "Train step": lambda lbl: "train_step" in lbl,
    }

    print()
    print("Summary (average ratio by category, <1 = faster than PyTorch):")
    sum_hdr = f"  {'Category':<20}"
    for name in backend_names:
        sum_hdr += f" {name:>10}"
    print(sum_hdr)
    print("  " + "-" * (len(sum_hdr) - 2))

    for cat, pred in cat_map.items():
        line = f"  {cat:<20}"
        for name in backend_names:
            ratios = []
            for label in all_labels:
                if not pred(label):
                    continue
                c_ms = available[name].get(label)
                pt_ms = pt_results.get(label)
                if c_ms is not None and pt_ms is not None and pt_ms > 0:
                    ratios.append(c_ms / pt_ms)
            if ratios:
                avg = sum(ratios) / len(ratios)
                line += f" {avg:>9.2f}x"
            else:
                line += f" {'---':>10}"
        print(line)

    print()
    print("Ratio = Backend / PyTorch (<1 = faster, >1 = slower)")
    print()
    print("These measure raw C backend speed (no Idris/Chez overhead).")
    print("End-to-end Idris training adds ~50ms/epoch Chez runtime overhead")
    print("(GC, thunk evaluation, allocation — not FFI marshaling).")
    print("Run `make bench-compare` for full Idris vs PyTorch training comparison.")


if __name__ == "__main__":
    main()
