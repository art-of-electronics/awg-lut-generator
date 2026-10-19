"""Standalone AWG performance benchmark (not a unit test).

Run from repository root: python benchmarks/benchmark.py
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.waveforms import generate_waveform  # noqa: E402
from core.modulation import combine  # noqa: E402
from services.csv_export import export_lut  # noqa: E402

SIZES = (1024, 16384, 65536)
WAVEFORMS = ("Sine", "Square", "Wave", "Gaussian", "Exponential")
OPERATIONS = ("Multiply", "Add", "Subtract")


def measure(callback, repetitions: int, warmups: int) -> tuple[float, float]:
    for _ in range(warmups):
        callback()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        callback()
        times.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(times), min(times)


def run(repetitions: int, warmups: int) -> list[tuple[str, str, int, float, float]]:
    rows = []
    headers_dir = ROOT / "headers"
    with tempfile.TemporaryDirectory(prefix="awg_benchmark_") as temp:
        output_dir = Path(temp)
        for samples in SIZES:
            envelope, _, _ = generate_waveform("Gaussian", samples, 2, 0.1)
            carrier, _, _ = generate_waveform("Sine", samples, 8, 0.1)
            exported, _ = combine("Multiply", envelope, carrier)

            tasks = []
            for name in WAVEFORMS:
                tasks.append(("generation", name, lambda name=name: generate_waveform(name, samples, 1, 0.2)))
            for operation in OPERATIONS:
                tasks.append(("modulation", operation, lambda operation=operation: combine(operation, envelope, carrier)))
            tasks.append(("export", "RAW", lambda: export_lut(exported, "bench.csv", "RAW", output_dir, headers_dir)))
            if (headers_dir / "Header_SAG1021.csv").is_file():
                tasks.append(("export", "SAG1021", lambda: export_lut(exported, "bench.csv", "SAG1021", output_dir, headers_dir)))

            for category, name, callback in tasks:
                median, minimum = measure(callback, repetitions, warmups)
                rows.append((category, name, samples, median, minimum))
                print(f"{samples:>6}  {category:<11} {name:<12} median={median:>9.3f} ms  min={minimum:>9.3f} ms")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=15, help="timed runs per case (default: 15)")
    parser.add_argument("--warmup", type=int, default=3, help="warmup runs per case (default: 3)")
    parser.add_argument("--csv", type=Path, help="optional CSV report path")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be >= 1 and --warmup must be >= 0")
    print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__}")
    print("Results are wall-clock measurements; CSV export includes disk I/O.\n")
    rows = run(args.repeat, args.warmup)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("category", "name", "samples", "median_ms", "min_ms"))
            for category, name, samples, median, minimum in rows:
                writer.writerow((category, name, samples, f"{median:.6f}", f"{minimum:.6f}"))
        print(f"\nReport: {args.csv}")


if __name__ == "__main__":
    main()
