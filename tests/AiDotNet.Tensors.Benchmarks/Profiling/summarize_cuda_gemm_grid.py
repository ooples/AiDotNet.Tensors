#!/usr/bin/env python3
"""Summarize #1261 CUDA GEMM grid evidence from several process launches into Markdown tables.

Each input is the console log of one `--direct-ptx-dense-linear-full R --only=gemm-grid[-NAME]` process. Every
(process, run) pair contributes one independent sample per cell: that run's device-time median over 101 event-timed
samples. A cell's reported value is the median of those per-run medians, with the min..max range as its spread,
so n = processes x runs.

Usage:
    python summarize_cuda_gemm_grid.py --sm-count 24 --fp32-lanes-per-sm 64 LOG [LOG ...] > table.md
"""

import argparse
import collections
import json
import statistics
import sys

EVIDENCE = "dense_linear_evidence_json="
EXTERNAL = "dense_linear_external_json="
STATUS = "dense_linear_status_json="
CLOCK = "gemm_grid_clock_json="
GRID_PREFIX = "gemm-grid-"
DIRECT_PREFIX = "Direct PTX"
VENDOR_PREFIXES = ("cuBLAS", "PyTorch")


def parse(paths):
    """Returns (samples, unsupported, skipped, clocks) parsed from the logs."""
    samples = collections.defaultdict(list)
    unsupported = {}
    skipped = {}
    clocks = collections.defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                line = raw.strip()
                for prefix in (EVIDENCE, EXTERNAL, STATUS, CLOCK):
                    if line.startswith(prefix):
                        record = json.loads(line[len(prefix):])
                        break
                else:
                    continue
                operation = record.get("operation", "")
                if not operation.startswith(GRID_PREFIX):
                    continue
                key = (operation, record.get("method", ""))
                if prefix == CLOCK:
                    if record.get("sm_clock_mhz", 0) > 0:
                        clocks[key].append(record["sm_clock_mhz"])
                elif prefix == STATUS:
                    unsupported[key] = (record.get("shape", ""), record.get("reason", ""))
                elif record.get("status", "ok") == "skip":
                    skipped[key] = (record.get("shape", ""), record.get("reason", "").splitlines()[0])
                else:
                    record["source"] = f"{path}#run{record.get('run')}"
                    samples[key].append(record)
    return samples, unsupported, skipped, clocks


def peak_tflops(clock_mhz, sm_count, lanes_per_sm):
    return sm_count * lanes_per_sm * 2 * clock_mhz * 1e6 / 1e12


def summarize_cell(records):
    medians = sorted(r["device_median_us"] for r in records)
    median = statistics.median(medians)
    return {
        "n": len(medians),
        "median_us": median,
        "min_us": medians[0],
        "max_us": medians[-1],
        "spread_pct": 100.0 * (medians[-1] - medians[0]) / median,
        "p95_us": statistics.median(r["device_p95_us"] for r in records),
        "e2e_median_us": statistics.median(r["e2e_median_us"] for r in records),
        "work": records[0]["work_flops"],
        "shape": records[0]["shape"],
        "max_error": max(r["max_error"] for r in records),
        "tolerance": records[0]["tolerance"],
    }


def format_time(us):
    return f"{us / 1000:.2f} ms" if us >= 1000 else f"{us:.1f} us"


def ordered_operations(samples, unsupported):
    """Operations in first-seen evidence order (the grid's own order), unsupported-only ones last."""
    seen = list(dict.fromkeys(op for op, _ in samples))
    seen += sorted({op for op, _ in unsupported} - set(seen))
    return seen


def cell_row(name, method, cell, clock, best_vendor, sm_count, lanes):
    tflops = cell["work"] / (cell["median_us"] * 1e6)
    percent = f"{100 * tflops / peak_tflops(clock, sm_count, lanes):.0f}%" if clock else "n/a"
    ratio = f"{best_vendor / cell['median_us']:.2f}x" if best_vendor else "n/a"
    spread = (f"+/-{cell['spread_pct'] / 2:.1f}% "
              f"({format_time(cell['min_us'])}..{format_time(cell['max_us'])})")
    return (f"| {name} ({cell['shape']}) | {method} | {cell['n']} | {format_time(cell['median_us'])} | {spread} | "
            f"{format_time(cell['p95_us'])} | {tflops:.2f} | {percent} | {ratio} | {cell['max_error']:.1e} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--sm-count", type=int, required=True)
    parser.add_argument("--fp32-lanes-per-sm", type=int, required=True)
    args = parser.parse_args()

    samples, unsupported, skipped, clocks = parse(args.logs)
    if not samples:
        print("No GEMM grid evidence found.", file=sys.stderr)
        return 1
    all_clocks = [c for values in clocks.values() for c in values]
    # The PyTorch lanes run in their own process and have no sampled clock; they use the median of all samples.
    fallback_clock = statistics.median(all_clocks) if all_clocks else None

    print("| Shape | Method | n | Median | Spread (min..max) | P95 | TFLOP/s | % peak | vs best vendor | Max rel. error |")
    print("|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    verdicts = []
    for operation in ordered_operations(samples, unsupported):
        cells = {method: summarize_cell(records)
                 for (op, method), records in samples.items() if op == operation}
        vendor = {m: c["median_us"] for m, c in cells.items() if m.startswith(VENDOR_PREFIXES)}
        best_vendor_method = min(vendor, key=vendor.get) if vendor else None
        best_vendor = vendor[best_vendor_method] if best_vendor_method else None
        name = operation[len(GRID_PREFIX):]
        for method in sorted(cells, key=lambda m: cells[m]["median_us"]):
            clock_values = clocks.get((operation, method))
            clock = statistics.median(clock_values) if clock_values else fallback_clock
            print(cell_row(name, method, cells[method], clock, best_vendor,
                           args.sm_count, args.fp32_lanes_per_sm))
            if method.startswith(DIRECT_PREFIX) and best_vendor:
                verdicts.append((name, method, best_vendor_method, best_vendor / cells[method]["median_us"]))
        for (op, method), (shape, reason) in sorted(unsupported.items()):
            if op == operation:
                print(f"| {name} ({shape}) | {method} | - | unsupported | {reason} | | | | | |")

    if verdicts:
        print()
        print("| Shape | PTX kernel | Best vendor lane | PTX speed vs best vendor |")
        print("|---|---|---|---:|")
        for name, method, vendor_method, ratio in verdicts:
            print(f"| {name} | {method} | {vendor_method} | {ratio:.2f}x |")

    if skipped:
        print()
        print("Skipped lanes:")
        for reason in sorted({reason for _, reason in skipped.values()}):
            methods = sorted({m for (_, m), (_, r) in skipped.items() if r == reason})
            print(f"- {', '.join(methods)}: {reason}")

    dotnet_ops = {op for (op, m) in samples if not m.startswith("PyTorch")}
    python_ops = {op for (op, m) in samples if m.startswith("PyTorch")}
    for missing, side in ((dotnet_ops - python_ops, "PyTorch"), (python_ops - dotnet_ops, ".NET")):
        for op in sorted(missing):
            print(f"WARNING: {op} has no {side} evidence", file=sys.stderr)
    if fallback_clock:
        print()
        print(f"% peak = TFLOP/s / ({args.sm_count} SMs x {args.fp32_lanes_per_sm} FP32 lanes x 2 FLOP x SM clock). "
              f"The clock is the one sampled after each .NET route; PyTorch lanes use the median of all samples "
              f"({fallback_clock:.0f} MHz).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
