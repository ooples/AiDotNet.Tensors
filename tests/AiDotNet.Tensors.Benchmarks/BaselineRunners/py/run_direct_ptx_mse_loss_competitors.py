#!/usr/bin/env python3
"""Resident PyTorch/Triton competitors for issue #847 per-sample MSE loss.

Measures the strongest resident NVIDIA paths for mean-squared-error over the
feature axis on identical device tensors: eager, CUDA-graph, max-autotune
compile, and compile+graph. Error is scored against a double-precision oracle.
CPU MKL/OpenBLAS are intentionally ineligible.
"""

import argparse
import glob
import importlib.util
import os
import platform
import statistics
import sys
import time
import tracemalloc

import torch


SHAPES = ((256, 128), (2048, 64), (2048, 128), (8192, 128))
WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 50


def configure_windows_triton():
    if os.name != "nt":
        return
    if "CC" not in os.environ:
        candidates = glob.glob(
            r"C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
        ) + glob.glob(
            r"C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
        )
        if candidates:
            os.environ["CC"] = sorted(candidates)[-1]
    spec = importlib.util.find_spec("triton")
    if spec is None or not spec.submodule_search_locations:
        return
    triton_root = os.path.join(
        next(iter(spec.submodule_search_locations)), "backends", "nvidia"
    )
    cuda_import_library = os.path.join(triton_root, "lib", "x64", "cuda.lib")
    if not os.path.isfile(cuda_import_library):
        return
    os.environ.setdefault("CUDA_PATH", triton_root)
    os.environ.setdefault("CUDA_HOME", triton_root)
    current_lib = os.environ.get("LIB", "")
    os.environ["LIB"] = os.path.dirname(cuda_import_library) + (
        os.pathsep + current_lib if current_lib else ""
    )


configure_windows_triton()


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def measure_device(operation):
    for _ in range(WARMUPS):
        result = operation()
        del result
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(DEVICE_LAUNCHES):
            result = operation()
    torch.cuda.synchronize()
    values = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(SAMPLES):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / DEVICE_LAUNCHES)
    del result
    return summarize(values)


def measure_e2e(operation):
    for _ in range(WARMUPS):
        result = operation()
        del result
    torch.cuda.synchronize()
    values = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        result = operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1000.0)
        del result
    return summarize(values)


def peak_temporary_bytes(operation):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return max(0, peak - baseline)


def managed_peak_bytes(operation):
    for _ in range(8):
        result = operation()
        del result
    torch.cuda.synchronize()
    tracemalloc.start()
    peaks = []
    try:
        for _ in range(SAMPLES):
            before = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            result = operation()
            _, peak = tracemalloc.get_traced_memory()
            peaks.append(max(0, peak - before))
            del result
        torch.cuda.synchronize()
    finally:
        tracemalloc.stop()
    return int(statistics.median(peaks))


def make_graph(operation):
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            result = operation()
            del result
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = operation()
    torch.cuda.synchronize()
    resident_bytes = max(0, torch.cuda.memory_allocated() - before)

    def replay():
        graph.replay()
        return static_output

    return replay, resident_bytes


def print_header():
    print("PyTorch GPU competitors (same resident FP32 per-sample MSE loss)")
    print(
        "run rows  cols  method                    dev mean/med/p95/p99 us       "
        "e2e mean/med/p95/p99 us       GB/s allocB tmpB err"
    )
    print("-" * 168)


def emit(run, rows, columns, method, device, e2e, managed_bytes, temporary_bytes, error):
    moved = rows * columns * 4.0 * 2.0 + rows * 4.0
    bandwidth = moved / (device["median"] * 1.0e-6) / 1.0e9
    print(
        f"{run:3d} {rows:5d} {columns:5d} {method:<25} "
        f"{device['mean']:6.2f}/{device['median']:6.2f}/{device['p95']:6.2f}/{device['p99']:6.2f}   "
        f"{e2e['mean']:6.2f}/{e2e['median']:6.2f}/{e2e['p95']:6.2f}/{e2e['p99']:6.2f}   "
        f"{bandwidth:5.1f} {managed_bytes:6d} {temporary_bytes:8d} {error:8.1E}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not torch.cuda.is_available():
        print("CUDA-enabled Python PyTorch is required.", file=sys.stderr)
        return 2

    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    print(
        f"GPU: {properties.name}; SM={capability[0]}{capability[1]}; "
        f"VRAM={properties.total_memory}; torch={torch.__version__}; "
        f"CUDA={torch.version.cuda}; Python={platform.python_version()}; "
        f"OS={platform.platform()}; {WARMUPS} warmups + {SAMPLES} samples"
    )
    print_header()
    for run in range(1, args.runs + 1):
        for rows, columns in SHAPES:
            seed = 20268470 + run * 100 + rows + columns
            base = torch.arange(rows * columns, device="cuda", dtype=torch.int64)
            pred = (((base * 17 + seed) % 257 - 128).float() / 32.0).reshape(rows, columns)
            target = (((base * 29 + seed) % 257 - 128).float() / 32.0).reshape(rows, columns)
            del base
            torch.cuda.synchronize()
            expected = (pred.double() - target.double()).pow(2).mean(dim=1)

            def eager():
                return (pred - target).pow(2).mean(dim=1)

            probe = eager()
            error = (probe.double() - expected).abs().max().item()
            del probe
            eager_device = measure_device(eager)
            emit(run, rows, columns, "PyTorch CUDA eager", eager_device,
                 measure_e2e(eager), managed_peak_bytes(eager),
                 peak_temporary_bytes(eager), error)

            try:
                graph_operation, graph_bytes = make_graph(eager)
                probe = graph_operation()
                torch.cuda.synchronize()
                graph_error = (probe.double() - expected).abs().max().item()
                emit(run, rows, columns, "PyTorch CUDA graph", eager_device,
                     measure_e2e(graph_operation), managed_peak_bytes(graph_operation),
                     graph_bytes, graph_error)
            except Exception as exception:
                print(f"SKIP run={run} rows={rows} cols={columns} PyTorch CUDA graph: "
                      f"{type(exception).__name__}: {exception}")

            try:
                compiled = torch.compile(eager, fullgraph=True, mode="max-autotune-no-cudagraphs")
                probe = compiled()
                torch.cuda.synchronize()
                compiled_error = (probe.double() - expected).abs().max().item()
                del probe
                compiled_device = measure_device(compiled)
                emit(run, rows, columns, "PyTorch compile", compiled_device,
                     measure_e2e(compiled), managed_peak_bytes(compiled),
                     peak_temporary_bytes(compiled), compiled_error)
                compiled_graph, compiled_graph_bytes = make_graph(compiled)
                probe = compiled_graph()
                torch.cuda.synchronize()
                compiled_graph_error = (probe.double() - expected).abs().max().item()
                emit(run, rows, columns, "PyTorch compile graph", compiled_device,
                     measure_e2e(compiled_graph), managed_peak_bytes(compiled_graph),
                     compiled_graph_bytes, compiled_graph_error)
            except Exception as exception:
                print(f"SKIP run={run} rows={rows} cols={columns} PyTorch compile: "
                      f"{type(exception).__name__}: {exception}")

            del pred, target, expected
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
