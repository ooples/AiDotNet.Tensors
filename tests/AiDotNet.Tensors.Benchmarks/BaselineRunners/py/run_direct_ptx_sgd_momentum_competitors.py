#!/usr/bin/env python3
"""Resident PyTorch/Triton competitors for issue #848 fused SGD-with-momentum.

The naive PyTorch path runs three elementwise ops where the first materializes
the decayed gradient (grad + wd*param) — the intermediate the fused direct-PTX
kernel elides. Measures eager, CUDA-graph, max-autotune compile, and
compile+graph. A small learning rate keeps the repeated update loop bounded.
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


SIZES = (65_536, 262_144, 1_048_576, 4_194_304)
LR = 1e-6
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
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
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(DEVICE_LAUNCHES):
            operation()
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
    return summarize(values)


def measure_e2e(operation):
    for _ in range(WARMUPS):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1000.0)
    return summarize(values)


def peak_temporary_bytes(operation):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return max(0, peak - baseline)


def print_header():
    print("PyTorch GPU competitors (same resident FP32 SGD-momentum 3-op update)")
    print(
        "run size      method                    dev mean/med/p95/p99 us       "
        "e2e mean/med/p95/p99 us       GB/s tmpB"
    )
    print("-" * 150)


def emit(run, size, method, device, e2e, temporary_bytes):
    moved = size * 4.0 * 5.0
    bandwidth = moved / (device["median"] * 1.0e-6) / 1.0e9
    print(
        f"{run:3d} {size:9d} {method:<25} "
        f"{device['mean']:6.2f}/{device['median']:6.2f}/{device['p95']:6.2f}/{device['p99']:6.2f}   "
        f"{e2e['mean']:6.2f}/{e2e['median']:6.2f}/{e2e['p95']:6.2f}/{e2e['p99']:6.2f}   "
        f"{bandwidth:5.1f} {temporary_bytes:8d}"
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
        for size in SIZES:
            seed = 20268480 + run * 100 + (size % 9973)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            param = torch.rand(size, device="cuda", generator=generator) * 2 - 1
            grad = (torch.rand(size, device="cuda", generator=generator) * 2 - 1) * 0.5
            velocity = (torch.rand(size, device="cuda", generator=generator) * 2 - 1) * 0.25

            def eager():
                with torch.no_grad():
                    decayed = grad.add(param, alpha=WEIGHT_DECAY)  # materialized intermediate
                    velocity.mul_(MOMENTUM).add_(decayed)
                    param.add_(velocity, alpha=-LR)

            eager_device = measure_device(eager)
            emit(run, size, "PyTorch SGD 3-op eager", eager_device,
                 measure_e2e(eager), peak_temporary_bytes(eager))

            try:
                compiled = torch.compile(eager, fullgraph=False, mode="max-autotune-no-cudagraphs")
                compiled()
                torch.cuda.synchronize()
                compiled_device = measure_device(compiled)
                emit(run, size, "PyTorch SGD compile", compiled_device,
                     measure_e2e(compiled), peak_temporary_bytes(compiled))
            except Exception as exception:
                print(f"SKIP run={run} size={size} PyTorch compile: "
                      f"{type(exception).__name__}: {exception}")

            del param, grad, velocity
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
