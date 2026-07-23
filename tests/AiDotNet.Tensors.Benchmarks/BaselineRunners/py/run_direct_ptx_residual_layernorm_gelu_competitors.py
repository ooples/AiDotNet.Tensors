#!/usr/bin/env python3
"""GPU-only PyTorch competitors for issue #838's exact D=64 boundary."""

import argparse
import glob
import importlib.util
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F


D = 64
ROWS = (256, 2048, 8192)
WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 50
EPSILON = 1.0e-5


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


def peak_temporary_bytes(operation, output_bytes):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return max(0, peak - baseline - output_bytes)


def make_graph(operation, output_bytes):
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
    resident_bytes = max(0, torch.cuda.memory_allocated() - before - output_bytes)

    def replay():
        graph.replay()
        return static_output

    return replay, resident_bytes


def reference(input_tensor, residual, bias, gamma, beta):
    values = input_tensor.double() + residual.double() + bias.double()
    mean = values.mean(dim=-1, keepdim=True)
    variance = ((values - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (values - mean) / torch.sqrt(variance + EPSILON)
    affine = normalized * gamma.double() + beta.double()
    return (0.5 * affine * (1.0 + torch.tanh(
        0.7978845608 * (affine + 0.044715 * affine * affine * affine)))).float()


def print_header():
    print("PyTorch GPU competitors (same resident FP32 tensors and formula)")
    print(
        "run rows  method                    dev mean/med/p95/p99 us       "
        "e2e mean/med/p95/p99 us       GFLOPS  GB/s allocB tmpB err      R/S/L/B"
    )
    print("-" * 164)


def emit(run, rows, method, device, e2e, temporary_bytes, error):
    operations = (19.0 * D + 3.0) * rows
    gflops = operations / (device["median"] * 1.0e-6) / 1.0e9
    useful_bytes = (3.0 * rows * D + 3.0 * D) * 4.0
    bandwidth = useful_bytes / (device["median"] * 1.0e-6) / 1.0e9
    print(
        f"{run:3d} {rows:5d} {method:<25} "
        f"{device['mean']:6.2f}/{device['median']:6.2f}/{device['p95']:6.2f}/{device['p99']:6.2f}   "
        f"{e2e['mean']:6.2f}/{e2e['median']:6.2f}/{e2e['p95']:6.2f}/{e2e['p99']:6.2f}   "
        f"{gflops:7.1f} {bandwidth:5.1f} {'n/a':>6} {temporary_bytes:4d} "
        f"{error:8.1E} -/-/-/-"
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

    print(
        f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; "
        f"CUDA={torch.version.cuda}; {WARMUPS} warmups + {SAMPLES} samples"
    )
    print_header()
    for run in range(1, args.runs + 1):
        for rows in ROWS:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(20261800 + run * 100 + rows)
            input_tensor = (torch.rand((rows, D), device="cuda", generator=generator) * 2.0 - 1.0)
            residual = (torch.rand((rows, D), device="cuda", generator=generator) * 2.0 - 1.0) * 0.25
            bias = (torch.rand((D,), device="cuda", generator=generator) * 2.0 - 1.0) * 0.05
            gamma = 0.75 + torch.arange(D, device="cuda", dtype=torch.float32) / 256.0
            beta = (torch.rand((D,), device="cuda", generator=generator) * 2.0 - 1.0) * 0.025
            expected = reference(input_tensor, residual, bias, gamma, beta)
            output_bytes = rows * D * 4

            def eager():
                return F.gelu(F.layer_norm(
                    input_tensor + residual + bias,
                    (D,), gamma, beta, EPSILON), approximate="tanh")

            probe = eager()
            error = (probe - expected).abs().max().item()
            del probe
            eager_device = measure_device(eager)
            emit(run, rows, "PyTorch CUDA eager", eager_device,
                 measure_e2e(eager), peak_temporary_bytes(eager, output_bytes), error)

            try:
                graph_operation, graph_bytes = make_graph(eager, output_bytes)
                probe = graph_operation()
                torch.cuda.synchronize()
                graph_error = (probe - expected).abs().max().item()
                emit(run, rows, "PyTorch CUDA graph", eager_device,
                     measure_e2e(graph_operation), graph_bytes, graph_error)
            except Exception as exception:
                print(f"SKIP run={run} rows={rows} PyTorch CUDA graph: {type(exception).__name__}: {exception}")

            try:
                compiled = torch.compile(
                    eager, fullgraph=True, mode="max-autotune-no-cudagraphs")
                probe = compiled()
                torch.cuda.synchronize()
                compiled_error = (probe - expected).abs().max().item()
                del probe
                compiled_device = measure_device(compiled)
                emit(run, rows, "PyTorch compile", compiled_device,
                     measure_e2e(compiled), peak_temporary_bytes(compiled, output_bytes), compiled_error)
                compiled_graph, compiled_graph_bytes = make_graph(compiled, output_bytes)
                probe = compiled_graph()
                torch.cuda.synchronize()
                compiled_graph_error = (probe - expected).abs().max().item()
                emit(run, rows, "PyTorch compile graph", compiled_device,
                     measure_e2e(compiled_graph), compiled_graph_bytes, compiled_graph_error)
            except Exception as exception:
                print(f"SKIP run={run} rows={rows} PyTorch compile: {type(exception).__name__}: {exception}")

            del input_tensor, residual, bias, gamma, beta, expected
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
