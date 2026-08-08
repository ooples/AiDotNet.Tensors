#!/usr/bin/env python3
"""GPU-only PyTorch/Triton competitors for issue #839's FP32 GeGLU cells."""

import argparse
import glob
import importlib.util
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F


SHAPES = ((1, 4096), (32, 4096), (256, 4096), (256, 11008))
WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 50
FLOPS_PER_OUTPUT = 10.0
BYTES_PER_OUTPUT = 12.0


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


def reference(input_tensor, half_dimension):
    value = input_tensor[:, :half_dimension].double()
    gate = input_tensor[:, half_dimension:].double()
    inner = 0.7978845608 * (gate + 0.044715 * gate * gate * gate)
    gelu = 0.5 * gate * (1.0 + torch.tanh(inner))
    return (value * gelu).float()


def reference_backward(grad_output, input_tensor, half_dimension):
    value = input_tensor[:, :half_dimension].double()
    gate = input_tensor[:, half_dimension:].double()
    grad = grad_output.double()
    inner = 0.7978845608 * (gate + 0.044715 * gate * gate * gate)
    tanh_inner = torch.tanh(inner)
    gelu = 0.5 * gate * (1.0 + tanh_inner)
    derivative = 0.5 * (1.0 + tanh_inner) + (
        0.5
        * gate
        * (1.0 - tanh_inner * tanh_inner)
        * 0.7978845608
        * (1.0 + 0.134145 * gate * gate)
    )
    return torch.cat((grad * gelu, grad * value * derivative), dim=1).float()


def print_header(phase):
    print(
        "PyTorch GPU competitors (same resident split-row FP32 tensors and "
        f"GeGLU {phase} formula)"
    )
    print(
        "run outer D     method                    dev mean/med/p95/p99 us       "
        "e2e mean/med/p95/p99 us       GFLOPS  GB/s allocB tmpB err      R/S/L/B"
    )
    print("-" * 176)


def emit(run, outer, half_dimension, method, device, e2e, temporary_bytes, error):
    outputs = outer * half_dimension
    gflops = outputs * FLOPS_PER_OUTPUT / (device["median"] * 1.0e-6) / 1.0e9
    bandwidth = outputs * BYTES_PER_OUTPUT / (device["median"] * 1.0e-6) / 1.0e9
    print(
        f"{run:3d} {outer:5d} {half_dimension:5d} {method:<25} "
        f"{device['mean']:6.2f}/{device['median']:6.2f}/{device['p95']:6.2f}/{device['p99']:6.2f}   "
        f"{e2e['mean']:6.2f}/{e2e['median']:6.2f}/{e2e['p95']:6.2f}/{e2e['p99']:6.2f}   "
        f"{gflops:7.1f} {bandwidth:5.1f} {'n/a':>6} {temporary_bytes:8d} "
        f"{error:8.1E} -/-/-/-"
    )


def main():
    global FLOPS_PER_OUTPUT, BYTES_PER_OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--phase", choices=("forward", "backward"), default="forward")
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
    if args.phase == "backward":
        FLOPS_PER_OUTPUT = 26.0
        BYTES_PER_OUTPUT = 20.0
    print_header(args.phase)
    for run in range(1, args.runs + 1):
        for outer, half_dimension in SHAPES:
            generator = torch.Generator(device="cuda")
            seed_base = 20262000 if args.phase == "backward" else 20261900
            generator.manual_seed(seed_base + run * 100 + outer + half_dimension)
            input_tensor = (
                torch.rand(
                    (outer, 2 * half_dimension), device="cuda", generator=generator
                )
                * 2.0
                - 1.0
            ) * 2.0
            if args.phase == "backward":
                grad_output = torch.rand(
                    (outer, half_dimension), device="cuda", generator=generator
                ) * 2.0 - 1.0
                expected = reference_backward(
                    grad_output, input_tensor, half_dimension
                )
                output_bytes = outer * half_dimension * 2 * 4

                def eager():
                    value = input_tensor[:, :half_dimension]
                    gate = input_tensor[:, half_dimension:]
                    inner = 0.7978845608 * (
                        gate + 0.044715 * gate * gate * gate
                    )
                    tanh_inner = torch.tanh(inner)
                    gelu = 0.5 * gate * (1.0 + tanh_inner)
                    derivative = 0.5 * (1.0 + tanh_inner) + (
                        0.5
                        * gate
                        * (1.0 - tanh_inner * tanh_inner)
                        * 0.7978845608
                        * (1.0 + 0.134145 * gate * gate)
                    )
                    return torch.cat(
                        (grad_output * gelu, grad_output * value * derivative),
                        dim=1,
                    )
            else:
                expected = reference(input_tensor, half_dimension)
                output_bytes = outer * half_dimension * 4

                def eager():
                    value = input_tensor[:, :half_dimension]
                    gate = input_tensor[:, half_dimension:]
                    return value * F.gelu(gate, approximate="tanh")

            probe = eager()
            error = (probe - expected).abs().max().item()
            del probe
            eager_device = measure_device(eager)
            emit(
                run,
                outer,
                half_dimension,
                "PyTorch CUDA eager",
                eager_device,
                measure_e2e(eager),
                peak_temporary_bytes(eager, output_bytes),
                error,
            )

            try:
                graph_operation, graph_bytes = make_graph(eager, output_bytes)
                probe = graph_operation()
                torch.cuda.synchronize()
                graph_error = (probe - expected).abs().max().item()
                emit(
                    run,
                    outer,
                    half_dimension,
                    "PyTorch CUDA graph",
                    eager_device,
                    measure_e2e(graph_operation),
                    graph_bytes,
                    graph_error,
                )
            except Exception as exception:
                print(
                    f"SKIP run={run} outer={outer} D={half_dimension} PyTorch CUDA graph: "
                    f"{type(exception).__name__}: {exception}"
                )

            try:
                compiled = torch.compile(
                    eager, fullgraph=True, mode="max-autotune-no-cudagraphs"
                )
                probe = compiled()
                torch.cuda.synchronize()
                compiled_error = (probe - expected).abs().max().item()
                del probe
                compiled_device = measure_device(compiled)
                emit(
                    run,
                    outer,
                    half_dimension,
                    "PyTorch compile",
                    compiled_device,
                    measure_e2e(compiled),
                    peak_temporary_bytes(compiled, output_bytes),
                    compiled_error,
                )
                compiled_graph, compiled_graph_bytes = make_graph(compiled, output_bytes)
                probe = compiled_graph()
                torch.cuda.synchronize()
                compiled_graph_error = (probe - expected).abs().max().item()
                emit(
                    run,
                    outer,
                    half_dimension,
                    "PyTorch compile graph",
                    compiled_device,
                    measure_e2e(compiled_graph),
                    compiled_graph_bytes,
                    compiled_graph_error,
                )
            except Exception as exception:
                print(
                    f"SKIP run={run} outer={outer} D={half_dimension} PyTorch compile: "
                    f"{type(exception).__name__}: {exception}"
                )

            del input_tensor, expected
            if args.phase == "backward":
                del grad_output
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
