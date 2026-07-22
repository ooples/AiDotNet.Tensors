#!/usr/bin/env python3
"""Exact-shape PyTorch CUDA competitors for issue #838 normalization kernels."""

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


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    return (
        statistics.fmean(values), percentile(values, 0.50),
        percentile(values, 0.95), percentile(values, 0.99)
    )


def discard(value):
    if isinstance(value, tuple):
        for item in value:
            if item is not None:
                del item
    del value


def measure_device(operation):
    for _ in range(WARMUPS):
        discard(operation())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(DEVICE_LAUNCHES):
            result = operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(SAMPLES):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / DEVICE_LAUNCHES)
    discard(result)
    return summarize(values)


def measure_e2e(operation):
    for _ in range(WARMUPS):
        discard(operation())
    torch.cuda.synchronize()
    values = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        result = operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1000.0)
        discard(result)
    return summarize(values)


def peak_temporary_bytes(operation, output_bytes):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    discard(result)
    return max(0, peak - baseline - output_bytes)


def print_header():
    print("PyTorch CUDA competitors (same shapes, precision, epsilon, and output contract)")
    print(
        "run extent operation                 method            "
        "dev mean/med/p95/p99 us       e2e mean/med/p95/p99 us       tmpB"
    )
    print("-" * 145)


def emit(run, extent, name, method, device, e2e, temporary_bytes):
    print(
        f"{run:3d} {extent:6d} {name:<25} {method:<17} "
        f"{device[0]:6.2f}/{device[1]:6.2f}/{device[2]:6.2f}/{device[3]:6.2f}   "
        f"{e2e[0]:6.2f}/{e2e[1]:6.2f}/{e2e[2]:6.2f}/{e2e[3]:6.2f}   "
        f"{temporary_bytes:10d}"
    )


def benchmark(run, extent, name, operation, output_bytes, include_compile):
    try:
        emit(run, extent, name, "PyTorch eager",
             measure_device(operation), measure_e2e(operation),
             peak_temporary_bytes(operation, output_bytes))
    except Exception as exception:
        print(f"SKIP run={run} {name} eager: {type(exception).__name__}: {exception}")
        return
    if not include_compile:
        return
    try:
        compiled = torch.compile(
            operation, fullgraph=True, mode="max-autotune-no-cudagraphs")
        discard(compiled())
        torch.cuda.synchronize()
        emit(run, extent, name, "PyTorch compile",
             measure_device(compiled), measure_e2e(compiled),
             peak_temporary_bytes(compiled, output_bytes))
        del compiled
    except Exception as exception:
        print(f"SKIP run={run} {name} compile: {type(exception).__name__}: {exception}")


def row_cells(run, include_compile):
    for rows in ROWS:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20263800 + run * 100 + rows)
        x = torch.rand((rows, D), device="cuda", generator=generator) * 1.5 - 0.75
        dy = torch.rand((rows, D), device="cuda", generator=generator) * 0.5 - 0.25
        gamma = 0.75 + torch.arange(D, device="cuda", dtype=torch.float32) / 256.0
        beta = torch.rand((D,), device="cuda", generator=generator) * 0.1 - 0.05
        mean = x.mean(dim=1)
        variance = x.var(dim=1, correction=0)
        rstd = torch.rsqrt(variance + EPSILON)
        norms = torch.linalg.vector_norm(x, dim=1)
        scalar_gradient = torch.rand((rows,), device="cuda", generator=generator) * 0.5 - 0.25
        floats = rows * D * 4

        benchmark(run, rows, "LayerNorm forward",
                  lambda: torch.ops.aten.native_layer_norm.default(
                      x, [D], gamma, beta, EPSILON), floats + rows * 8, include_compile)
        benchmark(run, rows, "LayerNorm backward",
                  lambda: torch.ops.aten.native_layer_norm_backward.default(
                      dy, x, [D], mean[:, None], rstd[:, None], gamma, beta,
                      [True, True, True]), floats + D * 8, include_compile)
        benchmark(run, rows, "RMSNorm forward",
                  lambda: F.rms_norm(x, [D], gamma, EPSILON), floats, include_compile)

        def rms_backward():
            inverse_rms = torch.rsqrt((x * x).mean(dim=1, keepdim=True) + EPSILON)
            scaled = dy * gamma
            projection = (scaled * x).mean(dim=1, keepdim=True)
            grad_input = scaled * inverse_rms - x * projection * inverse_rms.pow(3)
            grad_gamma = (dy * x * inverse_rms).sum(dim=0)
            return grad_input, grad_gamma

        benchmark(run, rows, "RMSNorm backward", rms_backward,
                  floats + D * 4, include_compile)
        benchmark(run, rows, "L2 norm axis",
                  lambda: torch.linalg.vector_norm(x, dim=1), rows * 4, include_compile)
        benchmark(run, rows, "L2 norm backward",
                  lambda: scalar_gradient[:, None] * x / norms[:, None], floats, include_compile)
        benchmark(run, rows, "NormalizeL2",
                  lambda: F.normalize(x, p=2.0, dim=1), floats, include_compile)
        benchmark(run, rows, "NormalizeRowsFused",
                  lambda: F.normalize(x, p=2.0, dim=1), floats, include_compile)
        benchmark(run, rows, "ReduceNormL2",
                  lambda: torch.linalg.vector_norm(x), 4, include_compile)

        xh = x.half()
        dyh = dy.half()
        gammah = gamma.half()
        betah = beta.half()
        half_mean = xh.float().mean(dim=1)
        half_variance = xh.float().var(dim=1, correction=0)
        half_rstd = torch.rsqrt(half_variance + EPSILON)
        half_bytes = rows * D * 2
        benchmark(run, rows, "FP16 LayerNorm forward",
                  lambda: torch.ops.aten.native_layer_norm.default(
                      xh, [D], gammah, betah, EPSILON), half_bytes + rows * 8,
                  include_compile)
        benchmark(run, rows, "FP16 LayerNorm backward",
                  lambda: torch.ops.aten.native_layer_norm_backward.default(
                      dyh, xh, [D], half_mean[:, None], half_rstd[:, None], gammah,
                      betah, [True, False, False]), half_bytes, include_compile)
        benchmark(run, rows, "FP16 LayerNorm params",
                  lambda: torch.ops.aten.native_layer_norm_backward.default(
                      dyh, xh, [D], half_mean[:, None], half_rstd[:, None], gammah,
                      betah, [False, True, True]), D * 4, include_compile)

        del (x, dy, gamma, beta, mean, variance, rstd, norms, scalar_gradient,
             xh, dyh, gammah, betah, half_mean, half_variance, half_rstd)
        torch.cuda.empty_cache()


def channel_cells(run, include_compile):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20264800 + run)

    batch, channels, spatial = 8, 64, 8
    elements = batch * channels * spatial
    x = torch.rand((batch, channels, spatial), device="cuda", generator=generator) * 1.5 - 0.75
    residual = torch.rand(x.shape, device="cuda", generator=generator) * 0.4 - 0.2
    dy = torch.rand(x.shape, device="cuda", generator=generator) * 0.5 - 0.25
    gamma = 0.75 + torch.arange(channels, device="cuda", dtype=torch.float32) / 256.0
    beta = torch.rand((channels,), device="cuda", generator=generator) * 0.1 - 0.05
    running_mean = torch.rand((channels,), device="cuda", generator=generator) * 0.1 - 0.05
    running_variance = 0.75 + torch.rand((channels,), device="cuda", generator=generator) * 0.5
    training_result = torch.ops.aten.native_batch_norm.default(
        x, gamma, beta, None, None, True, 0.1, EPSILON)
    save_mean, save_rstd = training_result[1], training_result[2]
    del training_result
    output_bytes = elements * 4
    benchmark(run, elements, "BatchNorm training",
              lambda: torch.ops.aten.native_batch_norm.default(
                  x, gamma, beta, None, None, True, 0.1, EPSILON),
              output_bytes + channels * 8, include_compile)
    benchmark(run, elements, "BatchNorm inference",
              lambda: F.batch_norm(x, running_mean, running_variance, gamma, beta,
                                   False, 0.0, EPSILON), output_bytes, include_compile)
    activations = (
        ("BatchNorm+ReLU", torch.relu),
        ("BatchNorm+GELU", lambda value: F.gelu(value, approximate="tanh")),
        ("BatchNorm+Sigmoid", torch.sigmoid),
        ("BatchNorm+Tanh", torch.tanh),
    )
    for name, activation in activations:
        benchmark(run, elements, name,
                  lambda activation=activation: activation(F.batch_norm(
                      x, running_mean, running_variance, gamma, beta,
                      False, 0.0, EPSILON)), output_bytes, include_compile)
    benchmark(run, elements, "Residual+BN+ReLU",
              lambda: torch.relu(F.batch_norm(
                  x, running_mean, running_variance, gamma, beta,
                  False, 0.0, EPSILON) + residual), output_bytes, include_compile)
    benchmark(run, elements, "BatchNorm backward",
              lambda: torch.ops.aten.native_batch_norm_backward.default(
                  dy, x, gamma, None, None, save_mean, save_rstd, True, EPSILON,
                  [True, True, True]), output_bytes + channels * 8, include_compile)
    del (x, residual, dy, gamma, beta, running_mean, running_variance,
         save_mean, save_rstd)
    torch.cuda.empty_cache()

    batch, channels, groups, spatial = 32, 64, 8, 8
    elements = batch * channels * spatial
    x = torch.rand((batch, channels, spatial), device="cuda", generator=generator) * 1.5 - 0.75
    right = torch.rand(x.shape, device="cuda", generator=generator) * 0.4 - 0.2
    dy = torch.rand(x.shape, device="cuda", generator=generator) * 0.5 - 0.25
    gamma = 0.75 + torch.arange(channels, device="cuda", dtype=torch.float32) / 256.0
    beta = torch.rand((channels,), device="cuda", generator=generator) * 0.1 - 0.05
    stats = torch.ops.aten.native_group_norm.default(
        x, gamma, beta, batch, channels, spatial, groups, EPSILON)
    mean, rstd = stats[1], stats[2]
    del stats
    output_bytes = elements * 4

    def group_forward_exact():
        output, saved_mean, saved_rstd = torch.ops.aten.native_group_norm.default(
            x, gamma, beta, batch, channels, spatial, groups, EPSILON)
        saved_variance = saved_rstd.reciprocal().square() - EPSILON
        return output, saved_mean, saved_variance

    benchmark(run, elements, "GroupNorm forward", group_forward_exact,
              output_bytes + batch * groups * 8, include_compile)
    benchmark(run, elements, "GroupNorm+Swish",
              lambda: F.silu(F.group_norm(x, groups, gamma, beta, EPSILON)),
              output_bytes, include_compile)
    benchmark(run, elements, "Add+GroupNorm",
              lambda: F.group_norm(x + right, groups, gamma, beta, EPSILON),
              output_bytes, include_compile)
    xh = x.half()
    benchmark(run, elements, "FP16 GroupNorm+Swish",
              lambda: F.silu(F.group_norm(
                  xh.float(), groups, gamma, beta, EPSILON)).half(),
              elements * 2, include_compile)
    benchmark(run, elements, "GroupNorm backward",
              lambda: torch.ops.aten.native_group_norm_backward.default(
                  dy, x, mean, rstd, gamma, batch, channels, spatial, groups,
                  [True, True, True]), output_bytes + channels * 8, include_compile)
    del x, right, dy, gamma, beta, mean, rstd, xh
    torch.cuda.empty_cache()

    batch, channels, spatial = 32, 64, 64
    groups = channels
    elements = batch * channels * spatial
    x = torch.rand((batch, channels, spatial), device="cuda", generator=generator) * 1.5 - 0.75
    dy = torch.rand(x.shape, device="cuda", generator=generator) * 0.5 - 0.25
    gamma = 0.75 + torch.arange(channels, device="cuda", dtype=torch.float32) / 256.0
    beta = torch.rand((channels,), device="cuda", generator=generator) * 0.1 - 0.05
    stats = torch.ops.aten.native_group_norm.default(
        x, gamma, beta, batch, channels, spatial, groups, EPSILON)
    mean, rstd = stats[1], stats[2]
    del stats
    output_bytes = elements * 4
    benchmark(run, elements, "InstanceNorm forward",
              lambda: torch.ops.aten.native_group_norm.default(
                  x, gamma, beta, batch, channels, spatial, groups, EPSILON),
              output_bytes + batch * channels * 8, include_compile)
    benchmark(run, elements, "InstanceNorm backward",
              lambda: torch.ops.aten.native_group_norm_backward.default(
                  dy, x, mean, rstd, gamma, batch, channels, spatial, groups,
                  [True, True, True]), output_bytes + channels * 8, include_compile)
    del x, dy, gamma, beta, mean, rstd
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--scope", choices=("all", "row", "channel"), default="all")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not torch.cuda.is_available():
        print("CUDA-enabled Python PyTorch is required.", file=sys.stderr)
        return 2
    torch.set_grad_enabled(False)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; "
        f"CUDA={torch.version.cuda}; {WARMUPS} warmups + {SAMPLES} samples"
    )
    print_header()
    for run in range(1, args.runs + 1):
        if args.scope != "channel":
            row_cells(run, not args.no_compile)
        if args.scope != "row":
            channel_cells(run, not args.no_compile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
