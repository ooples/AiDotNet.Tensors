#!/usr/bin/env python3
"""Resident PyTorch CUDA eager/graph peers for issue #836 operation families."""

import argparse
import json
import math
import platform
import statistics
import sys
import time

import torch
import torch.nn.functional as functional


WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 10


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    return (
        statistics.fmean(values),
        percentile(values, 0.50),
        percentile(values, 0.95),
        percentile(values, 0.99),
    )


def measure_device(operation):
    for _ in range(WARMUPS):
        operation()
    torch.cuda.synchronize()
    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(SAMPLES):
        start.record()
        for _ in range(DEVICE_LAUNCHES):
            operation()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / DEVICE_LAUNCHES)
    return summarize(timings)


def measure_e2e(operation):
    for _ in range(WARMUPS):
        operation()
    torch.cuda.synchronize()
    timings = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - start) / 1000.0)
    return summarize(timings)


def maximum_error(actual, expected):
    if isinstance(actual, tuple):
        return max(
            (part.double() - oracle).abs().max().item()
            for part, oracle in zip(actual, expected)
        )
    return (actual.double() - expected).abs().max().item()


def make_case(name, run, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(20263000 + run * 100 + sum(ord(c) for c in name))

    def values(shape, scale=0.125, dtype=torch.float32):
        return (torch.rand(shape, generator=generator, device=device, dtype=dtype) * 2 - 1) * scale

    if name == "decode-gelu":
        m, k, n = 1, 512, 2048
        x, w, bias = values((m, k)), values((k, n), 0.0625), values((n,), 0.0625)

        def op():
            return functional.gelu(x @ w + bias, approximate="tanh")

        expected = functional.gelu(x.double() @ w.double() + bias.double(), approximate="tanh")
        return "M1 K512 N2048", 2.0 * m * k * n, op, expected

    if name == "gemm-fp32":
        m, k, n = 64, 256, 256
        left, right = values((m, k)), values((k, n), 0.0625)

        def op():
            return left @ right

        return "M64 K256 N256", 2.0 * m * k * n, op, left.double() @ right.double()

    if name == "fused-gelu":
        m, k, n = 64, 256, 256
        x, w, bias = values((m, k)), values((k, n), 0.0625), values((n,), 0.0625)

        def op():
            return functional.gelu(x @ w + bias, approximate="tanh")

        expected = functional.gelu(x.double() @ w.double() + bias.double(), approximate="tanh")
        return "M64 K256 N256", 2.0 * m * k * n, op, expected

    if name == "batched-gemm":
        batch, m, k, n = 4, 64, 256, 256
        left, right = values((batch, m, k)), values((batch, k, n), 0.0625)

        def op():
            return torch.bmm(left, right)

        return "B4 M64 K256 N256", 2.0 * batch * m * k * n, op, torch.bmm(left.double(), right.double())

    if name == "gemm-fp16":
        m, k, n = 16, 32, 16
        left, right = values((m, k), dtype=torch.float16), values((k, n), 0.0625, torch.float16)

        def op():
            return (left @ right).float()

        return "M16 K32 N16", 2.0 * m * k * n, op, left.double() @ right.double()

    if name == "lora":
        batch, input_features, rank, output_features = 8, 256, 8, 256
        scale = 0.125
        x = values((batch, input_features))
        base = values((batch, output_features))
        a = values((input_features, rank), 0.0625)
        b = values((rank, output_features), 0.0625)

        def op():
            return base + scale * ((x @ a) @ b)

        expected = base.double() + scale * ((x.double() @ a.double()) @ b.double())
        work = 2.0 * batch * input_features * rank + 2.0 * batch * rank * output_features
        return "B8 I256 R8 O256", work, op, expected

    if name == "linear-ce-index":
        rows, hidden, vocab = 4, 16, 32
        x, w, bias = values((rows, hidden)), values((hidden, vocab), 0.0625), values((vocab,), 0.03125)
        target = torch.tensor([1, 7, 15, 31], device=device, dtype=torch.long)

        def op():
            return functional.cross_entropy(x @ w + bias, target)

        expected = functional.cross_entropy(x.double() @ w.double() + bias.double(), target)
        return "B4 K16 V32", 4.0 * rows * hidden * vocab, op, expected

    if name == "linear-backward-relu":
        m, k, n = 64, 256, 256
        grad = values((m, n))
        x = values((m, k))
        w = values((k, n), 0.0625)
        saved = values((m, n), 0.25)

        def op():
            masked = grad * (saved > 0)
            return masked @ w.t(), x.t() @ masked, masked.sum(dim=0)

        masked64 = grad.double() * (saved > 0)
        expected = (masked64 @ w.double().t(), x.double().t() @ masked64, masked64.sum(dim=0))
        work = 2.0 * m * n * (k + k) + m * n
        return "M64 K256 N256", work, op, expected

    if name == "dot":
        length = 4096
        left, right = values((length,)), values((length,))

        def op():
            return torch.dot(left, right)

        return "K4096", 2.0 * length, op, torch.dot(left.double(), right.double())

    if name == "outer":
        m, n = 64, 128
        left, right = values((m,)), values((n,))

        def op():
            return torch.outer(left, right)

        return "M64 N128", float(m * n), op, torch.outer(left.double(), right.double())

    if name == "batched-dot":
        batch, dimension = 4, 512
        left, right = values((batch, dimension)), values((batch, dimension))

        def op():
            return (left * right).sum(dim=1)

        return "B4 K512", 2.0 * batch * dimension, op, (left.double() * right.double()).sum(dim=1)

    if name == "strided-dot":
        length = 512
        left, right = values((length,)), values((length,))

        def op():
            return torch.dot(left, torch.flip(right, (0,)))

        return "A512 B512 reverse", 2.0 * length, op, torch.dot(left.double(), torch.flip(right.double(), (0,)))

    raise ValueError(name)


CASES = (
    "decode-gelu",
    "gemm-fp32",
    "fused-gelu",
    "batched-gemm",
    "gemm-fp16",
    "lora",
    "linear-ce-index",
    "linear-backward-relu",
    "dot",
    "outer",
    "batched-dot",
    "strided-dot",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json-lines", action="store_true")
    parser.add_argument("--only", choices=CASES)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not torch.cuda.is_available():
        print("CUDA-enabled Python PyTorch is required.", file=sys.stderr)
        return 2

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    software_fingerprint = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }
    for run in range(1, args.runs + 1):
        operations = (args.only,) if args.only else CASES
        for operation in operations:
            shape, work, eager, expected = make_case(operation, run, device)
            eager_probe = eager()
            eager_error = maximum_error(eager_probe, expected)

            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    graph_output = eager()
            torch.cuda.current_stream().wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = eager()

            def graph_operation():
                graph.replay()
                return graph_output

            for method, measured, error in (
                ("PyTorch CUDA eager", eager, eager_error),
                ("PyTorch CUDA graph", graph_operation, eager_error),
            ):
                device_values = measure_device(measured)
                e2e_values = measure_e2e(measured)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                result = measured()
                torch.cuda.synchronize()
                peak_bytes = max(0, torch.cuda.max_memory_allocated() - baseline)
                median_us = device_values[1]
                record = {
                    "status": "ok",
                    "run": run,
                    "operation": operation,
                    "shape": shape,
                    "method": method,
                    "work_flops": work,
                    "device_mean_us": device_values[0],
                    "device_median_us": device_values[1],
                    "device_p95_us": device_values[2],
                    "device_p99_us": device_values[3],
                    "e2e_mean_us": e2e_values[0],
                    "e2e_median_us": e2e_values[1],
                    "e2e_p95_us": e2e_values[2],
                    "e2e_p99_us": e2e_values[3],
                    "tflops": work / (median_us * 1_000_000.0),
                    "gflops": work / (median_us * 1_000.0),
                    "managed_bytes": None,
                    "temporary_device_bytes": peak_bytes,
                    "peak_device_bytes": peak_bytes,
                    "max_error": error,
                    **software_fingerprint,
                }
                print(json.dumps(record, separators=(",", ":")))
                del result
            del eager_probe, expected, graph_output, graph, capture_stream
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
