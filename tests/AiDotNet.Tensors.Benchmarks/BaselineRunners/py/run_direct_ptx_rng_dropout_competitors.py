#!/usr/bin/env python3
"""Resident PyTorch CUDA peers for issue #849 dropout forward + saved mask."""

import argparse
import json
import statistics
import sys
import time

import torch


P = 0.1
WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 10
SHAPES = (
    ("dropout-n4096", 4096),
    ("dropout-n65536", 65536),
    ("dropout-n1048576", 1048576),
)


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
    values = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(SAMPLES):
        start.record()
        for _ in range(DEVICE_LAUNCHES):
            operation()
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


def validate(sample, source):
    output, mask = sample
    expected = source * mask.to(source.dtype) / (1.0 - P)
    return (output - expected).abs().max().item()


def emit(run, shape, elements, method, operation, source, max_error):
    device = measure_device(operation)
    e2e = measure_e2e(operation)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = operation()
    torch.cuda.synchronize()
    peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    del result
    useful_bytes = 3.0 * elements * 4
    record = {
        "status": "ok",
        "run": run,
        "shape": shape,
        "method": method,
        "device_mean_us": device[0],
        "device_median_us": device[1],
        "device_p95_us": device[2],
        "device_p99_us": device[3],
        "e2e_mean_us": e2e[0],
        "e2e_median_us": e2e[1],
        "e2e_p95_us": e2e[2],
        "e2e_p99_us": e2e[3],
        "gb_per_second": useful_bytes / (device[1] * 1e-6) / 1e9,
        "peak_device_bytes": peak,
        "max_error": max_error,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
    }
    print(json.dumps(record, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not torch.cuda.is_available():
        print("CUDA-enabled Python PyTorch is required.", file=sys.stderr)
        return 2

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    for run in range(1, args.runs + 1):
        torch.manual_seed(849_000 + run)
        torch.cuda.manual_seed_all(849_000 + run)
        for shape, elements in SHAPES:
            source = torch.linspace(-1.0, 1.0, elements, device=device)

            def eager():
                return torch.ops.aten.native_dropout.default(source, P, True)

            eager_probe = eager()
            eager_error = validate(eager_probe, source)
            del eager_probe

            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    graph_result = eager()
            torch.cuda.current_stream().wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_result = eager()

            def graph_operation():
                graph.replay()
                return graph_result

            graph_error = validate(graph_operation(), source)

            compiled = torch.compile(eager, mode="max-autotune", fullgraph=True)
            for _ in range(3):
                compiled_result = compiled()
            torch.cuda.synchronize()
            compiled_error = validate(compiled_result, source)
            del compiled_result

            emit(run, shape, elements, "PyTorch native_dropout eager", eager, source, eager_error)
            emit(run, shape, elements, "PyTorch CUDA graph", graph_operation, source, graph_error)
            emit(run, shape, elements, "PyTorch compile max-autotune", compiled, source, compiled_error)
            del compiled, graph, graph_result, source
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
