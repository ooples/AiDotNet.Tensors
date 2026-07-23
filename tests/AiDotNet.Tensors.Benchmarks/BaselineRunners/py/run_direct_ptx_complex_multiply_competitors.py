#!/usr/bin/env python3
"""Resident PyTorch CUDA peers for issue #850 complex multiplication."""

import argparse
import json
import statistics
import sys
import time

import torch


WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 50
PAIR_COUNTS = (65536, 262144, 1048576, 4194304)


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
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


def record(run, pairs, method, operation, maximum_error):
    device = measure_device(operation)
    end_to_end = measure_e2e(operation)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = operation()
    torch.cuda.synchronize()
    temporary_bytes = max(0, torch.cuda.max_memory_allocated() - baseline)
    del result
    seconds = device[1] * 1e-6
    return {
        "status": "ok",
        "run": run,
        "pairs": pairs,
        "method": method,
        "device_mean_us": device[0],
        "device_median_us": device[1],
        "device_p95_us": device[2],
        "device_p99_us": device[3],
        "e2e_mean_us": end_to_end[0],
        "e2e_median_us": end_to_end[1],
        "e2e_p95_us": end_to_end[2],
        "e2e_p99_us": end_to_end[3],
        "gflops": 6.0 * pairs / seconds / 1e9,
        "effective_gbps": 3.0 * pairs * 2 * 4 / seconds / 1e9,
        "managed_bytes": 0,
        "temporary_device_bytes": temporary_bytes,
        "max_error": maximum_error,
    }


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
        torch.manual_seed(20260722 + run)
        for pairs in PAIR_COUNTS:
            left_parts = (torch.rand((pairs, 2), device=device) * 2.0 - 1.0) * 2.0
            right_parts = (torch.rand((pairs, 2), device=device) * 2.0 - 1.0) * 2.0
            left = torch.view_as_complex(left_parts)
            right = torch.view_as_complex(right_parts)
            output = torch.empty_like(left)

            def eager_operation():
                return torch.mul(left, right, out=output)

            expected = torch.complex(
                left_parts[:, 0].double() * right_parts[:, 0].double()
                - left_parts[:, 1].double() * right_parts[:, 1].double(),
                left_parts[:, 0].double() * right_parts[:, 1].double()
                + left_parts[:, 1].double() * right_parts[:, 0].double(),
            )
            eager_operation()
            torch.cuda.synchronize()
            maximum_error = (output.to(torch.complex128) - expected).abs().max().item()

            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    eager_operation()
            torch.cuda.current_stream().wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                eager_operation()

            def graph_operation():
                graph.replay()
                return output

            for method, operation in (
                ("PyTorch CUDA eager", eager_operation),
                ("PyTorch CUDA graph", graph_operation),
            ):
                print(json.dumps(
                    record(run, pairs, method, operation, maximum_error),
                    separators=(",", ":"),
                ))
            del expected, output, left, right, left_parts, right_parts, graph
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
