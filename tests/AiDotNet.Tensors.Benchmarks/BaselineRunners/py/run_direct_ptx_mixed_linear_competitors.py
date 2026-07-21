#!/usr/bin/env python3
"""PyTorch CUDA FP16 peers for issue #837 M=1 linear+bias+tanh-GELU."""

import argparse
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as functional


WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 50
SHAPES = (
    ("decode-256x256", 256, 256),
    ("decode-up-512x2048", 512, 2048),
    ("decode-up-1024x4096", 1024, 4096),
)


def summarize(values):
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return statistics.fmean(ordered), percentile(0.50), percentile(0.95), percentile(0.99)


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
        torch.manual_seed(20261500 + run)
        for name, input_features, output_features in SHAPES:
            x = (((torch.rand(input_features, device=device) * 2.0 - 1.0) * 0.125)
                 .to(torch.float16))
            weights = (((torch.rand((output_features, input_features), device=device) * 2.0 - 1.0)
                        * 0.0625).to(torch.float16))
            # FP32 bias/output matches the direct kernel's public ABI. PyTorch
            # promotes the FP16 dot result before this FP32 epilogue.
            bias = ((torch.rand(output_features, device=device) * 2.0 - 1.0) * 0.0625)

            def operation():
                projected = functional.linear(x, weights, None).float() + bias
                return functional.gelu(projected, approximate="tanh")

            probe = operation()
            expected = functional.gelu(
                functional.linear(x.double(), weights.double(), None) + bias.double(),
                approximate="tanh",
            )
            max_error = (probe.double() - expected).abs().max().item()
            del probe, expected

            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    graph_output = operation()
            torch.cuda.current_stream().wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = operation()

            def graph_operation():
                graph.replay()
                return graph_output

            for method, measured in (
                ("PyTorch FP16 eager", operation),
                ("PyTorch FP16 graph", graph_operation),
            ):
                device_values = measure_device(measured)
                e2e_values = measure_e2e(measured)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                result = measured()
                torch.cuda.synchronize()
                peak_bytes = max(0, torch.cuda.max_memory_allocated() - baseline)
                print(json.dumps({
                    "status": "ok",
                    "run": run,
                    "shape": name,
                    "method": method,
                    "device_mean_us": device_values[0],
                    "device_median_us": device_values[1],
                    "device_p95_us": device_values[2],
                    "device_p99_us": device_values[3],
                    "e2e_mean_us": e2e_values[0],
                    "e2e_median_us": e2e_values[1],
                    "e2e_p95_us": e2e_values[2],
                    "e2e_p99_us": e2e_values[3],
                    "peak_device_bytes": peak_bytes,
                    "max_error": max_error,
                }, separators=(",", ":")))
                del result
            del x, weights, bias, graph_output, graph, capture_stream
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
