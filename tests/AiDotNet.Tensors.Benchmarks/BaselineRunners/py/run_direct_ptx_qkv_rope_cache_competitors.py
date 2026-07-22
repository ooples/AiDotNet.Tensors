#!/usr/bin/env python3
"""PyTorch CUDA peer for issue #835 fused decode QKV/RoPE/cache."""

import argparse
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as functional


D = 64
WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 10
SHAPES = (
    ("decode-h4", 4, 16, 0),
    ("decode-h8", 8, 64, 17),
    ("decode-h16", 16, 128, 127),
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


def rope(value, cosine, sine, position):
    even = value[..., 0::2]
    odd = value[..., 1::2]
    c = cosine[position]
    s = sine[position]
    return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)


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
    parser.add_argument("--json-lines", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not torch.cuda.is_available():
        print("CUDA-enabled Python PyTorch is required.", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    for run in range(1, args.runs + 1):
        torch.manual_seed(20261300 + run)
        for name, heads, capacity, position in SHAPES:
            model = heads * D
            x = (torch.rand((1, model), device=device) * 2.0 - 1.0) * 0.125
            weights = (torch.rand((3 * model, model), device=device) * 2.0 - 1.0) * 0.0625
            bias = (torch.rand((3 * model,), device=device) * 2.0 - 1.0) * 0.0625
            positions = torch.arange(capacity, device=device, dtype=torch.float32).unsqueeze(1)
            pairs = torch.arange(D // 2, device=device, dtype=torch.float32).unsqueeze(0)
            angles = positions * torch.pow(10000.0, -2.0 * pairs / D)
            cosine, sine = torch.cos(angles), torch.sin(angles)
            key_cache = torch.zeros((capacity, heads, D), device=device)
            value_cache = torch.zeros_like(key_cache)

            def operation():
                projected = functional.linear(x, weights, bias).view(3, heads, D)
                query = rope(projected[0], cosine, sine, position)
                key_cache[position].copy_(rope(projected[1], cosine, sine, position))
                value_cache[position].copy_(projected[2])
                return query

            probe = operation()
            projected_reference = functional.linear(
                x.double(), weights.double(), bias.double()).view(3, heads, D)
            expected_query = rope(
                projected_reference[0], cosine.double(), sine.double(), position)
            expected_key = rope(
                projected_reference[1], cosine.double(), sine.double(), position)
            max_error = max(
                (probe.double() - expected_query).abs().max().item(),
                (key_cache[position].double() - expected_key).abs().max().item(),
                (value_cache[position].double() - projected_reference[2]).abs().max().item(),
            )
            del probe

            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    graph_query = operation()
            torch.cuda.current_stream().wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_query = operation()

            def graph_operation():
                graph.replay()
                return graph_query

            # Compile and validate outside every timing distribution. This lane
            # is mandatory on the release machine: silently omitting an
            # available Inductor/Triton fusion would not establish the strongest
            # relevant PyTorch competitor.
            compiled_operation = torch.compile(
                operation, mode="max-autotune", fullgraph=True)
            for _ in range(3):
                compiled_query = compiled_operation()
            torch.cuda.synchronize()
            compiled_error = max(
                (compiled_query.double() - expected_query).abs().max().item(),
                (key_cache[position].double() - expected_key).abs().max().item(),
                (value_cache[position].double() - projected_reference[2]).abs().max().item(),
            )
            del compiled_query, projected_reference, expected_query, expected_key

            useful_flops = 6.0 * model * model
            for method, measured_operation, method_error in (
                ("PyTorch CUDA eager", operation, max_error),
                ("PyTorch CUDA graph", graph_operation, max_error),
                ("PyTorch compile max-autotune", compiled_operation, compiled_error),
            ):
                device_values = measure_device(measured_operation)
                e2e_values = measure_e2e(measured_operation)
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                result = measured_operation()
                torch.cuda.synchronize()
                peak_bytes = max(0, torch.cuda.max_memory_allocated() - baseline)
                del result
                record = {
                    "status": "ok",
                    "run": run,
                    "shape": name,
                    "method": method,
                    "device_mean_us": device_values[0],
                    "device_median_us": device_values[1],
                    "device_p95_us": device_values[2],
                    "device_p99_us": device_values[3],
                    "device_tokens_per_second": 1e6 / device_values[1],
                    "e2e_mean_us": e2e_values[0],
                    "e2e_median_us": e2e_values[1],
                    "e2e_p95_us": e2e_values[2],
                    "e2e_p99_us": e2e_values[3],
                    "e2e_tokens_per_second": 1e6 / e2e_values[1],
                    "tflops": useful_flops / (device_values[1] * 1e-6) / 1e12,
                    "peak_device_bytes": peak_bytes,
                    "max_error": method_error,
                }
                print(json.dumps(record, separators=(",", ":")))
            del compiled_operation, x, weights, bias, cosine, sine, key_cache, value_cache
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
