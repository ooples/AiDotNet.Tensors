#!/usr/bin/env python3
"""Exact-contract PyTorch CUDA peers for issue #834 probability backward.

Inputs include the materialized softmax probabilities, matching AiDotNet's
SDPA/GQA backward ABI. The eager composition is backed by PyTorch/cuBLAS CUDA
matmuls; torch.compile(max-autotune) is attempted as a separate lane and never
silently relabeled when unavailable.
"""

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


D = 64
SCALE = 0.125
SHAPES = (
    ("backward-mha", 8, 8, 16, 16),
    ("backward-gqa", 8, 2, 16, 32),
    ("backward-mqa", 8, 1, 32, 16),
    ("backward-long", 8, 2, 64, 64),
)


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure(operation, warmups=30, samples=101):
    for _ in range(warmups):
        result = operation()
        del result
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    timings = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        result = operation()
        torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - start) / 1000.0)
        del result
    return {
        "median_us": percentile(timings, 0.50),
        "p95_us": percentile(timings, 0.95),
        "p99_us": percentile(timings, 0.99),
        "mean_us": statistics.fmean(timings),
        "peak_device_bytes": max(0, torch.cuda.max_memory_allocated() - baseline_memory),
    }


def backward_composition(grad_output, query, key_by_query, value_by_query, probabilities, ratio, hkv):
    grad_probability = torch.matmul(grad_output, value_by_query.transpose(-2, -1))
    row_delta = (probabilities * grad_probability).sum(dim=-1, keepdim=True)
    grad_score = probabilities * (grad_probability - row_delta) * SCALE
    grad_query = torch.matmul(grad_score, key_by_query)
    grad_key_by_query = torch.matmul(grad_score.transpose(-2, -1), query)
    grad_value_by_query = torch.matmul(probabilities.transpose(-2, -1), grad_output)
    if ratio == 1:
        return grad_query, grad_key_by_query, grad_value_by_query
    grad_key = grad_key_by_query.reshape(1, hkv, ratio, grad_key_by_query.shape[-2], D).sum(dim=2)
    grad_value = grad_value_by_query.reshape(1, hkv, ratio, grad_value_by_query.shape[-2], D).sum(dim=2)
    return grad_query, grad_key, grad_value


def max_error(actual, expected):
    return max((a.double() - e).abs().max().item() for a, e in zip(actual, expected))


def emit(record, json_lines):
    if json_lines:
        print(json.dumps(record, separators=(",", ":")))
        return
    print(
        f"{record['run']:3d} {record['shape']:<14} {record['method']:<30} "
        f"{record['median_us']:10.2f} {record['p95_us']:10.2f} "
        f"{record['p99_us']:10.2f} {record['mean_us']:10.2f} "
        f"{record['gflops']:9.2f} {record['peak_device_bytes']:12d} "
        f"{record['max_error']:10.4g}"
    )


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

    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    if not args.json_lines:
        print(
            f"GPU: {torch.cuda.get_device_name(device)}; torch={torch.__version__}; "
            f"CUDA={torch.version.cuda}; exact materialized-probability FP32 backward; "
            "30 warmups + 101 synchronized E2E samples"
        )
        print(
            f"{'Run':>3} {'Shape':<14} {'Method':<30} {'median us':>10} "
            f"{'p95 us':>10} {'p99 us':>10} {'mean us':>10} {'GFLOPS':>9} "
            f"{'peak B':>12} {'max err':>10}"
        )
        print("-" * 136)

    compile_failure = None
    for run in range(1, args.runs + 1):
        torch.manual_seed(20261100 + run)
        for shape_name, hq, hkv, sq, sk in SHAPES:
            ratio = hq // hkv
            grad_output = (torch.rand((1, hq, sq, D), device=device) - 0.5) * 0.5
            query = (torch.rand((1, hq, sq, D), device=device) - 0.5) * 0.5
            key = (torch.rand((1, hkv, sk, D), device=device) - 0.5) * 0.5
            value = (torch.rand((1, hkv, sk, D), device=device) - 0.5) * 0.5
            key_by_query = key.repeat_interleave(ratio, dim=1)
            value_by_query = value.repeat_interleave(ratio, dim=1)
            logits = torch.matmul(query, key_by_query.transpose(-2, -1)) * SCALE
            probabilities = torch.softmax(logits, dim=-1)
            del logits

            def eager_operation():
                return backward_composition(
                    grad_output, query, key_by_query, value_by_query,
                    probabilities, ratio, hkv
                )

            with torch.no_grad():
                expected = backward_composition(
                    grad_output.double(), query.double(), key_by_query.double(),
                    value_by_query.double(), probabilities.double(), ratio, hkv
                )
                probe = eager_operation()
                error = max_error(probe, expected)
                del probe
                timing = measure(eager_operation)
            useful_flops = 8.0 * hq * sq * sk * D
            timing.update({
                "run": run,
                "shape": shape_name,
                "method": "PyTorch/cuBLAS eager",
                "gflops": useful_flops / (timing["median_us"] * 1e-6) / 1e9,
                "max_error": error,
            })
            emit(timing, args.json_lines)

            if compile_failure is None:
                try:
                    compiled = torch.compile(
                        eager_operation, mode="max-autotune", fullgraph=True)
                    with torch.no_grad():
                        compiled_probe = compiled()
                        compiled_error = max_error(compiled_probe, expected)
                        del compiled_probe
                        compiled_timing = measure(compiled)
                    compiled_timing.update({
                        "run": run,
                        "shape": shape_name,
                        "method": "PyTorch compile max-autotune",
                        "gflops": useful_flops / (compiled_timing["median_us"] * 1e-6) / 1e9,
                        "max_error": compiled_error,
                    })
                    emit(compiled_timing, args.json_lines)
                    del compiled
                except Exception as exc:
                    compile_failure = f"{type(exc).__name__}: {exc}"
                    if not args.json_lines:
                        print(f"SKIP PyTorch compile max-autotune: {compile_failure}")

            del expected, grad_output, query, key, value, key_by_query, value_by_query, probabilities
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
