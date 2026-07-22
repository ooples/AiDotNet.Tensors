#!/usr/bin/env python3
"""Forced native PyTorch CUDA peers for issue #834 Flash backward.

The forward graph is built outside timing. Each sample measures only the
native SDPA backward selected by its explicitly forced backend, including the
framework launch and synchronization. Unsupported backends are recorded as
skips and are never allowed to silently fall back.
"""

import argparse
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


D = 64
SCALE = 0.125
SHAPES = (
    ("flash-bwd-mha", 8, 16, 16, False, False),
    ("flash-bwd-rect", 8, 16, 32, False, False),
    ("flash-bwd-bias", 8, 16, 32, False, True),
    ("flash-bwd-causal", 8, 32, 32, True, False),
    ("flash-bwd-long", 8, 64, 64, True, False),
)
BACKENDS = (
    ("cuDNN", SDPBackend.CUDNN_ATTENTION),
    ("Flash", SDPBackend.FLASH_ATTENTION),
    ("Efficient", SDPBackend.EFFICIENT_ATTENTION),
    ("Math", SDPBackend.MATH),
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


def reference(query, key, value, grad_output, is_causal, attention_bias):
    q = query.double().detach().requires_grad_(True)
    k = key.double().detach().requires_grad_(True)
    v = value.double().detach().requires_grad_(True)
    scores = torch.matmul(q, k.transpose(-2, -1)) * SCALE
    if attention_bias is not None:
        scores = scores + attention_bias.double()
    if is_causal:
        mask = torch.ones(
            (scores.shape[-2], scores.shape[-1]),
            dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v)
    return torch.autograd.grad(output, (q, k, v), grad_output.double())


def max_error(actual, expected):
    return max((a.double() - e).abs().max().item() for a, e in zip(actual, expected))


def emit(record, json_lines):
    if json_lines:
        print(json.dumps(record, separators=(",", ":")))
        return
    if record.get("status") == "skip":
        print(
            f"{record['run']:3d} {record['shape']:<16} {record['method']:<24} "
            f"SKIP {record['reason']}")
        return
    print(
        f"{record['run']:3d} {record['shape']:<16} {record['method']:<24} "
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

    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda")
    if not args.json_lines:
        print(
            f"GPU: {torch.cuda.get_device_name(device)}; torch={torch.__version__}; "
            f"CUDA={torch.version.cuda}; native FP32 SDPA backward only; "
            "30 warmups + 101 synchronized E2E samples"
        )
        print(
            f"{'Run':>3} {'Shape':<16} {'Method':<24} {'median us':>10} "
            f"{'p95 us':>10} {'p99 us':>10} {'mean us':>10} {'GFLOPS':>9} "
            f"{'peak B':>12} {'max err':>10}"
        )
        print("-" * 136)

    for run in range(1, args.runs + 1):
        torch.manual_seed(20261200 + run)
        for shape_name, heads, sq, sk, is_causal, has_bias in SHAPES:
            grad_output = (torch.rand((1, heads, sq, D), device=device) - 0.5) * 0.5
            query_seed = (torch.rand((1, heads, sq, D), device=device) - 0.5) * 0.5
            key_seed = (torch.rand((1, heads, sk, D), device=device) - 0.5) * 0.5
            value_seed = (torch.rand((1, heads, sk, D), device=device) - 0.5) * 0.5
            attention_bias = ((torch.rand((1, heads, sq, sk), device=device) - 0.5) * 0.5
                              if has_bias else None)
            expected = reference(
                query_seed, key_seed, value_seed, grad_output, is_causal, attention_bias)
            useful_flops = 8.0 * heads * sq * sk * D

            for backend_name, backend in BACKENDS:
                try:
                    query = query_seed.detach().requires_grad_(True)
                    key = key_seed.detach().requires_grad_(True)
                    value = value_seed.detach().requires_grad_(True)
                    with sdpa_kernel([backend]):
                        output = functional.scaled_dot_product_attention(
                            query, key, value, attn_mask=attention_bias,
                            is_causal=is_causal, scale=SCALE)

                    def operation():
                        return torch.autograd.grad(
                            output, (query, key, value), grad_output,
                            retain_graph=True)

                    probe = operation()
                    error = max_error(probe, expected)
                    del probe
                    timing = measure(operation)
                    timing.update({
                        "status": "ok",
                        "run": run,
                        "shape": shape_name,
                        "method": f"PyTorch {backend_name}",
                        "gflops": useful_flops / (timing["median_us"] * 1e-6) / 1e9,
                        "max_error": error,
                    })
                    emit(timing, args.json_lines)
                    del output, query, key, value
                except Exception as exc:
                    emit({
                        "status": "skip",
                        "run": run,
                        "shape": shape_name,
                        "method": f"PyTorch {backend_name}",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }, args.json_lines)
            del expected, grad_output, query_seed, key_seed, value_seed, attention_bias
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
