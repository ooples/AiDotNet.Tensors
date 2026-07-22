#!/usr/bin/env python3
"""Forced PyTorch CUDA peers for issue #834 FP32 paged-prefill semantics.

PyTorch exposes no public block-table SDPA ABI. Logical K/V gathering is
therefore prepared outside the timed region and the strongest forced dense
backend is measured with the exact absolute-position causal mask. Unsupported
backends are printed as skips and are never silently substituted.
"""

import argparse
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


D = 64
SHAPES = (
    ("prefill-mha", 8, 8, 4, 12),
    ("prefill-gqa", 8, 2, 8, 24),
    ("prefill-mqa", 8, 1, 16, 48),
    ("prefill-long", 8, 2, 16, 112),
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


def emit(record, json_lines):
    if json_lines:
        print(json.dumps(record, separators=(",", ":")))
        return
    print(
        f"{record['run']:3d} {record['shape']:<12} {record['method']:<27} "
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

    device = torch.device("cuda")
    if not args.json_lines:
        print(
            f"GPU: {torch.cuda.get_device_name(device)}; torch={torch.__version__}; "
            f"CUDA={torch.version.cuda}; FP32; logical page gather excluded; "
            "30 warmups + 101 synchronized E2E samples"
        )
        print(
            f"{'Run':>3} {'Shape':<12} {'Method':<27} {'median us':>10} "
            f"{'p95 us':>10} {'p99 us':>10} {'mean us':>10} {'GFLOPS':>9} "
            f"{'peak B':>12} {'max err':>10}"
        )
        print("-" * 129)

    backends = (
        ("PyTorch Flash-SDPA", SDPBackend.FLASH_ATTENTION),
        ("PyTorch cuDNN-SDPA", SDPBackend.CUDNN_ATTENTION),
        ("PyTorch Efficient-SDPA", SDPBackend.EFFICIENT_ATTENTION),
        ("PyTorch Math-SDPA", SDPBackend.MATH),
    )
    for run in range(1, args.runs + 1):
        torch.manual_seed(20261000 + run)
        for shape_name, hq, hkv, queries, start in SHAPES:
            maximum_key_length = start + queries
            q = (torch.rand((1, hq, queries, D), device=device, dtype=torch.float32) - 0.5) * 0.5
            # Represents the already-gathered logical view of paged K/V. Page
            # translation is intentionally excluded, favoring the competitor.
            k = (torch.rand((1, hkv, maximum_key_length, D), device=device, dtype=torch.float32) - 0.5) * 0.5
            v = (torch.rand((1, hkv, maximum_key_length, D), device=device, dtype=torch.float32) - 0.5) * 0.5
            query_positions = torch.arange(start, start + queries, device=device).unsqueeze(1)
            key_positions = torch.arange(maximum_key_length, device=device).unsqueeze(0)
            mask = key_positions <= query_positions

            def operation():
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0,
                    is_causal=False, scale=0.125, enable_gqa=hq != hkv
                )

            with sdpa_kernel(backends=[SDPBackend.MATH]):
                reference = operation()
            for backend_name, backend in backends:
                try:
                    with sdpa_kernel(backends=[backend]):
                        probe = operation()
                        max_error = (probe - reference).abs().max().item()
                        del probe
                        timing = measure(operation)
                except Exception as exc:
                    if not args.json_lines:
                        print(
                            f"SKIP run={run} {shape_name} {backend_name}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    continue
                key_visits = queries * (start + 1) + queries * (queries - 1) // 2
                flops = 4.0 * hq * key_visits * D
                timing.update({
                    "run": run,
                    "shape": shape_name,
                    "method": backend_name,
                    "gflops": flops / (timing["median_us"] * 1e-6) / 1e9,
                    "max_error": max_error,
                })
                emit(timing, args.json_lines)
            del reference, q, k, v, mask, query_positions, key_positions
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
