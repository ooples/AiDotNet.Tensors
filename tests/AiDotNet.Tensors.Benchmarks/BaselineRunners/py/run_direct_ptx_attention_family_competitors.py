#!/usr/bin/env python3
"""Forced PyTorch CUDA baselines for the direct-PTX v3 attention family.

The shape/mask contract matches AiDotNet ScaledDotProductAttention: dense BHSD,
GQA without expanded K/V, and bottom-right causal alignment for rectangular
attention. Unsupported forced backends are reported rather than substituted.
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
    ("rect-mha", 2, 8, 8, 32, 64),
    ("rect-gqa", 2, 8, 2, 32, 64),
    ("rect-mqa", 2, 8, 1, 32, 64),
    ("long-gqa", 1, 16, 4, 64, 128),
    ("q-long-gqa", 1, 16, 4, 128, 64),
)


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bottom_right_causal_mask(sq, skv, device):
    query = torch.arange(sq, device=device).unsqueeze(1)
    key = torch.arange(skv, device=device).unsqueeze(0)
    return key <= query + (skv - sq)


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
        f"{record['run']:3d} {record['shape']:<12} {record['mode']:<7} "
        f"{record['method']:<27} {record['median_us']:10.2f} {record['p95_us']:10.2f} "
        f"{record['p99_us']:10.2f} {record['mean_us']:10.2f} {record['tflops']:9.3f} "
        f"{record['peak_device_bytes']:12d} {record['max_error']:10.4g}"
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
            f"CUDA={torch.version.cuda}; 30 warmups + 101 samples"
        )
        print(
            f"{'Run':>3} {'Shape':<12} {'Mode':<7} {'Method':<27} "
            f"{'median us':>10} {'p95 us':>10} {'p99 us':>10} {'mean us':>10} "
            f"{'TFLOPS':>9} {'peak B':>12} {'max err':>10}"
        )
        print("-" * 139)

    backends = (
        ("PyTorch Flash-SDPA", SDPBackend.FLASH_ATTENTION),
        ("PyTorch cuDNN-SDPA", SDPBackend.CUDNN_ATTENTION),
        ("PyTorch Efficient-SDPA", SDPBackend.EFFICIENT_ATTENTION),
        ("PyTorch Math-SDPA", SDPBackend.MATH),
    )
    for run in range(1, args.runs + 1):
        torch.manual_seed(20260800 + run)
        for name, batch, hq, hkv, sq, skv in SHAPES:
            q = (torch.rand((batch, hq, sq, D), device=device, dtype=torch.float16) - 0.5) * 0.5
            k = (torch.rand((batch, hkv, skv, D), device=device, dtype=torch.float16) - 0.5) * 0.5
            v = (torch.rand((batch, hkv, skv, D), device=device, dtype=torch.float16) - 0.5) * 0.5
            for causal in (False, True):
                mask = bottom_right_causal_mask(sq, skv, device) if causal else None

                def operation():
                    return F.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, dropout_p=0.0,
                        is_causal=False, scale=1.0 / 8.0, enable_gqa=hq != hkv
                    ).float()

                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    reference = operation()
                for backend_name, backend in backends:
                    try:
                        with sdpa_kernel(backends=[backend]):
                            probe = operation()
                            max_error = (probe - reference).abs().max().item()
                            del probe
                            # The forced-backend context is outside the timed
                            # call, matching a resident backend configuration.
                            # Each sample includes public dispatch + completion,
                            # just like the C# Stopwatch/Synchronize harness.
                            timing = measure(operation)
                    except Exception as exc:
                        if not args.json_lines:
                            print(
                                f"SKIP run={run} {name} {'causal' if causal else 'plain'} "
                                f"{backend_name}: {type(exc).__name__}: {exc}"
                            )
                        continue
                    flops = 4.0 * batch * hq * sq * skv * D
                    timing.update({
                        "run": run,
                        "shape": name,
                        "mode": "causal" if causal else "plain",
                        "method": backend_name,
                        "tflops": flops / (timing["median_us"] * 1e-6) / 1e12,
                        "max_error": max_error,
                    })
                    emit(timing, args.json_lines)
                del reference
                if mask is not None:
                    del mask
            del q, k, v
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
