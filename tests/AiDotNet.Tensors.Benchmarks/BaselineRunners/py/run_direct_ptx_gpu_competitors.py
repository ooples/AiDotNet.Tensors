#!/usr/bin/env python3
"""Hard NVIDIA SDPA baselines for the direct-PTX release gate.

Uses CUDA events and resident tensors. Every backend is forced explicitly;
unsupported combinations are reported rather than silently falling back.
"""

import argparse
import json
import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


SHAPES = ((12, 16), (12, 32), (12, 64), (12, 128), (128, 128), (512, 128))
D = 64


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def tanh_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def measure(fn, samples=101, warmups=30):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    timings = []
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop))
        del result
    peak_memory = max(0, torch.cuda.max_memory_allocated() - baseline_memory)
    return {
        "median_us": percentile(timings, 0.50) * 1000.0,
        "p95_us": percentile(timings, 0.95) * 1000.0,
        "p99_us": percentile(timings, 0.99) * 1000.0,
        "mean_us": statistics.fmean(timings) * 1000.0,
        "peak_device_bytes": peak_memory,
    }


def emit(record, json_lines):
    if json_lines:
        print(json.dumps(record, separators=(",", ":")))
    else:
        print(
            f"{record['shape']:<12} {record['mode']:<7} {record['lane']:<10} "
            f"{record['method']:<31} {record['median_us']:10.2f} {record['p95_us']:10.2f} "
            f"{record['p99_us']:10.2f} {record['mean_us']:10.2f} {record['tflops']:9.3f} "
            f"{record['peak_device_bytes']:12d} {record['max_error']:10.4g}"
        )


def run_sdpa_backend(name, backend, q, k, v, gamma, beta, causal, fused, compile_graph):
    def core():
        output = F.scaled_dot_product_attention(q, k, v, is_causal=causal).float()
        if fused:
            output = tanh_gelu(F.layer_norm(output, (D,), gamma, beta, 1e-5))
        return output

    method = name + ("+LN+GELU" if fused else "")
    if compile_graph:
        core = torch.compile(core, fullgraph=True, mode="max-autotune")
        method += " [compile]"

    def operation():
        with sdpa_kernel(backends=[backend]):
            return core()

    # Force compilation/backend eligibility outside the measured distribution.
    probe = operation()
    probe.sum().item()
    del probe
    return method, operation


def run_flash_attention_package(q, k, v, gamma, beta, causal, fused):
    from flash_attn import flash_attn_func

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    def operation():
        output = flash_attn_func(q_bshd, k_bshd, v_bshd, causal=causal).transpose(1, 2).float()
        if fused:
            output = tanh_gelu(F.layer_norm(output, (D,), gamma, beta, 1e-5))
        return output

    return "FlashAttention package" + ("+LN+GELU" if fused else ""), operation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-lines", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA-enabled Python PyTorch is required.", file=sys.stderr)
        return 2

    torch.manual_seed(20260720)
    device = torch.device("cuda")
    if not args.json_lines:
        print(f"GPU: {torch.cuda.get_device_name(device)}; torch={torch.__version__}; CUDA={torch.version.cuda}")
        print(f"{'Shape':<12} {'Mode':<7} {'Lane':<10} {'Method':<31} {'median us':>10} {'p95 us':>10} {'p99 us':>10} {'mean us':>10} {'TFLOPS':>9} {'peak B':>12} {'max err':>10}")
        print("-" * 152)

    backends = (
        ("PyTorch Flash-SDPA", SDPBackend.FLASH_ATTENTION),
        ("PyTorch cuDNN-SDPA", SDPBackend.CUDNN_ATTENTION),
        ("PyTorch Efficient-SDPA", SDPBackend.EFFICIENT_ATTENTION),
        ("PyTorch Math-SDPA", SDPBackend.MATH),
    )
    for batch_heads, sequence in SHAPES:
        q = (torch.rand((1, batch_heads, sequence, D), device=device, dtype=torch.float16) - 0.5) * 0.5
        k = (torch.rand_like(q) - 0.5) * 0.5
        v = (torch.rand_like(q) - 0.5) * 0.5
        gamma = torch.linspace(0.75, 0.99609375, D, device=device)
        beta = torch.linspace(-0.0625, 0.060546875, D, device=device)
        for causal in (False, True):
            for fused in (False, True):
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    reference = F.scaled_dot_product_attention(q, k, v, is_causal=causal).float()
                if fused:
                    reference = tanh_gelu(F.layer_norm(reference, (D,), gamma, beta, 1e-5))
                candidates = []
                for name, backend in backends:
                    candidates.append((name, backend, False))
                    if not args.no_compile:
                        candidates.append((name, backend, True))
                for name, backend, compiled in candidates:
                    try:
                        method, operation = run_sdpa_backend(
                            name, backend, q, k, v, gamma, beta, causal, fused, compiled)
                        candidate_output = operation()
                        max_error = (candidate_output - reference).abs().max().item()
                        del candidate_output
                        timing = measure(operation)
                    except Exception as exc:
                        if not args.json_lines:
                            print(f"SKIP BH{batch_heads}S{sequence} {name} compiled={compiled}: {type(exc).__name__}: {exc}")
                        continue
                    flops = 4.0 * batch_heads * sequence * sequence * D
                    timing.update({
                        "shape": f"BH{batch_heads}S{sequence}",
                        "mode": "causal" if causal else "plain",
                        "lane": "attn+epi" if fused else "attention",
                        "method": method,
                        "tflops": flops / (timing["median_us"] * 1e-6) / 1e12,
                        "max_error": max_error,
                    })
                    emit(timing, args.json_lines)
                try:
                    method, operation = run_flash_attention_package(q, k, v, gamma, beta, causal, fused)
                    candidate_output = operation()
                    max_error = (candidate_output - reference).abs().max().item()
                    del candidate_output
                    timing = measure(operation)
                    flops = 4.0 * batch_heads * sequence * sequence * D
                    timing.update({
                        "shape": f"BH{batch_heads}S{sequence}",
                        "mode": "causal" if causal else "plain",
                        "lane": "attn+epi" if fused else "attention",
                        "method": method,
                        "tflops": flops / (timing["median_us"] * 1e-6) / 1e12,
                        "max_error": max_error,
                    })
                    emit(timing, args.json_lines)
                except (ImportError, RuntimeError) as exc:
                    if not args.json_lines:
                        print(f"SKIP FlashAttention package: {type(exc).__name__}: {exc}")
                del reference
        del q, k, v, gamma, beta
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
