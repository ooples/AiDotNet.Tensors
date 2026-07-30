#!/usr/bin/env python3
"""Hard NVIDIA SDPA baselines for the direct-PTX release gate.

Uses CUDA events and resident tensors. Every backend is forced explicitly;
unsupported combinations are reported rather than silently falling back.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


SHAPES = ((1, 16), (1, 128), (12, 16), (12, 32), (12, 64), (12, 128), (128, 128), (512, 128))
D = 64
DEVICE_LAUNCHES_PER_SAMPLE = 10
MIXED_COMPUTE_CONFLICT_THRESHOLD_PERCENT = 5


def require_no_foreign_compute(label):
    monitor = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    conflicts = []
    for line in monitor.stdout.splitlines():
        cells = line.split()
        if not cells or cells[0].startswith("#") or len(cells) < 9:
            continue
        try:
            process_id = int(cells[1])
        except ValueError:
            continue
        process_type, sm_utilization = cells[2], cells[3]
        try:
            sm_percent = int(sm_utilization)
        except ValueError:
            sm_percent = 0
        active_compute = process_type.upper() == "C" or (
            "C" in process_type.upper()
            and sm_percent > MIXED_COMPUTE_CONFLICT_THRESHOLD_PERCENT
        )
        if process_id != os.getpid() and active_compute:
            conflicts.append(
                f"pid={process_id} {cells[-1]} type={process_type} sm={sm_utilization}%"
            )
    if conflicts:
        raise RuntimeError(
            f"[{label}] foreign GPU workload detected; clean benchmark refused: "
            + "; ".join(conflicts)
        )
    temperature = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    if temperature.isdigit() and int(temperature) > 75:
        raise RuntimeError(
            f"[{label}] GPU temperature {temperature} C exceeds the 75 C evidence ceiling"
        )


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
    device_timings = []
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    result_bytes = 0
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = None
        for _launch in range(DEVICE_LAUNCHES_PER_SAMPLE):
            result = fn()
        stop.record()
        stop.synchronize()
        device_timings.append(
            start.elapsed_time(stop) * 1000.0 / DEVICE_LAUNCHES_PER_SAMPLE
        )
        assert result is not None
        result_bytes = max(result_bytes, result.numel() * result.element_size())
        del result
    peak_memory = max(0, torch.cuda.max_memory_allocated() - baseline_memory)
    temporary_memory = max(0, peak_memory - result_bytes)

    e2e_timings = []
    for _ in range(samples):
        start_ns = time.perf_counter_ns()
        result = fn()
        torch.cuda.synchronize()
        e2e_timings.append((time.perf_counter_ns() - start_ns) / 1000.0)
        del result

    def summarize(boundary, timings):
        return {
            "boundary": boundary,
            "median_us": percentile(timings, 0.50),
            "p95_us": percentile(timings, 0.95),
            "p99_us": percentile(timings, 0.99),
            "mean_us": statistics.fmean(timings),
            "managed_bytes_per_call": None,
            "peak_device_bytes": peak_memory,
            "temporary_device_bytes": temporary_memory,
            "registers_per_thread": None,
            "static_shared_bytes": None,
            "dynamic_shared_bytes": None,
            "local_bytes_per_thread": None,
            "occupancy": None,
        }

    return [summarize("device", device_timings), summarize("E2E", e2e_timings)]


def emit(record, json_lines):
    if json_lines:
        print(json.dumps(record, separators=(",", ":")))
    else:
        print(
            f"{record['shape']:<12} {record['mode']:<7} {record['lane']:<10} "
            f"{record['boundary']:<7} {record['method']:<31} {record['median_us']:10.2f} {record['p95_us']:10.2f} "
            f"{record['p99_us']:10.2f} {record['mean_us']:10.2f} {record['gflops']:10.2f} "
            f"{record['tflops']:9.3f} {'n/a':>10} "
            f"{record['peak_device_bytes']:12d} {record['temporary_device_bytes']:12d} "
            f"{record['max_error']:10.4g} {record['max_relative_error']:10.4g} "
            f"{'n/a':>6} {'n/a':>8} {'n/a':>9} {'n/a':>8} {'n/a':>7}"
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
        print(
            f"samples=101; warmups=30; device launches/sample={DEVICE_LAUNCHES_PER_SAMPLE}; "
            "E2E launches/sample=1; FP16 Q/K/V; FP32 output"
        )
        print(f"{'Shape':<12} {'Mode':<7} {'Lane':<10} {'Bound':<7} {'Method':<31} {'median us':>10} {'p95 us':>10} {'p99 us':>10} {'mean us':>10} {'GFLOPS':>10} {'TFLOPS':>9} {'managed B':>10} {'peak B':>12} {'tmp B':>12} {'max abs':>10} {'max rel':>10} {'regs':>6} {'static B':>8} {'dynamic B':>9} {'local B':>8} {'occ':>7}")
        print("-" * 262)

    backends = (
        ("PyTorch Flash-SDPA", SDPBackend.FLASH_ATTENTION),
        ("PyTorch cuDNN-SDPA", SDPBackend.CUDNN_ATTENTION),
        ("PyTorch Efficient-SDPA", SDPBackend.EFFICIENT_ATTENTION),
        ("PyTorch Math-SDPA", SDPBackend.MATH),
    )
    for batch_heads, sequence in SHAPES:
        require_no_foreign_compute(f"python-baseline-BH{batch_heads}S{sequence}")
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
                successful_candidates = 0
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
                        difference = (candidate_output - reference).abs()
                        max_error = difference.max().item()
                        max_relative_error = (
                            2.0 * difference /
                            (candidate_output.abs() + reference.abs() + 1e-3)
                        ).max().item()
                        del difference, candidate_output
                        tolerance = 3e-3 if fused else 5e-4
                        if not math.isfinite(max_error) or max_error > tolerance:
                            raise RuntimeError(
                                f"maximum absolute error {max_error:.9g} exceeds {tolerance:.9g}"
                            )
                        timings = measure(operation)
                    except Exception as exc:
                        if not args.json_lines:
                            print(f"SKIP BH{batch_heads}S{sequence} {name} compiled={compiled}: {type(exc).__name__}: {exc}")
                        continue
                    flops = 4.0 * batch_heads * sequence * sequence * D
                    for timing in timings:
                        timing.update({
                            "shape": f"BH{batch_heads}S{sequence}",
                            "mode": "causal" if causal else "plain",
                            "lane": "attn+epi" if fused else "attention",
                            "method": method,
                            "gflops": flops / (timing["median_us"] * 1e-6) / 1e9,
                            "tflops": flops / (timing["median_us"] * 1e-6) / 1e12,
                            "max_error": max_error,
                            "max_relative_error": max_relative_error,
                        })
                        emit(timing, args.json_lines)
                    successful_candidates += 1
                try:
                    method, operation = run_flash_attention_package(q, k, v, gamma, beta, causal, fused)
                    candidate_output = operation()
                    difference = (candidate_output - reference).abs()
                    max_error = difference.max().item()
                    max_relative_error = (
                        2.0 * difference /
                        (candidate_output.abs() + reference.abs() + 1e-3)
                    ).max().item()
                    del difference, candidate_output
                    tolerance = 3e-3 if fused else 5e-4
                    if not math.isfinite(max_error) or max_error > tolerance:
                        raise RuntimeError(
                            f"maximum absolute error {max_error:.9g} exceeds {tolerance:.9g}"
                        )
                    timings = measure(operation)
                    flops = 4.0 * batch_heads * sequence * sequence * D
                    for timing in timings:
                        timing.update({
                            "shape": f"BH{batch_heads}S{sequence}",
                            "mode": "causal" if causal else "plain",
                            "lane": "attn+epi" if fused else "attention",
                            "method": method,
                            "gflops": flops / (timing["median_us"] * 1e-6) / 1e9,
                            "tflops": flops / (timing["median_us"] * 1e-6) / 1e12,
                            "max_error": max_error,
                            "max_relative_error": max_relative_error,
                        })
                        emit(timing, args.json_lines)
                    successful_candidates += 1
                except (ImportError, RuntimeError) as exc:
                    if not args.json_lines:
                        print(f"SKIP FlashAttention package: {type(exc).__name__}: {exc}")
                if successful_candidates == 0:
                    raise RuntimeError(
                        f"no valid GPU competitor for BH{batch_heads}S{sequence} "
                        f"causal={causal} fused={fused}"
                    )
                del reference
        del q, k, v, gamma, beta
        require_no_foreign_compute(f"python-baseline-BH{batch_heads}S{sequence}-end")

    require_no_foreign_compute("python-baseline-end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
