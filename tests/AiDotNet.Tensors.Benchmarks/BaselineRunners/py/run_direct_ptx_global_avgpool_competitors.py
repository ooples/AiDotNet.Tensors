#!/usr/bin/env python3
"""Resident NVIDIA competitors for issue #842 global average pooling.

The script does not run as part of normal CI. It is consumed during the
controlled GPU evidence campaign and emits one machine-readable JSON record per
run, shape, and PyTorch route. Input and output tensors remain resident and the
output allocation is reused for every measured call.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch


WARMUPS = 30
SAMPLES = 101
LAUNCHES_PER_DEVICE_SAMPLE = 50
SHAPES = ((256, 128), (2048, 64), (2048, 128), (8192, 128))


def percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def summarize(values_us: list[float]) -> dict[str, float]:
    ordered = sorted(values_us)
    return {
        "mean_us": statistics.fmean(ordered),
        "median_us": percentile(ordered, 0.50),
        "p95_us": percentile(ordered, 0.95),
        "p99_us": percentile(ordered, 0.99),
    }


def device_distribution(launch) -> dict[str, float]:
    for _ in range(WARMUPS):
        launch()
    torch.cuda.synchronize()
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(SAMPLES):
        start.record()
        for _ in range(LAUNCHES_PER_DEVICE_SAMPLE):
            launch()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000.0 / LAUNCHES_PER_DEVICE_SAMPLE)
    return summarize(samples)


def end_to_end_distribution(launch) -> dict[str, float]:
    for _ in range(WARMUPS):
        launch()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        launch()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    return summarize(samples)


def temporary_device_bytes(launch) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    for _ in range(SAMPLES):
        launch()
    torch.cuda.synchronize()
    return max(0, torch.cuda.max_memory_allocated() - before)


def measure(run: int, rows: int, spatial: int, method: str, launch, x, output) -> None:
    launch()
    torch.cuda.synchronize()
    oracle = x.double().mean(dim=1).float()
    max_error = float((output - oracle).abs().max().item())
    device = device_distribution(launch)
    end_to_end = end_to_end_distribution(launch)
    temp_bytes = temporary_device_bytes(launch)
    bytes_moved = (rows * spatial + rows) * 4
    gbps = bytes_moved / (device["median_us"] * 1e-6) / 1e9
    record = {
        "status": "ok",
        "run": run,
        "shape": f"r{rows}_s{spatial}",
        "rows": rows,
        "spatial": spatial,
        "method": method,
        "device": device,
        "end_to_end": end_to_end,
        "effective_gbps": gbps,
        "managed_bytes_per_call": None,
        "temporary_device_bytes": temp_bytes,
        "max_abs_error": max_error,
        "launches_per_device_sample": LAUNCHES_PER_DEVICE_SAMPLE,
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    print("AIDOTNET_PTX_GAP_JSON=" + json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.manual_seed(20260722)
    torch.cuda.manual_seed_all(20260722)
    for run in range(1, args.runs + 1):
        for rows, spatial in SHAPES:
            x = torch.randn((rows, spatial), device="cuda", dtype=torch.float32)
            eager_output = torch.empty((rows,), device="cuda", dtype=torch.float32)

            def eager_launch() -> None:
                torch.mean(x, dim=1, out=eager_output)

            measure(run, rows, spatial, "PyTorch mean(out=)", eager_launch, x, eager_output)

            graph_output = torch.empty((rows,), device="cuda", dtype=torch.float32)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                torch.mean(x, dim=1, out=graph_output)

            measure(run, rows, spatial, "PyTorch CUDA graph mean(out=)", graph.replay, x, graph_output)


if __name__ == "__main__":
    main()
