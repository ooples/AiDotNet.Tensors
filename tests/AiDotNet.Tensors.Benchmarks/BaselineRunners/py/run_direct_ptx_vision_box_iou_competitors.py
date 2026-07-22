#!/usr/bin/env python3
"""Resident NVIDIA PyTorch/torchvision competitors for issue #851."""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time

import torch
import torchvision
from torchvision.ops import box_iou as torchvision_box_iou

WARMUPS = 30
SAMPLES = 101
LAUNCHES = 25
SHAPES = (("n256-m256", 256, 256), ("n1024-m256", 1024, 256),
          ("n1024-m1024", 1024, 1024), ("n4096-m256", 4096, 256))


def require_no_foreign_compute(label):
    monitor = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
                             check=True, capture_output=True, text=True, timeout=5)
    conflicts = []
    for line in monitor.stdout.splitlines():
        cells = line.split()
        if not cells or cells[0].startswith("#") or len(cells) < 9:
            continue
        try:
            pid = int(cells[1])
        except ValueError:
            continue
        try:
            sm = int(cells[3])
        except ValueError:
            sm = 0
        process_type = cells[2].upper()
        if pid != os.getpid() and (process_type == "C" or
                                   ("C" in process_type and sm > 5)):
            conflicts.append(f"pid={pid} {cells[-1]} type={process_type} sm={sm}%")
    if conflicts:
        raise RuntimeError(f"[{label}] foreign GPU workload: " + "; ".join(conflicts))


def box_iou_torch(a, b):
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) *
              (a[:, 3] - a[:, 1]).clamp(min=0))
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) *
              (b[:, 3] - b[:, 1]).clamp(min=0))
    left_top = torch.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "p95": percentile(values, .95), "p99": percentile(values, .99)
    }


def measure_device(fn):
    for _ in range(WARMUPS):
        fn()
    torch.cuda.synchronize()
    values = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(SAMPLES):
        start.record()
        for _ in range(LAUNCHES):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / LAUNCHES)
    return distribution(values)


def measure_e2e(fn):
    for _ in range(WARMUPS):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1000.0)
    return distribution(values)


def run_cell(run, shape_name, n, m, method, fn, reference):
    torch.cuda.reset_peak_memory_stats()
    actual = fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    device = measure_device(fn)
    e2e = measure_e2e(fn)
    peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    error = (actual - reference).abs().max().item()
    cells = n * m
    print(json.dumps({
        "status": "ok", "run": run, "shape": shape_name, "method": method,
        "device_mean_us": device["mean"], "device_median_us": device["median"],
        "device_p95_us": device["p95"], "device_p99_us": device["p99"],
        "e2e_mean_us": e2e["mean"], "e2e_median_us": e2e["median"],
        "e2e_p95_us": e2e["p95"], "e2e_p99_us": e2e["p99"],
        "gflops": cells * 20.0 / (device["median"] * 1000.0),
        "algorithmic_gbps": cells * 36.0 / (device["median"] * 1000.0),
        "managed_bytes": -1, "temporary_device_bytes": peak,
        "max_error": error, "registers_per_thread": -1,
        "static_shared_bytes": -1, "local_bytes_per_thread": -1,
        "active_blocks_per_sm": -1
    }, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    driver_getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    print(json.dumps({
        "status": "environment", "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "cuda_driver": driver_getter() if driver_getter is not None else None,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(getattr(properties, "uuid", "unknown")),
        "compute_capability": [properties.major, properties.minor],
        "total_device_memory": properties.total_memory,
        "python": platform.python_version(), "os": platform.platform(),
        "warmups": WARMUPS,
        "samples": SAMPLES, "launches_per_device_sample": LAUNCHES
    }, separators=(",", ":")))
    for run in range(1, args.runs + 1):
        require_no_foreign_compute(f"vision-box-iou-run-{run}-start")
        torch.manual_seed(20260722 + run * 1000)
        for shape_name, n, m in SHAPES:
            xy_a = torch.rand((n, 2), device="cuda", dtype=torch.float32) * 512
            xy_b = torch.rand((m, 2), device="cuda", dtype=torch.float32) * 512
            a = torch.cat((xy_a, xy_a + torch.rand_like(xy_a) * 128), dim=1)
            b = torch.cat((xy_b, xy_b + torch.rand_like(xy_b) * 128), dim=1)
            reference = box_iou_torch(a.double(), b.double()).float()
            run_cell(run, shape_name, n, m, "torchvision.ops.box_iou",
                     lambda: torchvision_box_iou(a, b), reference)
            graph = torch.cuda.CUDAGraph()
            torchvision_box_iou(a, b)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                graph_output = torchvision_box_iou(a, b)

            def graph_launch():
                graph.replay()
                return graph_output

            run_cell(run, shape_name, n, m, "torchvision BoxIoU CUDA graph",
                     graph_launch, reference)
            compiled = torch.compile(box_iou_torch, mode="max-autotune", fullgraph=True)
            compiled(a, b)
            torch.cuda.synchronize()
            run_cell(run, shape_name, n, m, "PyTorch compile max-autotune",
                     lambda: compiled(a, b), reference)
        torch.cuda.synchronize()
        require_no_foreign_compute(f"vision-box-iou-run-{run}-end")


if __name__ == "__main__":
    main()
