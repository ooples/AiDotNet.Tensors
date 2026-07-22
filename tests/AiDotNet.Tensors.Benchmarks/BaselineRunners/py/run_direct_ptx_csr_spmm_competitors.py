"""Resident PyTorch CUDA sparse-CSR competitors for issue #852.

Hardware execution is intentionally opt-in. The C# and Python harnesses use the
same deterministic CSR matrix, dense matrix, warmup/sample counts, and FLOP model.
"""

import json
import math
import os
import platform
import statistics
import sys
import time

import torch

ROWS = 1024
INNER = 1024
COLS = 64
NNZ = 16384
WARMUPS = 30
SAMPLES = 101
LAUNCHES_PER_SAMPLE = 20


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def summarize(values):
    return {
        "mean_us": statistics.mean(values),
        "median_us": percentile(values, 0.50),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
    }


def inputs(device):
    rows = torch.arange(ROWS + 1, dtype=torch.int32, device=device) * 16
    host_columns = []
    host_values = []
    for row in range(ROWS):
        for item in range(16):
            host_columns.append((row * 17 + item * 13) & 1023)
            host_values.append((item - 7.5) / 32.0)
    columns = torch.tensor(host_columns, dtype=torch.int32, device=device)
    values = torch.tensor(host_values, dtype=torch.float32, device=device)
    dense = torch.tensor(
        [((index * 19) % 101 - 50) / 128.0 for index in range(INNER * COLS)],
        dtype=torch.float32,
        device=device,
    ).reshape(INNER, COLS)
    sparse = torch.sparse_csr_tensor(rows, columns, values, (ROWS, INNER), device=device)
    reference = torch.sparse.mm(sparse.double(), dense.double())
    return sparse, dense, reference


def measure(run, method, launch, reference):
    for _ in range(WARMUPS):
        output = launch()
    torch.cuda.synchronize()
    device_samples = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(SAMPLES)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(SAMPLES)]
    for index in range(SAMPLES):
        starts[index].record()
        for _ in range(LAUNCHES_PER_SAMPLE):
            output = launch()
        stops[index].record()
        stops[index].synchronize()
        device_samples.append(starts[index].elapsed_time(stops[index]) * 1000.0 / LAUNCHES_PER_SAMPLE)
    e2e_samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        output = launch()
        torch.cuda.synchronize()
        e2e_samples.append((time.perf_counter_ns() - started) / 1000.0)
    distribution = summarize(device_samples)
    e2e = summarize(e2e_samples)
    maximum_error = (output.double() - reference).abs().max().item()
    managed_before = torch.cuda.memory_allocated()
    for _ in range(SAMPLES):
        output = launch()
    torch.cuda.synchronize()
    temporary = max(0, torch.cuda.memory_allocated() - managed_before)
    median = distribution["median_us"]
    record = {
        "status": "hardware-unverified",
        "run": run,
        "shape": "m1024-k1024-n64-nnz16384",
        "method": method,
        "device_mean_us": distribution["mean_us"],
        "device_median_us": median,
        "device_p95_us": distribution["p95_us"],
        "device_p99_us": distribution["p99_us"],
        "end_to_end_mean_us": e2e["mean_us"],
        "end_to_end_median_us": e2e["median_us"],
        "end_to_end_p95_us": e2e["p95_us"],
        "end_to_end_p99_us": e2e["p99_us"],
        "gflops": (2.0 * NNZ * COLS) / (median * 1000.0),
        "managed_bytes_per_call": 0,
        "temporary_device_bytes": temporary,
        "maximum_error": maximum_error,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "sm": torch.cuda.get_device_capability(0),
            "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else "unknown",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    print(json.dumps(record, sort_keys=True))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and NVIDIA GPU are required.")
    runs = int(os.environ.get("AIDOTNET_PTX_INDEPENDENT_RUNS", "3"))
    for run in range(1, runs + 1):
        sparse, dense, reference = inputs("cuda")
        measure(run, "PyTorch sparse.mm eager", lambda: torch.sparse.mm(sparse, dense), reference)
        try:
            captured_output = torch.empty((ROWS, COLS), dtype=torch.float32, device="cuda")
            graph = torch.cuda.CUDAGraph()
            for _ in range(WARMUPS):
                captured_output.copy_(torch.sparse.mm(sparse, dense))
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                captured_output.copy_(torch.sparse.mm(sparse, dense))

            def graph_launch():
                graph.replay()
                return captured_output

            measure(run, "PyTorch sparse.mm CUDA graph", graph_launch, reference)
        except Exception as error:
            print(json.dumps({
                "status": "unsupported",
                "run": run,
                "shape": "m1024-k1024-n64-nnz16384",
                "method": "PyTorch sparse.mm CUDA graph",
                "reason": f"{type(error).__name__}: {error}",
            }, sort_keys=True))


if __name__ == "__main__":
    main()
