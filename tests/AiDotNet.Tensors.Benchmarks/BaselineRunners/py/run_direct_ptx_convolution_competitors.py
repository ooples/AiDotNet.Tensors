"""Resident PyTorch/cuDNN peers for the issue-#841 convolution evidence cell."""

import json
import math
import statistics
import torch
import torch.nn.functional as functional

WARMUPS = 30
SAMPLES = 101
N, C, H, W, K = 1, 64, 16, 16, 64
FLOPS = 2 * N * K * H * W * C


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure(name, launch):
    for _ in range(WARMUPS):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    median_us = statistics.median(samples)
    return {
        "method": name,
        "median_us": median_us,
        "p95_us": percentile(samples, 0.95),
        "p99_us": percentile(samples, 0.99),
        "mean_us": statistics.mean(samples),
        "gflops": FLOPS / median_us / 1000.0,
        "tflops": FLOPS / median_us / 1_000_000.0,
        "managed_bytes_per_call": None,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    torch.manual_seed(841)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    x = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32) * 0.25
    weight = torch.randn((K, C, 1, 1), device="cuda", dtype=torch.float32) * 0.25
    bias = torch.randn((K,), device="cuda", dtype=torch.float32) * 0.25

    def eager():
        return torch.relu(functional.conv2d(x, weight, bias))

    eager_result = measure("PyTorch cuDNN eager", eager)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = eager()
    torch.cuda.synchronize()
    eager_result["temporary_device_bytes"] = max(
        0, torch.cuda.max_memory_allocated() - baseline - result.numel() * result.element_size())

    static_output = torch.empty((N, K, H, W), device="cuda", dtype=torch.float32)
    graph = torch.cuda.CUDAGraph()
    for _ in range(3):
        static_output.copy_(eager())
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        static_output.copy_(eager())

    graph_result = measure("PyTorch cuDNN CUDA Graph", graph.replay)
    graph_result["temporary_device_bytes"] = 0

    fingerprint = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
    }
    print(json.dumps({"environment": fingerprint, "results": [eager_result, graph_result]}, indent=2))


if __name__ == "__main__":
    main()
