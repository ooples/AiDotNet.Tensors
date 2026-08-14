"""Strongest compiled PyTorch/cuDNN peers for the issue-#841 ResNet-class
convolution evidence matrix.

This is the competitor bar the direct-PTX ResNet-class specializations must beat
(>=1.10x median, p95 <=+10%) to be promoted. It measures cuDNN *at its best*:
per-shape algorithm autotuning (cudnn.benchmark=True), the autotuned kernel
captured in a CUDA graph to strip launch overhead (the strongest latency-bound
lane), and a torch.compile lane. Everything is FP32 NCHW, matching the
direct-PTX contracts; TF32 is left off so the comparison is apples-to-apples
against our fp32 accumulation.

Emits one JSON object per shape: {environment, shape, results:[...]}. The .NET
harness (DirectPtxConvolutionExperiment) parses these and applies the promotion
gate against the matching direct-PTX cell.
"""

import json
import statistics
import torch
import torch.nn.functional as functional

WARMUPS = 30
SAMPLES = 101

# ResNet-class workhorse shapes (N, C, H, W, K, kernel, stride, pad) where cuDNN
# heuristics (Winograd / implicit-precomp-GEMM / Tensor-Core) are strongest.
SHAPES = [
    # 3x3 s1 p1 stages
    ("resnet_3x3_c64_56", 32, 64, 56, 56, 64, 3, 1, 1),
    ("resnet_3x3_c128_28", 16, 128, 28, 28, 128, 3, 1, 1),
    ("resnet_3x3_c256_14", 8, 256, 14, 14, 256, 3, 1, 1),
    # 1x1 projections at the same stage resolutions
    ("resnet_1x1_c64_56", 32, 64, 56, 56, 64, 1, 1, 0),
    ("resnet_1x1_c128_28", 16, 128, 28, 28, 128, 1, 1, 0),
    ("resnet_1x1_c256_14", 8, 256, 14, 14, 256, 1, 1, 0),
]


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure(name, launch, flops):
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
        samples.append(start.elapsed_time(end) * 1000.0)  # us
    median_us = statistics.median(samples)
    return {
        "method": name,
        "median_us": median_us,
        "p95_us": percentile(samples, 0.95),
        "p99_us": percentile(samples, 0.99),
        "mean_us": statistics.mean(samples),
        "gflops": flops / median_us / 1000.0,
        "tflops": flops / median_us / 1_000_000.0,
    }


def bench_shape(scope, n, c, h, w, k, kernel, stride, pad):
    torch.manual_seed(841)
    # cuDNN at its best: autotune the algorithm for this exact shape.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.float32) * 0.25
    weight = torch.randn((k, c, kernel, kernel), device="cuda", dtype=torch.float32) * 0.25
    bias = torch.randn((k,), device="cuda", dtype=torch.float32) * 0.25
    out_h = (h + 2 * pad - kernel) // stride + 1
    out_w = (w + 2 * pad - kernel) // stride + 1
    flops = 2.0 * n * k * out_h * out_w * c * kernel * kernel

    def eager():
        return torch.relu(functional.conv2d(x, weight, bias, stride=stride, padding=pad))

    autotuned = measure("PyTorch cuDNN autotuned eager", eager, flops)

    # Autotuned kernel captured in a CUDA graph — strongest latency-bound lane.
    static_output = torch.empty((n, k, out_h, out_w), device="cuda", dtype=torch.float32)
    graph = torch.cuda.CUDAGraph()
    for _ in range(5):
        static_output.copy_(eager())
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        static_output.copy_(eager())
    graph_result = measure("PyTorch cuDNN autotuned CUDA Graph", graph.replay, flops)

    results = [autotuned, graph_result]

    # torch.compile lane (best-effort; some builds lack inductor CUDA).
    try:
        compiled = torch.compile(lambda t: torch.relu(functional.conv2d(t, weight, bias, stride=stride, padding=pad)))
        compiled(x)  # trigger compile
        torch.cuda.synchronize()
        results.append(measure("PyTorch compile", lambda: compiled(x), flops))
    except Exception as exc:  # pragma: no cover - environment dependent
        results.append({"method": "PyTorch compile", "error": str(exc)})

    fingerprint = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
    }
    return {
        "environment": fingerprint,
        "shape": {"scope": scope, "N": n, "C": c, "H": h, "W": w, "K": k,
                  "kernel": kernel, "stride": stride, "pad": pad,
                  "outH": out_h, "outW": out_w},
        "results": results,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    payload = [bench_shape(*shape) for shape in SHAPES]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
