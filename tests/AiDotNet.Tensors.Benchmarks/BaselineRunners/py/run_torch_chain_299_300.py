#!/usr/bin/env python3
"""Raw PyTorch baseline for issues #299/#300.

Measures the same two-stage Linear -> ReLU -> Linear shape used by the
compiled-plan chain benchmark: batch x 256 -> batch x 256 -> batch x 10.

Examples:
    python run_torch_chain_299_300.py --device cpu
    python run_torch_chain_299_300.py --device cuda --batches 1 32 128
"""

import argparse
import math
import statistics
import sys
import time

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.stderr.write("torch not installed\n")
    sys.exit(2)


def _percentile(sorted_values, q):
    index = max(0, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[index]


def _make_data(batch_size, device):
    generator = torch.Generator(device="cpu").manual_seed(299300 + batch_size)
    x = torch.randn(batch_size, 256, generator=generator, dtype=torch.float32, device="cpu").to(device)
    w1 = torch.randn(256, 256, generator=generator, dtype=torch.float32, device="cpu").to(device)
    w2 = torch.randn(256, 10, generator=generator, dtype=torch.float32, device="cpu").to(device)
    b1 = torch.randn(256, generator=generator, dtype=torch.float32, device="cpu").to(device)
    b2 = torch.randn(10, generator=generator, dtype=torch.float32, device="cpu").to(device)
    return x, w1, w2, b1, b2


def _make_module(w1, w2, b1, b2):
    module = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(w1.device)
    with torch.no_grad():
        module[0].weight.copy_(w1.t().contiguous())
        module[0].bias.copy_(b1)
        module[2].weight.copy_(w2.t().contiguous())
        module[2].bias.copy_(b2)
    module.eval()
    return module


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_cpu(fn, warmup, iters):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            stop = time.perf_counter()
            times.append((stop - start) * 1_000_000.0)
    return times


def _time_cuda(fn, warmup, iters, device):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        _synchronize(device)
        times = []
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start_event.record()
            fn()
            stop_event.record()
            stop_event.synchronize()
            times.append(start_event.elapsed_time(stop_event) * 1000.0)
    return times


def _summarize(times):
    ordered = sorted(times)
    return statistics.median(ordered), _percentile(ordered, 0.9), min(ordered), max(ordered)


def _run_one(batch_size, device, warmup, iters, include_compile):
    x, w1, w2, b1, b2 = _make_data(batch_size, device)
    w1_t = w1.t().contiguous()
    w2_t = w2.t().contiguous()
    module = _make_module(w1, w2, b1, b2)

    def functional_two_stage():
        return F.linear(F.relu(F.linear(x, w1_t, b1)), w2_t, b2)

    def sequential_module():
        return module(x)

    variants = [
        ("torch_functional_two_stage", functional_two_stage),
        ("torch_nn_sequential", sequential_module),
    ]

    if include_compile and hasattr(torch, "compile"):
        try:
            compiled = torch.compile(module, fullgraph=True)
            variants.append(("torch_compile_sequential", lambda: compiled(x)))
        except Exception as exc:  # pragma: no cover - depends on torch install
            sys.stderr.write(f"torch.compile unavailable for batch {batch_size}: {exc}\n")

    timer = _time_cuda if device.type == "cuda" else _time_cpu
    for name, fn in variants:
        times = timer(fn, warmup, iters, device) if device.type == "cuda" else timer(fn, warmup, iters)
        median, p90, best, worst = _summarize(times)
        print(f"{name},{device.type},{batch_size},{median:.3f},{p90:.3f},{best:.3f},{worst:.3f},{iters}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compile", action="store_true", help="also time torch.compile when available")
    args = parser.parse_args()

    if args.iters <= 0:
        sys.stderr.write("--iters must be > 0 to compute median/p90\n")
        sys.exit(2)
    if args.warmup < 0:
        sys.stderr.write("--warmup must be >= 0\n")
        sys.exit(2)
    for b in args.batches:
        if b <= 0:
            sys.stderr.write(f"--batches values must be > 0 (got {b})\n")
            sys.exit(2)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("CUDA requested but torch.cuda.is_available() is false\n")
        sys.exit(3)

    print(f"torch={torch.__version__}")
    print(f"device={device.type}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print("method,device,batch_size,median_us,p90_us,min_us,max_us,iters")
    for batch_size in args.batches:
        _run_one(batch_size, device, args.warmup, args.iters, args.compile)


if __name__ == "__main__":
    main()
