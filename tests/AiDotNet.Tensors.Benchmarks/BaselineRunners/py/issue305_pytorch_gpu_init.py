#!/usr/bin/env python3
"""PyTorch GPU random-initialization benchmark for AiDotNet.Tensors issue #305.

Backend selection:
- CUDA is used when torch.cuda is available.
- DirectML is used when torch-directml is installed, which is the expected
  PyTorch GPU route for AMD GPUs on Windows.

The benchmark measures in-place uniform_ and normal_ initialization on an
already-allocated tensor. CUDA timings use torch.cuda.Event. DirectML does not
expose an event/synchronize API, so the script uses a one-element CPU read to
force completion and reports that timing as synchronized wall-clock time.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch GPU init benchmark for issue #305")
    parser.add_argument("--elements", nargs="+", type=int, default=[1_000_000, 16_777_216])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min", type=float, default=-0.05)
    parser.add_argument("--max", type=float, default=0.05)
    parser.add_argument("--stddev", type=float, default=0.02)
    parser.add_argument("--device", choices=["auto", "cuda", "directml"], default="auto")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be > 0 (need at least one timed sample for median)")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.stddev < 0:
        parser.error("--stddev must be >= 0")
    if args.max < args.min:
        parser.error(f"--max ({args.max}) must be >= --min ({args.min})")
    for e in args.elements:
        if e <= 0:
            parser.error(f"--elements values must be > 0 (got {e})")

    return args


def select_device(requested: str):
    import torch

    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    if requested in ("auto", "directml"):
        try:
            import torch_directml  # type: ignore
        except Exception as exc:
            if requested == "directml":
                raise RuntimeError("torch-directml is not installed or failed to import") from exc
        else:
            return torch_directml.device(), "directml"

    raise RuntimeError(
        "No PyTorch GPU backend is available. Install a CUDA build of torch for NVIDIA, "
        "or install torch-directml for AMD/Intel GPUs on Windows."
    )


def synchronize(backend: str, tensor) -> None:
    if backend == "cuda":
        import torch

        torch.cuda.synchronize(tensor.device)
        return

    if backend == "directml":
        # torch-directml currently has no public synchronize/event timer.
        # A scalar read forces queued work to complete and keeps transfer size tiny.
        _ = tensor[0].detach().cpu().item()
        return

    raise RuntimeError(f"Unknown backend: {backend}")


def measure_cuda(tensor, runs: int, action: Callable[[int], None]) -> list[float]:
    import torch

    timings: list[float] = []
    for run in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        action(run)
        stop.record()
        torch.cuda.synchronize(tensor.device)
        timings.append(float(start.elapsed_time(stop)))
    return timings


def measure_wall_clock(backend: str, tensor, runs: int, action: Callable[[int], None]) -> list[float]:
    timings: list[float] = []
    for run in range(runs):
        start = time.perf_counter()
        action(run)
        synchronize(backend, tensor)
        stop = time.perf_counter()
        timings.append((stop - start) * 1000.0)
    return timings


def summarize(method: str, elements: int, timings: list[float]) -> None:
    timings = sorted(timings)
    mean_ms = statistics.fmean(timings)
    median_ms = statistics.median(timings)
    gbps = elements * 4 / (mean_ms / 1000.0) / 1_000_000_000.0
    print(
        f"{method:<32} {elements:>12,d} {mean_ms:>10.4f} "
        f"{median_ms:>10.4f} {timings[0]:>10.4f} {timings[-1]:>10.4f} {gbps:>10.2f}"
    )


def main() -> int:
    args = parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"FAILED: could not import torch: {exc}", file=sys.stderr)
        return 2

    try:
        device, backend = select_device(args.device)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        print(f"torch version: {torch.__version__}", file=sys.stderr)
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}", file=sys.stderr)
        return 3

    torch.set_grad_enabled(False)
    print("Issue #305 PyTorch GPU initialization benchmark")
    print(f"torch:   {torch.__version__}")
    print(f"backend: {backend}")
    print(f"device:  {device}")
    if backend == "cuda":
        print(f"name:    {torch.cuda.get_device_name(device)}")
    print("Timing is synchronized; DirectML includes a one-element readback sync.")
    print()
    print(f"{'Method':<32} {'Elements':>12} {'Mean ms':>10} {'Median ms':>10} {'Min ms':>10} {'Max ms':>10} {'GB/s':>10}")
    print("-" * 98)

    for elements in args.elements:
        tensor = torch.empty((elements,), dtype=torch.float32, device=device)

        for _ in range(args.warmup):
            tensor.uniform_(args.min, args.max)
            tensor.normal_(0.0, args.stddev)
        synchronize(backend, tensor)

        uniform_action = lambda _run: tensor.uniform_(args.min, args.max)
        normal_action = lambda _run: tensor.normal_(0.0, args.stddev)

        if backend == "cuda":
            uniform_timings = measure_cuda(tensor, args.runs, uniform_action)
            normal_timings = measure_cuda(tensor, args.runs, normal_action)
        else:
            uniform_timings = measure_wall_clock(backend, tensor, args.runs, uniform_action)
            normal_timings = measure_wall_clock(backend, tensor, args.runs, normal_action)

        summarize("PyTorch GPU uniform_", elements, uniform_timings)
        summarize("PyTorch GPU normal_", elements, normal_timings)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
