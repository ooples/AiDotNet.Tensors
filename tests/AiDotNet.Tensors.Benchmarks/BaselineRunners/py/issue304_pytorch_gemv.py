#!/usr/bin/env python3
"""Raw PyTorch GEMV benchmark for AiDotNet.Tensors issue #304.

Measures [rows, cols] @ [cols, 1] across CPU and available GPU backends.
CUDA device-only timings use torch.cuda.Event. GPU readback timings use
synchronized wall-clock time and copy the full result to CPU.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Shape:
    rows: int
    cols: int

    @property
    def ops(self) -> int:
        return self.rows * self.cols

    def label(self) -> str:
        return f"[{self.rows},{self.cols}]x[{self.cols},1]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch GEMV benchmark for issue #304")
    parser.add_argument("--sizes", type=str, required=True, help="semicolon-separated rowsxcols list")
    parser.add_argument("--runs", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "gpu", "all"], default="all")
    return parser.parse_args()


def parse_shapes(value: str) -> list[Shape]:
    shapes: list[Shape] = []
    for token in value.split(";"):
        token = token.strip().lower()
        if not token:
            continue
        rows, cols = token.split("x")
        shapes.append(Shape(int(rows), int(cols)))
    return shapes


def inner_iterations(shape: Shape) -> int:
    if shape.ops <= 1_024:
        return 1_000
    if shape.ops <= 8_192:
        return 300
    if shape.ops <= 65_536:
        return 100
    if shape.ops <= 524_288:
        return 20
    return 1


def gpu_device_inner_iterations(shape: Shape) -> int:
    if shape.ops <= 1_024:
        return 200
    if shape.ops <= 8_192:
        return 100
    if shape.ops <= 65_536:
        return 50
    if shape.ops <= 524_288:
        return 10
    return 1


def gpu_readback_inner_iterations(shape: Shape) -> int:
    if shape.ops <= 1_024:
        return 20
    if shape.ops <= 8_192:
        return 10
    if shape.ops <= 65_536:
        return 5
    return 1


def deterministic_value(value: int) -> float:
    x = (value * 747796405 + 2891336453) & 0xFFFFFFFF
    x = (((x >> ((x >> 28) + 4)) ^ x) * 277803737) & 0xFFFFFFFF
    x = ((x >> 22) ^ x) & 0xFFFFFFFF
    return ((x & 0xFFFF) / 65535.0) - 0.5


def make_data(length: int, seed_offset: int) -> list[float]:
    return [deterministic_value(i + seed_offset) for i in range(length)]


def make_tensors(torch, shape: Shape, device):
    weights = torch.tensor(
        make_data(shape.rows * shape.cols, 304 + shape.rows + shape.cols),
        dtype=torch.float32,
        device=device,
    ).reshape(shape.rows, shape.cols)
    queries = [
        torch.tensor(
            make_data(shape.cols, 10_000 + shape.rows + i * 97),
            dtype=torch.float32,
            device=device,
        ).reshape(shape.cols, 1)
        for i in range(8)
    ]
    return weights, queries


def select_gpu_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", torch.cuda.get_device_name(0)

    try:
        import torch_directml  # type: ignore
    except Exception:
        return None, None, None

    return torch_directml.device(), "directml", "DirectML"


def synchronize(torch, backend: str | None, tensor=None) -> None:
    if backend == "cuda":
        torch.cuda.synchronize(tensor.device if tensor is not None else None)
    elif backend == "directml" and tensor is not None:
        _ = tensor.reshape(-1)[0].detach().cpu().item()


def summarize(times: list[float]) -> tuple[float, float, float, float]:
    times = sorted(times)
    return statistics.fmean(times), statistics.median(times), times[0], times[-1]


def measure_wall(
    torch,
    backend: str | None,
    runs: int,
    warmup: int,
    inner: int,
    action: Callable[[int], object],
    sync_result: bool,
) -> list[float]:
    last = None
    for run in range(warmup):
        for step in range(inner):
            last = action(run + step)
    if sync_result:
        synchronize(torch, backend, last)

    times: list[float] = []
    for run in range(runs):
        start = time.perf_counter()
        for step in range(inner):
            last = action(run + step)
        if sync_result:
            synchronize(torch, backend, last)
        stop = time.perf_counter()
        times.append((stop - start) * 1000.0 / inner)
    return times


def measure_cuda_events(torch, runs: int, warmup: int, inner: int, action: Callable[[int], object]) -> list[float]:
    for run in range(warmup):
        for step in range(inner):
            action(run + step)
    torch.cuda.synchronize()

    times: list[float] = []
    for run in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for step in range(inner):
            action(run + step)
        stop.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(stop)) / inner)
    return times


def print_header() -> None:
    print(
        f"{'Shape':<18} {'Method':<38} {'Mean ms':>10} {'Median ms':>10} "
        f"{'Min ms':>10} {'Max ms':>10} {'GFLOP/s':>10}"
    )
    print("-" * 112)


def print_result(shape: Shape, method: str, times: list[float]) -> None:
    mean_ms, median_ms, min_ms, max_ms = summarize(times)
    gflops = (2.0 * shape.rows * shape.cols) / (mean_ms / 1000.0) / 1_000_000_000.0
    print(
        f"{shape.label():<18} {method:<38} {mean_ms:>10.5f} {median_ms:>10.5f} "
        f"{min_ms:>10.5f} {max_ms:>10.5f} {gflops:>10.2f}"
    )


def run_cpu(torch, shapes: list[Shape], runs: int, warmup: int) -> None:
    print("PyTorch CPU")
    print(f"torch threads: {torch.get_num_threads()} interop: {torch.get_num_interop_threads()}")
    print_header()
    for shape in shapes:
        weights, queries = make_tensors(torch, shape, torch.device("cpu"))
        inner = inner_iterations(shape)
        with torch.no_grad():
            times = measure_wall(
                torch,
                backend=None,
                runs=runs,
                warmup=warmup,
                inner=inner,
                action=lambda run: torch.matmul(weights, queries[run % len(queries)]),
                sync_result=False,
            )
        print_result(shape, "PyTorch CPU matmul", times)
    print()


def run_gpu(torch, shapes: list[Shape], runs: int, warmup: int) -> None:
    device, backend, name = select_gpu_device(torch)
    if device is None or backend is None:
        print("PyTorch GPU skipped: no CUDA or torch-directml backend available.")
        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print()
        return

    print("PyTorch GPU")
    print(f"torch: {torch.__version__}")
    print(f"backend: {backend}")
    print(f"device: {name}")
    print_header()

    for shape in shapes:
        weights, queries = make_tensors(torch, shape, device)
        inner = inner_iterations(shape)
        with torch.no_grad():
            if backend == "cuda":
                device_times = measure_cuda_events(
                    torch,
                    runs=runs,
                    warmup=warmup,
                    inner=inner,
                    action=lambda run: torch.matmul(weights, queries[run % len(queries)]),
                )
            else:
                device_times = measure_wall(
                    torch,
                    backend=backend,
                    runs=runs,
                    warmup=warmup,
                    inner=inner,
                    action=lambda run: torch.matmul(weights, queries[run % len(queries)]),
                    sync_result=True,
                )

            readback_times = measure_wall(
                torch,
                backend=backend,
                runs=runs,
                warmup=warmup,
                inner=gpu_readback_inner_iterations(shape),
                action=lambda run: torch.matmul(weights, queries[run % len(queries)]).detach().cpu(),
                sync_result=False,
            )

        print_result(shape, f"PyTorch GPU matmul sync ({backend})", device_times)
        print_result(shape, f"PyTorch GPU matmul+cpu ({backend})", readback_times)
        print()


def main() -> int:
    args = parse_args()
    shapes = parse_shapes(args.sizes)

    try:
        import torch
    except Exception as exc:
        print(f"PyTorch baseline failed: could not import torch: {exc}", file=sys.stderr)
        return 2

    torch.set_grad_enabled(False)
    if os.cpu_count():
        torch.set_num_threads(os.cpu_count())

    print("Raw PyTorch GEMV baseline")
    print(f"torch: {torch.__version__}")
    print()

    if args.device in ("cpu", "all"):
        run_cpu(torch, shapes, args.runs, args.warmup)
    if args.device in ("gpu", "all"):
        run_gpu(torch, shapes, args.runs, args.warmup)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
