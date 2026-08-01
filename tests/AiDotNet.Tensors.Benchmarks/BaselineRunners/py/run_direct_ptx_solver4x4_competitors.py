#!/usr/bin/env python3
"""Resident PyTorch/cuSOLVER competitors for issue #853 (not run by CI).

Required eager measurements run first in one uninterrupted resident process. Optional
CUDA-graph captures then run in disposable child processes, so an unsupported capture
cannot invalidate the eager context or make the next required competitor unavailable.
"""

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

import torch


MAX_WARMUPS = 30
MIN_WARMUPS = 3
FAST_SAMPLES = 101
SLOW_SAMPLES = 21
SLOW_OPERATION_US = 1_000.0
WARMUP_TARGET_US = 10_000.0
DEVICE_BATCH_TARGET_US = 1_000.0
MAX_DEVICE_LAUNCHES = 10
ALLOCATION_SAMPLES = 5
BATCHES = (1024, 4096, 16384, 65536)
OPERATIONS = (
    "cholesky", "lu-factor", "qr", "eigh", "eigh-lower", "svd", "lu-solve",
    "ldl-factor", "ldl-solve", "solve", "tri-lower", "tri-upper",
    "chol-backward", "solve-backward")


def environment():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    driver = torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None
    return {
        "gpu": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": driver,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {
        "mean_us": statistics.fmean(values),
        "median_us": percentile(values, 0.50),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
    }


def measurement_plan(calibration_us):
    if not math.isfinite(calibration_us) or calibration_us <= 0.0:
        raise ValueError("calibration_us must be finite and positive")
    warmups = max(MIN_WARMUPS, min(
        MAX_WARMUPS, math.ceil(WARMUP_TARGET_US / calibration_us)))
    launches = max(1, min(
        MAX_DEVICE_LAUNCHES, math.ceil(DEVICE_BATCH_TARGET_US / calibration_us)))
    samples = SLOW_SAMPLES if calibration_us >= SLOW_OPERATION_US else FAST_SAMPLES
    return warmups, launches, samples


def flops(operation, batch):
    return {
        "cholesky": batch * 64.0 / 3.0,
        "lu-factor": batch * 128.0 / 3.0,
        "qr": batch * 256.0 / 3.0,
        "eigh": batch * 8 * 6 * 96.0,
        "eigh-lower": batch * 8 * 6 * 96.0,
        "svd": batch * (10 * 6 * 96.0 + 256.0),
        "lu-solve": batch * 64.0,
        "ldl-factor": batch * 64.0 / 3.0,
        "ldl-solve": batch * 64.0,
        "solve": batch * 64.0,
        "tri-lower": batch * 32.0,
        "tri-upper": batch * 32.0,
        "chol-backward": batch * 384.0,
        "solve-backward": batch * 192.0,
    }[operation]


def bytes_moved(operation, batch):
    per_matrix = {
        "cholesky": 64 + 64 + 4,
        "lu-factor": 64 + 64 + 16,
        "qr": 64 + 64 + 64,
        "eigh": 64 + 16 + 64,
        "eigh-lower": 64 + 16 + 64,
        "svd": 64 + 64 + 16 + 64,
        "lu-solve": 64 + 16 + 16 + 16,
        "ldl-factor": 64 + 64 + 16,
        "ldl-solve": 64 + 16 + 16 + 16,
        "solve": 64 + 16 + 16 + 4,
        "tri-lower": 64 + 16 + 16,
        "tri-upper": 64 + 16 + 16,
        "chol-backward": 64 + 64 + 64,
        "solve-backward": 64 + 16 + 16 + 64 + 16,
    }[operation]
    return batch * per_matrix


def build(operation, batch):
    base = torch.tensor(
        [[9.0, 1.0, 2.0, 0.5], [1.0, 8.0, 0.25, 1.0],
         [2.0, 0.25, 7.0, 0.75], [0.5, 1.0, 0.75, 6.0]],
        device="cuda", dtype=torch.float32)
    identity = torch.eye(4, device="cuda", dtype=torch.float32)
    lower = torch.tensor(
        [[3., 0., 0., 0.], [1., 4., 0., 0.], [.5, 1., 5., 0.], [.25, .5, 1., 6.]],
        device="cuda")
    upper = lower.transpose(0, 1).contiguous()
    chol_factor = torch.diag(torch.tensor([3., 2.5, 2., 1.5], device="cuda"))
    selected = {
        "lu-solve": identity, "ldl-solve": identity, "solve-backward": identity,
        "tri-lower": lower, "tri-upper": upper, "chol-backward": chol_factor,
    }.get(operation, base)
    a = selected.expand(batch, -1, -1).clone()
    if operation == "cholesky":
        output = torch.empty_like(a)
        info = torch.empty(batch, device="cuda", dtype=torch.int32)
        return a, lambda: torch.linalg.cholesky_ex(a, check_errors=False, out=(output, info)), (output, info)
    if operation == "lu-factor":
        lu = torch.empty_like(a)
        pivots = torch.empty((batch, 4), device="cuda", dtype=torch.int32)
        info = torch.empty(batch, device="cuda", dtype=torch.int32)
        return a, lambda: torch.linalg.lu_factor_ex(a, check_errors=False, out=(lu, pivots, info)), (lu, pivots, info)
    if operation == "qr":
        q, r = torch.empty_like(a), torch.empty_like(a)
        return a, lambda: torch.linalg.qr(a, mode="reduced", out=(q, r)), (q, r)
    if operation == "eigh":
        w = torch.empty((batch, 4), device="cuda", dtype=torch.float32)
        v = torch.empty_like(a)
        return a, lambda: torch.linalg.eigh(a, UPLO="U", out=(w, v)), (w, v)
    if operation == "eigh-lower":
        w = torch.empty((batch, 4), device="cuda", dtype=torch.float32)
        v = torch.empty_like(a)
        return a, lambda: torch.linalg.eigh(a, UPLO="L", out=(w, v)), (w, v)
    if operation == "svd":
        u, vh = torch.empty_like(a), torch.empty_like(a)
        s = torch.empty((batch, 4), device="cuda", dtype=torch.float32)
        return a, lambda: torch.linalg.svd(a, full_matrices=False, out=(u, s, vh)), (u, s, vh)
    rhs_vector = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda").expand(batch, -1).clone()
    if operation == "lu-solve":
        lu, pivots = torch.linalg.lu_factor(a)
        rhs = rhs_vector.unsqueeze(-1)
        solution = torch.empty_like(rhs)
        return a, lambda: torch.linalg.lu_solve(lu, pivots, rhs, out=solution), (solution, rhs)
    if operation == "ldl-factor":
        ld = torch.empty_like(a)
        pivots = torch.empty((batch, 4), device="cuda", dtype=torch.int32)
        info = torch.empty(batch, device="cuda", dtype=torch.int32)
        action = lambda: torch.linalg.ldl_factor_ex(a, hermitian=False, check_errors=False, out=(ld, pivots, info))
        return a, action, (ld, pivots, info)
    if operation == "ldl-solve":
        ld, pivots = torch.linalg.ldl_factor(a)
        rhs = rhs_vector.unsqueeze(-1)
        solution = torch.empty_like(rhs)
        return a, lambda: torch.linalg.ldl_solve(ld, pivots, rhs, hermitian=False, out=solution), (solution, rhs)
    if operation == "solve":
        solution = torch.empty_like(rhs_vector)
        info = torch.empty(batch, device="cuda", dtype=torch.int32)
        return a, lambda: torch.linalg.solve_ex(a, rhs_vector, check_errors=False, out=(solution, info)), (solution, rhs_vector, info)
    if operation in ("tri-lower", "tri-upper"):
        rhs = rhs_vector.unsqueeze(-1)
        solution = torch.empty_like(rhs)
        is_upper = operation == "tri-upper"
        action = lambda: torch.linalg.solve_triangular(
            a, rhs, upper=is_upper, unitriangular=False, out=solution)
        return a, action, (solution, rhs)
    if operation == "chol-backward":
        source = (a @ a.transpose(-2, -1)).detach().requires_grad_(True)
        factor = torch.linalg.cholesky(source)
        grad = torch.eye(4, device="cuda").expand(batch, -1, -1).clone()
        result = torch.empty_like(source)
        def chol_backward():
            value, = torch.autograd.grad(factor, source, grad_outputs=grad, retain_graph=True)
            result.copy_(value)
        return a, chol_backward, (result,)
    source = a.detach().requires_grad_(True)
    rhs = rhs_vector.detach().requires_grad_(True)
    solution = torch.linalg.solve(source, rhs)
    grad = torch.tensor([.5, 1., 1.5, 2.], device="cuda").expand(batch, -1).clone()
    grad_a, grad_b = torch.empty_like(source), torch.empty_like(rhs)
    def solve_backward():
        value_a, value_b = torch.autograd.grad(
            solution, (source, rhs), grad_outputs=grad, retain_graph=True)
        grad_a.copy_(value_a)
        grad_b.copy_(value_b)
    return a, solve_backward, (grad_a, grad_b, solution, grad)


def residual(operation, a, outputs):
    if operation == "cholesky":
        return (outputs[0] @ outputs[0].transpose(-2, -1) - a).abs().max().item()
    if operation == "lu-factor":
        lu, pivots, info = outputs
        if not (info == 0).all().item():
            return math.inf
        p, lower, upper = torch.lu_unpack(lu, pivots)
        return (p @ lower @ upper - a).abs().max().item()
    if operation == "qr":
        return (outputs[0] @ outputs[1] - a).abs().max().item()
    if operation in ("eigh", "eigh-lower"):
        w, v = outputs
        return (a @ v - v * w.unsqueeze(-2)).abs().max().item()
    if operation == "svd":
        u, s, vh = outputs
        return (u @ torch.diag_embed(s) @ vh - a).abs().max().item()
    if operation == "ldl-factor":
        ld, pivots, info = outputs
        if not (info == 0).all().item():
            return math.inf
        recovered_identity = torch.linalg.ldl_solve(ld, pivots, a, hermitian=False)
        identity = torch.eye(4, device="cuda").expand_as(a)
        return (recovered_identity - identity).abs().max().item()
    if operation == "chol-backward":
        expected = torch.diag_embed(.5 / torch.diagonal(a, dim1=-2, dim2=-1))
        return (outputs[0] - expected).abs().max().item()
    if operation == "solve-backward":
        grad_a, grad_b, solution, grad = outputs
        expected_a = -grad.unsqueeze(-1) * solution.unsqueeze(-2)
        return max((grad_a - expected_a).abs().max().item(), (grad_b - grad).abs().max().item())
    solution, rhs = outputs[:2]
    return (a @ solution.unsqueeze(-1) - rhs.unsqueeze(-1)).abs().max().item() if solution.ndim == 2 else (a @ solution - rhs).abs().max().item()


def measure(action):
    # One untimed call absorbs lazy library initialization. A CUDA-event calibration
    # then bounds both warmup work and launches per timed sample. Fixed 10x batching
    # made a 0.70-second QR call execute 1,212 times per row and turned the three-process
    # release gate into a multi-hour job without adding useful evidence.
    action()
    torch.cuda.synchronize()
    calibration_start = torch.cuda.Event(enable_timing=True)
    calibration_stop = torch.cuda.Event(enable_timing=True)
    calibration_start.record()
    action()
    calibration_stop.record()
    torch.cuda.synchronize()
    calibration_us = max(calibration_start.elapsed_time(calibration_stop) * 1000.0, 0.001)

    warmups, launches, samples = measurement_plan(calibration_us)

    for _ in range(warmups):
        action()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for index in range(samples):
        starts[index].record()
        for _ in range(launches):
            action()
        stops[index].record()
    torch.cuda.synchronize()
    device = [
        start.elapsed_time(stop) * 1000.0 / launches
        for start, stop in zip(starts, stops)
    ]
    e2e = []
    for _ in range(samples):
        begin = time.perf_counter_ns()
        action()
        torch.cuda.synchronize()
        e2e.append((time.perf_counter_ns() - begin) / 1000.0)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    for _ in range(ALLOCATION_SAMPLES):
        action()
    torch.cuda.synchronize()
    temporary = max(0, torch.cuda.max_memory_allocated() - before)
    return (
        distribution(device), distribution(e2e), temporary,
        samples, launches, calibration_us,
    )


def emit(
        run, operation, batch, method, device, e2e, temporary, error,
        samples, launches, calibration_us):
    record = {
        "status": "ok", "run": run, "operation": operation, "batch": batch, "method": method,
        "device_mean_us": device["mean_us"], "device_median_us": device["median_us"],
        "device_p95_us": device["p95_us"], "device_p99_us": device["p99_us"],
        "e2e_mean_us": e2e["mean_us"], "e2e_median_us": e2e["median_us"],
        "e2e_p95_us": e2e["p95_us"], "e2e_p99_us": e2e["p99_us"],
        "gflops": flops(operation, batch) / (device["median_us"] * 1e-6) / 1e9,
        "gb_per_second": bytes_moved(operation, batch) / (device["median_us"] * 1e-6) / 1e9,
        "managed_bytes": 0, "temporary_device_bytes": temporary, "max_error": error,
        "samples": samples, "device_launches_per_sample": launches,
        "calibration_us": calibration_us,
        **environment(),
    }
    print(json.dumps(record, separators=(",", ":")), flush=True)


def emit_unavailable(run, operation, batch, method, error):
    print(json.dumps({
        "status": "unavailable", "run": run, "operation": operation,
        "batch": batch, "method": method, "reason": str(error),
        **environment(),
    }, separators=(",", ":")), flush=True)


def run_isolated_graph_cell(run, operation, batch):
    """Runs one optional graph competitor in a disposable CUDA process."""
    command = [
        sys.executable, "-u", os.path.abspath(__file__),
        "--runs", "1", "--operation", operation, "--batch", str(batch),
        "--isolated-graph-cell", "--run-number", str(run),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated graph competitor %s/B=%d failed with exit code %d" %
            (operation, batch, completed.returncode))


def measure_isolated_graph_cell(run, operation, batch):
    """Builds, captures, and measures one graph method in the disposable child."""
    graph_method = "PyTorch CUDA graph/cuSOLVER"
    try:
        a, eager, outputs = build(operation, batch)
        eager()
        torch.cuda.synchronize()
        error = residual(operation, a, outputs)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                eager()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            eager()
        replay = graph.replay
        evidence = measure(replay)
        emit(
            run, operation, batch, graph_method,
            *evidence[:3], error, *evidence[3:])
    except Exception as capture_error:
        # This process ends after the row, taking any invalid capture state with it.
        emit_unavailable(run, operation, batch, graph_method, capture_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--operation", action="append", choices=OPERATIONS,
        help="Run only this operation; repeat to select multiple operations.")
    parser.add_argument(
        "--batch", action="append", type=int, choices=BATCHES,
        help="Run only this batch size; repeat to select multiple batch sizes.")
    parser.add_argument("--isolated-graph-cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-number", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    operations = tuple(args.operation) if args.operation else OPERATIONS
    batches = tuple(args.batch) if args.batch else BATCHES
    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.isolated_graph_cell:
        if (len(operations) != 1 or len(batches) != 1 or args.runs != 1 or
                args.run_number is None or args.run_number <= 0):
            raise SystemExit("isolated graph mode requires one operation, one batch, "
                             "--runs 1, and a positive --run-number")
        measure_isolated_graph_cell(args.run_number, operations[0], batches[0])
        return

    graph_cells = []
    for run in range(1, args.runs + 1):
        for operation in operations:
            for batch in batches:
                eager_method = "PyTorch CUDA eager/cuSOLVER"
                try:
                    a, eager, outputs = build(operation, batch)
                    eager()
                    torch.cuda.synchronize()
                    error = residual(operation, a, outputs)
                    evidence = measure(eager)
                    emit(
                        run, operation, batch, eager_method,
                        *evidence[:3], error, *evidence[3:])
                except Exception as eager_error:
                    emit_unavailable(run, operation, batch, eager_method, eager_error)
                    graph_cells.append((run, operation, batch, eager_error))
                    continue

                # Release shape-local storage before the graph child allocates its copy.
                # The parent keeps only its warmed CUDA context and scalar evidence.
                del outputs
                del eager
                del a
                gc.collect()
                torch.cuda.empty_cache()
                graph_cells.append((run, operation, batch, None))

    graph_method = "PyTorch CUDA graph/cuSOLVER"
    for run, operation, batch, eager_error in graph_cells:
        if eager_error is not None:
            emit_unavailable(run, operation, batch, graph_method, eager_error)
            continue
        run_isolated_graph_cell(run, operation, batch)


if __name__ == "__main__":
    main()
