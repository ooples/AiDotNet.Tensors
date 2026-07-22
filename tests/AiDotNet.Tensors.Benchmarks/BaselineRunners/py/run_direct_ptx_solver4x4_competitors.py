#!/usr/bin/env python3
"""Resident PyTorch/cuSOLVER competitors for issue #853 (not run by CI)."""

import argparse
import json
import math
import platform
import statistics
import time

import torch


WARMUPS = 30
SAMPLES = 101
LAUNCHES = 10
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
    for _ in range(WARMUPS):
        action()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(SAMPLES)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(SAMPLES)]
    for index in range(SAMPLES):
        starts[index].record()
        for _ in range(LAUNCHES):
            action()
        stops[index].record()
    torch.cuda.synchronize()
    device = [start.elapsed_time(stop) * 1000.0 / LAUNCHES for start, stop in zip(starts, stops)]
    e2e = []
    for _ in range(SAMPLES):
        begin = time.perf_counter_ns()
        action()
        torch.cuda.synchronize()
        e2e.append((time.perf_counter_ns() - begin) / 1000.0)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    for _ in range(SAMPLES):
        action()
    torch.cuda.synchronize()
    temporary = max(0, torch.cuda.max_memory_allocated() - before)
    return distribution(device), distribution(e2e), temporary


def emit(run, operation, batch, method, device, e2e, temporary, error):
    record = {
        "status": "ok", "run": run, "operation": operation, "batch": batch, "method": method,
        "device_mean_us": device["mean_us"], "device_median_us": device["median_us"],
        "device_p95_us": device["p95_us"], "device_p99_us": device["p99_us"],
        "e2e_mean_us": e2e["mean_us"], "e2e_median_us": e2e["median_us"],
        "e2e_p95_us": e2e["p95_us"], "e2e_p99_us": e2e["p99_us"],
        "gflops": flops(operation, batch) / (device["median_us"] * 1e-6) / 1e9,
        "gb_per_second": bytes_moved(operation, batch) / (device["median_us"] * 1e-6) / 1e9,
        "managed_bytes": 0, "temporary_device_bytes": temporary, "max_error": error,
        **environment(),
    }
    print(json.dumps(record, separators=(",", ":")), flush=True)


def emit_unavailable(run, operation, batch, method, error):
    print(json.dumps({
        "status": "unavailable", "run": run, "operation": operation,
        "batch": batch, "method": method, "reason": str(error),
        **environment(),
    }, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    for run in range(1, args.runs + 1):
        for operation in OPERATIONS:
            for batch in BATCHES:
                eager_method = "PyTorch CUDA eager/cuSOLVER"
                graph_method = "PyTorch CUDA graph/cuSOLVER"
                try:
                    a, eager, outputs = build(operation, batch)
                    eager()
                    torch.cuda.synchronize()
                    error = residual(operation, a, outputs)
                    device, e2e, temporary = measure(eager)
                    emit(run, operation, batch, eager_method, device, e2e, temporary, error)
                except Exception as eager_error:
                    emit_unavailable(run, operation, batch, eager_method, eager_error)
                    emit_unavailable(run, operation, batch, graph_method, eager_error)
                    continue

                try:
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
                    device, e2e, temporary = measure(replay)
                    emit(run, operation, batch, graph_method, device, e2e, temporary, error)
                except Exception as capture_error:
                    # Some cuSOLVER paths are not capture-safe on every installed
                    # CUDA/PyTorch pair. Preserve that result instead of aborting
                    # the remaining competitor cells.
                    emit_unavailable(run, operation, batch, graph_method, capture_error)


if __name__ == "__main__":
    main()
