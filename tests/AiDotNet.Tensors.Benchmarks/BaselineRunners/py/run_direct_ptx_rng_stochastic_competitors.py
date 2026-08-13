#!/usr/bin/env python3
"""Resident PyTorch CUDA peers for every benchmarked issue-#849 operation."""

import argparse
import json
import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F


WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 10
P = 0.1
KEEP = 1.0 - P
TEMPERATURE = 0.7
VECTOR_SHAPES = (4096, 65536, 1048576)
ROW_SHAPES = (128, 2048, 32768)
RAY_SHAPES = (64, 1024, 16384)
BIAS_SHAPES = (16, 256, 4096)


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    return (
        statistics.fmean(values),
        percentile(values, 0.50),
        percentile(values, 0.95),
        percentile(values, 0.99),
    )


def measure_device(operation):
    for _ in range(WARMUPS):
        operation()
    torch.cuda.synchronize()
    values = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(SAMPLES):
        start.record()
        for _ in range(DEVICE_LAUNCHES):
            operation()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / DEVICE_LAUNCHES)
    return summarize(values)


def measure_e2e(operation):
    for _ in range(WARMUPS):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1000.0)
    return summarize(values)


def emit(run, operation_name, shape, method, operation, useful_bytes, validator):
    probe = operation()
    torch.cuda.synchronize()
    max_error = float(validator(probe))
    del probe
    device = measure_device(operation)
    e2e = measure_e2e(operation)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = operation()
    torch.cuda.synchronize()
    peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    del result
    record = {
        "status": "ok",
        "run": run,
        "operation": operation_name,
        "shape": shape,
        "method": method,
        "device_mean_us": device[0],
        "device_median_us": device[1],
        "device_p95_us": device[2],
        "device_p99_us": device[3],
        "e2e_mean_us": e2e[0],
        "e2e_median_us": e2e[1],
        "e2e_p95_us": e2e[2],
        "e2e_p99_us": e2e[3],
        "gb_per_second": useful_bytes / (device[1] * 1.0e-6) / 1.0e9,
        "peak_device_bytes": peak,
        "max_error": max_error,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
    }
    print(json.dumps(record, separators=(",", ":")), flush=True)


def emit_skipped(run, operation_name, shape, method, error):
    print(json.dumps({
        "status": "skipped",
        "run": run,
        "operation": operation_name,
        "shape": shape,
        "method": method,
        "error": f"{type(error).__name__}: {error}",
    }, separators=(",", ":")), flush=True)


def capture(operation):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = operation()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = operation()

    def replay():
        graph.replay()
        return result

    return replay, graph


def variants(run, operation_name, shape, operation, useful_bytes, validator):
    emit(run, operation_name, shape, "PyTorch eager", operation, useful_bytes, validator)
    try:
        graph_operation, graph = capture(operation)
        emit(run, operation_name, shape, "PyTorch CUDA graph", graph_operation,
             useful_bytes, validator)
        del graph, graph_operation
    except Exception as error:
        emit_skipped(run, operation_name, shape, "PyTorch CUDA graph", error)
    try:
        compiled = torch.compile(operation, mode="max-autotune", fullgraph=True)
        for _ in range(3):
            compiled()
        torch.cuda.synchronize()
        emit(run, operation_name, shape, "PyTorch compile max-autotune", compiled,
             useful_bytes, validator)
        del compiled
    except Exception as error:
        emit_skipped(run, operation_name, shape, "PyTorch compile max-autotune", error)


def maximum_error(actual, expected):
    return (actual - expected).abs().max().item()


def validate_range(value, minimum, maximum):
    if not torch.isfinite(value).all():
        return math.inf
    return max(
        torch.clamp(minimum - value, min=0).max().item(),
        torch.clamp(value - maximum, min=0).max().item(),
    )


def run_vector(run, elements, device):
    shape = f"N={elements}"
    source = torch.linspace(-1.0, 1.0, elements, device=device)
    gradient = torch.linspace(-0.75, 0.75, elements, device=device)
    saved_mask = (torch.arange(elements, device=device) % 10 != 0).to(torch.float32) / KEEP
    epsilon = torch.linspace(0.5, -0.5, elements, device=device)

    variants(run, "uniform", shape,
             lambda: torch.rand_like(source).mul_(2.0).add_(-1.0),
             4 * elements, lambda value: validate_range(value, -1.0, 1.0))

    def normal():
        return torch.randn_like(source).mul_(1.5).add_(0.25)

    def normal_error(value):
        if not torch.isfinite(value).all():
            return math.inf
        return max(abs(value.mean().item() - 0.25),
                   abs(value.std(unbiased=False).item() - 1.5))

    variants(run, "normal", shape, normal, 4 * elements, normal_error)
    variants(run, "gaussian-noise", shape, normal, 4 * elements, normal_error)

    def mask():
        return (torch.rand_like(source) < KEEP).to(torch.float32).mul_(1.0 / KEEP)

    mask_validator = lambda value: torch.minimum(
        value.abs(), (value - (1.0 / KEEP)).abs()).max().item()
    variants(run, "dropout-mask", shape, mask, 4 * elements, mask_validator)
    variants(run, "stateless-dropout-mask", shape, mask, 4 * elements, mask_validator)

    def dropout_forward():
        return torch.ops.aten.native_dropout.default(source, P, True)

    def dropout_error(value):
        output, bool_mask = value
        expected = source * bool_mask.to(source.dtype) / KEEP
        return maximum_error(output, expected)

    variants(run, "dropout-forward", shape, dropout_forward,
             12 * elements, dropout_error)
    variants(run, "dropout-backward", shape,
             lambda: gradient * saved_mask, 12 * elements,
             lambda value: maximum_error(value, gradient * saved_mask))

    x_coefficient = math.sqrt(0.8 / 0.7)
    epsilon_coefficient = math.sqrt(0.2) - math.sqrt(0.3) * x_coefficient
    ddim_expected = source * x_coefficient + epsilon * epsilon_coefficient
    variants(run, "ddim-step", shape,
             lambda: source * x_coefficient + epsilon * epsilon_coefficient,
             12 * elements, lambda value: maximum_error(value, ddim_expected))

    lower = 0.125
    upper = 1.0 / 3.0
    rrelu_noise = torch.empty_like(source)

    def rrelu_training():
        output = torch.ops.aten.rrelu_with_noise.default(
            source, rrelu_noise, lower, upper, True, None)
        return output, rrelu_noise

    def rrelu_training_error(value):
        output, noise = value
        expected = torch.where(source >= 0, source, source * noise)
        negative_noise = noise[source < 0]
        return max(maximum_error(output, expected),
                   validate_range(negative_noise, lower, upper))

    variants(run, "rrelu-training", shape, rrelu_training,
             12 * elements, rrelu_training_error)
    fixed_noise = torch.linspace(lower, upper, elements, device=device)
    rrelu_expected = torch.where(source >= 0, source, source * fixed_noise)
    variants(run, "rrelu-saved-noise-forward", shape,
             lambda: torch.where(source >= 0, source, source * fixed_noise),
             12 * elements, lambda value: maximum_error(value, rrelu_expected))
    rrelu_backward_expected = gradient * torch.where(
        source >= 0, torch.ones_like(source), fixed_noise)
    variants(run, "rrelu-backward", shape,
             lambda: gradient * torch.where(
                 source >= 0, torch.ones_like(source), fixed_noise),
             16 * elements,
             lambda value: maximum_error(value, rrelu_backward_expected))


def run_rows(run, rows, device):
    classes = 32
    shape = f"[{rows},32]"
    logits = torch.linspace(-2.0, 2.0, rows * classes, device=device).reshape(rows, classes)
    probabilities = torch.full((rows, classes), 1.0 / classes, device=device)
    gradient = torch.linspace(-1.0, 1.0, rows * classes, device=device).reshape(rows, classes)
    soft = torch.full((rows, classes), 1.0 / classes, device=device)

    def simplex_error(value):
        return max((value.sum(-1) - 1.0).abs().max().item(),
                   torch.clamp(-value, min=0).max().item())

    variants(run, "gumbel-softmax", shape,
             lambda: F.gumbel_softmax(logits, tau=TEMPERATURE, hard=False, dim=-1),
             8 * rows * classes, simplex_error)

    def categorical():
        indices = torch.multinomial(probabilities, 1).squeeze(-1)
        return F.one_hot(indices, num_classes=classes).to(torch.float32)

    def one_hot_error(value):
        return max((value.sum(-1) - 1.0).abs().max().item(),
                   torch.minimum(value.abs(), (value - 1.0).abs()).max().item())

    variants(run, "categorical-one-hot", shape, categorical,
             8 * rows * classes, one_hot_error)
    backward_expected = soft * (
        gradient - (gradient * soft).sum(-1, keepdim=True)) / TEMPERATURE
    variants(run, "gumbel-softmax-backward", shape,
             lambda: soft * (
                 gradient - (gradient * soft).sum(-1, keepdim=True)) / TEMPERATURE,
             12 * rows * classes,
             lambda value: maximum_error(value, backward_expected))


def run_importance(run, rays, device):
    samples = 64
    shape = f"[{rays},64,64]"
    t_values = torch.linspace(0.0, 1.0, samples, device=device).expand(rays, samples)
    weights = torch.linspace(0.25, 1.25, samples, device=device).expand(rays, samples)
    base = torch.arange(samples, device=device, dtype=torch.float32).expand(rays, samples)

    def importance():
        pdf = weights / weights.sum(-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat((torch.zeros_like(cdf[:, :1]), cdf), dim=-1)
        targets = (base + torch.rand_like(base)) / samples
        indices = torch.searchsorted(cdf.contiguous(), targets.contiguous(), right=True)
        below = (indices - 1).clamp(0, samples - 1)
        above = indices.clamp(0, samples - 1)
        cdf_below = torch.gather(cdf, 1, below)
        cdf_above = torch.gather(cdf, 1, above + 1)
        t_below = torch.gather(t_values, 1, below)
        t_above = torch.gather(t_values, 1, above)
        denominator = torch.clamp(cdf_above - cdf_below, min=1.0e-8)
        return t_below + (targets - cdf_below) / denominator * (t_above - t_below)

    variants(run, "importance-sampling", shape, importance,
             12 * rays * samples, lambda value: validate_range(value, 0.0, 1.0))


def run_bias_dropout(run, rows, device):
    columns = 256
    shape = f"[{rows},256]"
    source = torch.linspace(-1.0, 1.0, rows * columns, device=device).reshape(rows, columns)
    bias = torch.linspace(-0.5, 0.5, columns, device=device)

    def operation():
        return torch.ops.aten.native_dropout.default(source + bias, P, True)

    def validator(value):
        output, bool_mask = value
        expected = (source + bias) * bool_mask.to(source.dtype) / KEEP
        return maximum_error(output, expected)

    variants(run, "bias-dropout", shape, operation,
             12 * rows * columns + 4 * columns, validator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not torch.cuda.is_available():
        print("CUDA-enabled Python PyTorch is required.", file=sys.stderr)
        return 2

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    for run in range(1, args.runs + 1):
        torch.manual_seed(849_000 + run)
        torch.cuda.manual_seed_all(849_000 + run)
        for elements in VECTOR_SHAPES:
            run_vector(run, elements, device)
        for rows in ROW_SHAPES:
            run_rows(run, rows, device)
        for rays in RAY_SHAPES:
            run_importance(run, rays, device)
        for rows in BIAS_SHAPES:
            run_bias_dropout(run, rows, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
