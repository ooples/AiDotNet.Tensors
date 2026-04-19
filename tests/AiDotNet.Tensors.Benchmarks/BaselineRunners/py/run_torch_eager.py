#!/usr/bin/env python3
"""Parity-210 baseline: torch.* eager mode timing.

Invoked as a subprocess by PythonBaselineRunner.cs. Reads one line from
stdin:
    <op_name>|<args>|<warmup>|<iters>

args is op-specific; for einsum it's "equation;shape1,shape2,...".

Writes one CSV line to stdout:
    <median_ms>,<p90_ms>,<iters>

Exits non-zero if torch is unavailable or args malformed — the runner
treats that as "skipped".
"""
import sys
import time
import statistics

try:
    import torch
except ImportError:
    sys.stderr.write("torch not installed\n")
    sys.exit(2)


def _parse_shape(s):
    return tuple(int(x) for x in s.split('x') if x)


def _time_einsum(equation, shape_strs, warmup, iters):
    tensors = [torch.randn(*_parse_shape(s), dtype=torch.float32) for s in shape_strs]
    for _ in range(warmup):
        _ = torch.einsum(equation, *tensors)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = torch.einsum(equation, *tensors)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def _time_cumsum(shape_str, axis, warmup, iters):
    x = torch.randn(*_parse_shape(shape_str), dtype=torch.float32)
    for _ in range(warmup):
        _ = torch.cumsum(x, dim=axis)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = torch.cumsum(x, dim=axis)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def _time_sort(shape_str, axis, warmup, iters):
    x = torch.randn(*_parse_shape(shape_str), dtype=torch.float32)
    for _ in range(warmup):
        _ = torch.sort(x, dim=axis)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = torch.sort(x, dim=axis)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def main():
    line = sys.stdin.readline().strip()
    if not line:
        sys.exit(2)
    parts = line.split('|', 3)
    if len(parts) != 4:
        sys.exit(2)
    op, args, warmup_s, iters_s = parts
    warmup = int(warmup_s)
    iters = int(iters_s)

    if op == 'einsum':
        eq, shapes = args.split(';', 1)
        times = _time_einsum(eq, shapes.split(','), warmup, iters)
    elif op == 'cumsum':
        shape, axis = args.split(';', 1)
        times = _time_cumsum(shape, int(axis), warmup, iters)
    elif op == 'sort':
        shape, axis = args.split(';', 1)
        times = _time_sort(shape, int(axis), warmup, iters)
    else:
        sys.stderr.write(f"unsupported op: {op}\n")
        sys.exit(2)

    # Trim the slowest 10%, report median + p90 of remainder.
    times.sort()
    trimmed = times[:int(iters * 0.9)] if iters >= 10 else times
    median = statistics.median(trimmed)
    p90 = trimmed[int(len(trimmed) * 0.9) - 1] if len(trimmed) >= 10 else trimmed[-1]
    print(f"{median:.4f},{p90:.4f},{len(trimmed)}")


if __name__ == '__main__':
    main()
