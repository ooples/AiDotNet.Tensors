#!/usr/bin/env python3
"""Parity-210 baseline: opt_einsum path optimizer.

Compares our built-in greedy path optimizer against the industry-standard
opt_einsum package (https://github.com/dgasmith/opt_einsum).
"""
import sys
import time
import statistics

try:
    import numpy as np
    import opt_einsum as oe
except ImportError:
    sys.stderr.write("numpy or opt_einsum not installed\n")
    sys.exit(2)


def _parse_shape(s):
    return tuple(int(x) for x in s.split('x') if x)


def _time_einsum(equation, shape_strs, warmup, iters):
    tensors = [np.random.randn(*_parse_shape(s)).astype(np.float32) for s in shape_strs]
    # opt_einsum plans the contraction path ahead of time and reuses it.
    expr = oe.contract_expression(equation, *[t.shape for t in tensors])
    for _ in range(warmup):
        _ = expr(*tensors)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = expr(*tensors)
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

    if op != 'einsum':
        sys.stderr.write(f"opt_einsum runner only supports einsum, got: {op}\n")
        sys.exit(2)

    eq, shapes = args.split(';', 1)
    times = _time_einsum(eq, shapes.split(','), warmup, iters)
    times.sort()
    trimmed = times[:int(iters * 0.9)] if iters >= 10 else times
    median = statistics.median(trimmed)
    p90 = trimmed[int(len(trimmed) * 0.9) - 1] if len(trimmed) >= 10 else trimmed[-1]
    print(f"{median:.4f},{p90:.4f},{len(trimmed)}")


if __name__ == '__main__':
    main()
