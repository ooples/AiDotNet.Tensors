#!/usr/bin/env python3
"""Parity-210 baseline: torch.compile fullgraph mode.

Same protocol as run_torch_eager.py but wraps each op in torch.compile
fullgraph=True. Falls back to eager if torch.compile is unavailable (pre-2.x).
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


def _compile(fn):
    try:
        return torch.compile(fn, fullgraph=True)
    except Exception:
        return fn


def _time_einsum(equation, shape_strs, warmup, iters):
    tensors = [torch.randn(*_parse_shape(s), dtype=torch.float32) for s in shape_strs]
    fn = _compile(lambda *ts: torch.einsum(equation, *ts))
    for _ in range(warmup):
        _ = fn(*tensors)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(*tensors)
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
    else:
        sys.stderr.write(f"unsupported op for torch.compile: {op}\n")
        sys.exit(2)

    times.sort()
    trimmed = times[:int(iters * 0.9)] if iters >= 10 else times
    median = statistics.median(trimmed)
    p90 = trimmed[int(len(trimmed) * 0.9) - 1] if len(trimmed) >= 10 else trimmed[-1]
    print(f"{median:.4f},{p90:.4f},{len(trimmed)}")


if __name__ == '__main__':
    main()
