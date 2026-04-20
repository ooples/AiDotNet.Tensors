#!/usr/bin/env python3
"""Parity-210 baseline: JAX under jit."""
import sys
import time
import statistics

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    sys.stderr.write("jax not installed\n")
    sys.exit(2)


def _parse_shape(s):
    return tuple(int(x) for x in s.split('x') if x)


def _time_einsum(equation, shape_strs, warmup, iters):
    rng = jax.random.PRNGKey(0)
    tensors = []
    for i, s in enumerate(shape_strs):
        rng, sub = jax.random.split(rng)
        tensors.append(jax.random.normal(sub, _parse_shape(s)))
    fn = jax.jit(lambda *ts: jnp.einsum(equation, *ts))
    # Warmup includes the jit trace.
    for _ in range(warmup):
        _ = fn(*tensors).block_until_ready()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(*tensors).block_until_ready()
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
        sys.stderr.write(f"jax runner only supports einsum, got: {op}\n")
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
