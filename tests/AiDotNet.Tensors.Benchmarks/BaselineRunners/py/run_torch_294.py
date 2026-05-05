#!/usr/bin/env python3
"""Issue #294 PyTorch CPU baseline: matmul, conv2d, FlashAttention,
LayerNorm, BCE derivative.

Invoked as a subprocess by Issue294PyTorchParityBenchmark.cs. Reads
one line from stdin:
    <op_name>|<args>|<warmup>|<iters>

Writes one CSV line to stdout:
    <median_ms>,<p90_ms>,<iters>

op_name | args
--------|-----
matmul   | <m>x<k>x<n>
conv2d   | <n>x<c>x<h>x<w>;<oc>x<ic>x<kh>x<kw>;<stride>;<pad>
attn     | <b>x<h>x<sq>x<d>;<causal>      (FlashAttention rank-4)
layernorm | <b>x<f>
bcederiv | <n>                             (BCE forward+backward)
"""

import math
import sys
import time
import statistics

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.stderr.write("torch not installed\n")
    sys.exit(2)


def _parse_shape(s):
    return tuple(int(x) for x in s.split('x') if x)


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def _time_matmul(args, warmup, iters):
    parts = args.split('x')
    m, k, n = int(parts[0]), int(parts[1]), int(parts[2])
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    return _time(lambda: torch.matmul(a, b), warmup, iters)


def _time_conv2d(args, warmup, iters):
    in_str, w_str, stride_str, pad_str = args.split(';')
    n, c, h, w = _parse_shape(in_str)
    oc, ic, kh, kw = _parse_shape(w_str)
    stride = int(stride_str)
    pad = int(pad_str)
    x = torch.randn(n, c, h, w, dtype=torch.float32)
    weight = torch.randn(oc, ic, kh, kw, dtype=torch.float32)
    return _time(
        lambda: F.conv2d(x, weight, stride=stride, padding=pad),
        warmup, iters)


def _time_attn(args, warmup, iters):
    shape_str, causal_str = args.split(';')
    b, h, sq, d = _parse_shape(shape_str)
    causal = causal_str.lower() == 'true'
    q = torch.randn(b, h, sq, d, dtype=torch.float32)
    k = torch.randn(b, h, sq, d, dtype=torch.float32)
    v = torch.randn(b, h, sq, d, dtype=torch.float32)
    # Use scaled_dot_product_attention which dispatches to the
    # FlashAttention kernel on hardware that supports it; on CPU it
    # falls back to a math implementation. This is the closest
    # PyTorch CPU equivalent to our FlashAttention<T>.
    return _time(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
        warmup, iters)


def _time_layernorm(args, warmup, iters):
    parts = args.split('x')
    b, f = int(parts[0]), int(parts[1])
    x = torch.randn(b, f, dtype=torch.float32)
    weight = torch.ones(f, dtype=torch.float32)
    bias = torch.zeros(f, dtype=torch.float32)
    return _time(
        lambda: F.layer_norm(x, [f], weight, bias, eps=1e-5),
        warmup, iters)


def _time_bce_deriv(args, warmup, iters):
    n = int(args)
    pred = torch.rand(n, dtype=torch.float32, requires_grad=True)
    target = torch.rand(n, dtype=torch.float32)

    def step():
        if pred.grad is not None:
            pred.grad.zero_()
        loss = F.binary_cross_entropy(pred, target, reduction='sum')
        loss.backward()

    return _time(step, warmup, iters)


def main():
    line = sys.stdin.readline().strip()
    op, args, warmup, iters = line.split('|')
    warmup, iters = int(warmup), int(iters)

    if op == 'matmul':
        times = _time_matmul(args, warmup, iters)
    elif op == 'conv2d':
        times = _time_conv2d(args, warmup, iters)
    elif op == 'attn':
        times = _time_attn(args, warmup, iters)
    elif op == 'layernorm':
        times = _time_layernorm(args, warmup, iters)
    elif op == 'bcederiv':
        times = _time_bce_deriv(args, warmup, iters)
    else:
        sys.stderr.write(f"unknown op {op}\n")
        sys.exit(3)

    median = statistics.median(times)
    # P90 via zero-based index: ceil(0.9·N)-1 selects the 90th-percentile
    # element. The previous int(0.9·N) form was off-by-one — for N=10
    # it landed on index 9 (the maximum), reporting an artificially
    # high p90.
    p90 = sorted(times)[max(0, math.ceil(0.9 * len(times)) - 1)]
    sys.stdout.write(f"{median:.6f},{p90:.6f},{iters}\n")


if __name__ == '__main__':
    main()
