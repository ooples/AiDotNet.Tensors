# Copyright (c) CrowdTrainer LLC. All rights reserved.
"""PyTorch side of CpuOpBreakdownBench: the same operations and shapes, the same output format.

Every convolution and linear layer of the MLP, CNN and ResNet-18 parity networks, timed as forward, input gradient
and weight gradient at 1, 2 and 4 threads (median of 31 after 5 warm-ups). Convolutions are timed in NCHW and in
channels_last, since the parity harness lets PyTorch run at its best and channels_last is often that. The two layouts
of an operation are timed interleaved, alternating which runs first in each sample, so neither layout always runs on
the cache the other left behind.
Prints lines of the form: OPBREAKDOWN torch | <op> | <shape> | t<threads> | <median ms>
and, for each interleaved pair, one line of raw samples in run order:
OPSAMPLES torch | <op> | <shape> | t<threads> | <first layout>:<ms>,<second layout>:<ms> ...
"""
import os
import time

import torch

CONVS = [
    ("cnn", 64, 1, 32, 28, 3, 1, 1),
    ("cnn", 64, 32, 64, 14, 3, 1, 1),
    ("resnet18", 8, 3, 64, 32, 3, 1, 1),
    ("resnet18", 8, 64, 64, 32, 3, 1, 1),
    ("resnet18", 8, 64, 128, 32, 3, 2, 1),
    ("resnet18", 8, 64, 128, 32, 1, 2, 0),
    ("resnet18", 8, 128, 128, 16, 3, 1, 1),
    ("resnet18", 8, 128, 256, 16, 3, 2, 1),
    ("resnet18", 8, 128, 256, 16, 1, 2, 0),
    ("resnet18", 8, 256, 256, 8, 3, 1, 1),
    ("resnet18", 8, 256, 512, 8, 3, 2, 1),
    ("resnet18", 8, 256, 512, 8, 1, 2, 0),
    ("resnet18", 8, 512, 512, 4, 3, 1, 1),
    ("probe", 8, 64, 128, 16, 1, 1, 0),
    ("probe", 8, 128, 256, 8, 1, 1, 0),
    ("probe", 8, 256, 512, 4, 1, 1, 0),
]

DEPTHWISE = [
    ("mobilenetv2", 16, 32, 112, 1),
    ("mobilenetv2", 16, 96, 112, 2),
    ("mobilenetv2", 16, 144, 56, 1),
    ("mobilenetv2", 16, 144, 56, 2),
    ("mobilenetv2", 16, 192, 28, 1),
    ("mobilenetv2", 16, 192, 28, 2),
    ("mobilenetv2", 16, 384, 14, 1),
    ("mobilenetv2", 16, 576, 14, 1),
    ("mobilenetv2", 16, 576, 14, 2),
    ("mobilenetv2", 16, 960, 7, 1),
]

LINEARS = [
    ("mlp", 128, 784, 512),
    ("mlp", 128, 512, 256),
    ("mlp", 128, 256, 10),
    ("cnn", 64, 3136, 128),
    ("cnn", 64, 128, 10),
    ("resnet18", 8, 512, 10),
]


# Same profiling filters as the C# bench: a substring of "<op> | <shape>", the thread counts, the timed runs per row.
FILTER = os.environ.get("AIDOTNET_OP_FILTER", "")
THREADS = tuple(int(t) for t in os.environ.get("AIDOTNET_OP_THREADS", "1,2,4").split(","))
SAMPLES = int(os.environ.get("AIDOTNET_OP_SAMPLES", "75"))


def robust_ms(samples) -> float:
    """Probe branch: median over groups of three consecutive samples of each group's minimum (preemption-robust)."""
    groups = [min(samples[i:i + 3]) for i in range(0, len(samples) - 2, 3)]
    groups.sort()
    return groups[len(groups) // 2]


def median_ms(fn) -> float:
    # The same warm-up as the C# side: at least 5 calls and 250 ms.
    start = time.perf_counter()
    calls = 0
    while calls < 5 or time.perf_counter() - start < 0.25:
        fn()
        calls += 1
    samples = []
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return robust_ms(samples)


def row(op: str, shape: str, threads: int, fn) -> None:
    if FILTER and FILTER not in f"{op.replace('+cl', '')} | {shape}":
        return
    print(f"OPBREAKDOWN torch | {op} | {shape} | t{threads} | {median_ms(fn):.4f}", flush=True)


def layout_rows(op: str, shape: str, threads: int, contiguous, channels_last) -> None:
    """Times one operation in NCHW (<op>) and channels_last (<op>+cl), interleaved: sample i runs NCHW first when i is
    even and channels_last first when it is odd. Prints each layout's median, then the raw samples in run order."""
    if FILTER and FILTER not in f"{op} | {shape}":
        return
    fns = {"nchw": contiguous, "cl": channels_last}
    start = time.perf_counter()
    calls = 0
    while calls < 5 or time.perf_counter() - start < 0.25:
        contiguous()
        channels_last()
        calls += 1
    times = {"nchw": [], "cl": []}
    order = []
    for i in range(SAMPLES):
        pair = ("nchw", "cl") if i % 2 == 0 else ("cl", "nchw")
        sample = []
        for layout in pair:
            t0 = time.perf_counter()
            fns[layout]()
            ms = (time.perf_counter() - t0) * 1000.0
            times[layout].append(ms)
            sample.append(f"{layout}:{ms:.4f}")
        order.append(",".join(sample))
    for layout, suffix in (("nchw", ""), ("cl", "+cl")):
        print(f"OPBREAKDOWN torch | {op}{suffix} | {shape} | t{threads} | {robust_ms(times[layout]):.4f}", flush=True)
    print(f"OPSAMPLES torch | {op} | {shape} | t{threads} | {' '.join(order)}", flush=True)


def main() -> None:
    torch.manual_seed(7)
    with torch.no_grad():
        for threads in THREADS:
            torch.set_num_threads(threads)
            for net, b, cin, cout, hw, k, s, p in CONVS:
                shape = f"{net} b{b} {cin}->{cout} {hw}x{hw} k{k} s{s}"
                o = (hw + 2 * p - k) // s + 1
                conv_args = ([s, s], [p, p], [1, 1], False, [0, 0], 1)
                ops = {}
                for layout, fmt in (("nchw", torch.contiguous_format), ("cl", torch.channels_last)):
                    x = torch.randn(b, cin, hw, hw).contiguous(memory_format=fmt)
                    w = torch.randn(cout, cin, k, k).contiguous(memory_format=fmt)
                    dy = torch.randn(b, cout, o, o).contiguous(memory_format=fmt)
                    ops[layout] = (
                        lambda x=x, w=w: torch.nn.functional.conv2d(x, w, None, s, p),
                        lambda x=x, w=w, dy=dy: torch.ops.aten.convolution_backward(
                            dy, x, w, None, *conv_args, [True, False, False]),
                        lambda x=x, w=w, dy=dy: torch.ops.aten.convolution_backward(
                            dy, x, w, None, *conv_args, [False, True, False]),
                    )
                for i, op in enumerate(("conv.fwd", "conv.dgrad", "conv.wgrad")):
                    layout_rows(op, shape, threads, ops["nchw"][i], ops["cl"][i])
            for net, b, ch, hw, s in DEPTHWISE:
                shape = f"{net} b{b} dw{ch} {hw}x{hw} k3 s{s}"
                o = (hw + 2 - 3) // s + 1
                dw_args = ([s, s], [1, 1], [1, 1], False, [0, 0], ch)
                ops = {}
                for layout, fmt in (("nchw", torch.contiguous_format), ("cl", torch.channels_last)):
                    x = torch.randn(b, ch, hw, hw).contiguous(memory_format=fmt)
                    w = torch.randn(ch, 1, 3, 3).contiguous(memory_format=fmt)
                    dy = torch.randn(b, ch, o, o).contiguous(memory_format=fmt)
                    ops[layout] = (
                        lambda x=x, w=w: torch.nn.functional.conv2d(x, w, None, s, 1, 1, ch),
                        lambda x=x, w=w, dy=dy: torch.ops.aten.convolution_backward(
                            dy, x, w, None, *dw_args, [True, False, False]),
                        lambda x=x, w=w, dy=dy: torch.ops.aten.convolution_backward(
                            dy, x, w, None, *dw_args, [False, True, False]),
                    )
                for i, op in enumerate(("dwconv.fwd", "dwconv.dgrad", "dwconv.wgrad")):
                    layout_rows(op, shape, threads, ops["nchw"][i], ops["cl"][i])
            for net, b, fin, fout in LINEARS:
                shape = f"{net} b{b} {fin}->{fout}"
                x = torch.randn(b, fin)
                w = torch.randn(fin, fout)
                dy = torch.randn(b, fout)
                row("linear.fwd", shape, threads, lambda: torch.mm(x, w))
                row("linear.dgrad", shape, threads, lambda: torch.mm(dy, w.t()))
                row("linear.wgrad", shape, threads, lambda: torch.mm(x.t(), dy))


if __name__ == "__main__":
    main()
