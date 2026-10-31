# Copyright (c) CrowdTrainer LLC. All rights reserved.
"""PyTorch side of CpuOpBreakdownBench: the same operations and shapes, the same output format.

Every convolution and linear layer of the MLP, CNN and ResNet-18 parity networks, timed as forward, input gradient
and weight gradient at 1, 2 and 4 threads (median of 31 after 5 warm-ups). Convolutions are timed in NCHW and in
channels_last, since the parity harness lets PyTorch run at its best and channels_last is often that.
Prints lines of the form: OPBREAKDOWN torch | <op> | <shape> | t<threads> | <median ms>
"""
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
]

LINEARS = [
    ("mlp", 128, 784, 512),
    ("mlp", 128, 512, 256),
    ("mlp", 128, 256, 10),
    ("cnn", 64, 3136, 128),
    ("cnn", 64, 128, 10),
    ("resnet18", 8, 512, 10),
]


def median_ms(fn) -> float:
    for _ in range(5):
        fn()
    samples = []
    for _ in range(31):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def row(op: str, shape: str, threads: int, fn) -> None:
    print(f"OPBREAKDOWN torch | {op} | {shape} | t{threads} | {median_ms(fn):.4f}", flush=True)


def main() -> None:
    torch.manual_seed(7)
    with torch.no_grad():
        for threads in (1, 2, 4):
            torch.set_num_threads(threads)
            for net, b, cin, cout, hw, k, s, p in CONVS:
                shape = f"{net} b{b} {cin}->{cout} {hw}x{hw} k{k} s{s}"
                o = (hw + 2 * p - k) // s + 1
                for layout, fmt in (("", torch.contiguous_format), ("+cl", torch.channels_last)):
                    x = torch.randn(b, cin, hw, hw).contiguous(memory_format=fmt)
                    w = torch.randn(cout, cin, k, k).contiguous(memory_format=fmt)
                    dy = torch.randn(b, cout, o, o).contiguous(memory_format=fmt)
                    conv_args = ([s, s], [p, p], [1, 1], False, [0, 0], 1)
                    row("conv.fwd" + layout, shape, threads,
                        lambda: torch.nn.functional.conv2d(x, w, None, s, p))
                    row("conv.dgrad" + layout, shape, threads,
                        lambda: torch.ops.aten.convolution_backward(dy, x, w, None, *conv_args, [True, False, False]))
                    row("conv.wgrad" + layout, shape, threads,
                        lambda: torch.ops.aten.convolution_backward(dy, x, w, None, *conv_args, [False, True, False]))
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
