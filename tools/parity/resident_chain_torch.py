"""#890 phase 3: the PyTorch composition of the resident-program benchmark's chain, timed with the same protocol.

The chain (fp32, CUDA): h = relu(x * scale + bias) per column; p = softmax(h, dim=-1); y = p + residual. Run as eager
PyTorch ops and, for reference, replayed from a CUDA graph. 50 chains per timed region with one synchronize at the end,
51 samples, three runs; the per-chain median and P95/median are printed in microseconds. Compare with
AiDotNet.Tensors.Benchmarks --resident-program, run alternately with this script on the same GPU state.
"""
import sys
import time

import torch

WARMUP, SAMPLES, CHAINS, RUNS = 20, 51, 50, 3
SHAPES = [(64, 256), (128, 512), (256, 1024), (1024, 1024)]


def chain(x, scale, bias, residual):
    return torch.softmax(torch.relu(x * scale + bias), dim=-1) + residual


def time_region(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(CHAINS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e6 / CHAINS


def main():
    if not torch.cuda.is_available():
        print("needs a CUDA build of torch", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device="cuda").manual_seed(890)
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    print("shape        run    eager us   graph us   P95/med eager")
    for rows, cols in SHAPES:
        x = torch.rand(rows, cols, device="cuda", generator=gen) * 2 - 1
        scale = torch.rand(cols, device="cuda", generator=gen) * 2 - 1
        bias = torch.rand(cols, device="cuda", generator=gen) * 2 - 1
        residual = torch.rand(rows, cols, device="cuda", generator=gen) * 2 - 1

        eager = lambda: chain(x, scale, bias, residual)
        static_out = None
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                eager()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = eager()
        replay = graph.replay

        for _ in range(WARMUP):
            eager()
            replay()
        for run in range(1, RUNS + 1):
            e, g = [], []
            for s in range(SAMPLES):
                if s % 2 == 0:
                    e.append(time_region(eager)); g.append(time_region(replay))
                else:
                    g.append(time_region(replay)); e.append(time_region(eager))
            e.sort(); g.sort()
            med_e, med_g = e[len(e) // 2], g[len(g) // 2]
            p95 = e[min(len(e) - 1, int(0.95 * len(e) + 0.999) - 1)]
            print(f"{rows}x{cols:<8} {run:3d}  {med_e:9.2f}  {med_g:9.2f}   {p95 / med_e:12.2f}")
        del static_out
    return 0


if __name__ == "__main__":
    sys.exit(main())
