"""cuBLAS fp16 GEMM peers for the tensor-core lowering.

This is the denominator that matters. Beating our own scalar emitter by 5x only says the
tensor cores are faster than the FP32 pipes, which was never in doubt; the question the
blueprint asks is where we stand against the strongest thing a user could otherwise call.

The competitor runs in its CUDA-graph lane for the same reason every other competitor on
this branch does: eager mode pays a launch and an output allocation per call, and beating
that measures PyTorch's dispatcher rather than anyone's kernel.
"""

import json
import statistics
import torch

WARMUPS = 30
SAMPLES = 21
BATCH = 200      # launches per timed region -- see measure()

# (M, K, N) -- matching the shapes --tensorcore-check verifies.
SHAPES = [
    (64, 64, 64),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (256, 2048, 256),
    # LARGE SHAPES ARE THE ONLY TRUSTWORTHY ONES on this box. At 64^3 both sides sit on a
    # ~50us launch-submission floor under WDDM, so any ratio there measures the driver.
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure(launch):
    """Times a BATCH of launches and divides, rather than timing one.

    Timing a single launch on this box measures Windows' WDDM submission granularity, not
    the kernel: the first attempt at this reported 43-87us for every shape from 64^3 to
    1024^3, and had the fp16->fp32 lane -- which does strictly MORE work -- coming out
    faster than fp16->fp16 at every size. Both are the signature of a fixed floor swamping
    the signal. Amortising over a batch is also what the C# harness does, so the two sides
    of the comparison are now measured the same way.
    """
    for _ in range(WARMUPS):
        launch()
    torch.cuda.synchronize()

    samples = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(BATCH):
            launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / BATCH)
    return samples


def graphed(fn, *tensors):
    """Captures fn into a CUDA graph, removing launch overhead from the comparison."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(*tensors)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*tensors)
    return graph.replay


def sm_clock():
    """The clock, when nvidia-ml-py is present. Its absence must not lose a whole run."""
    try:
        return torch.cuda.clock_rate()
    except Exception:
        return None


def main():
    results = []
    device = torch.device("cuda")

    for m, k, n in SHAPES:
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)
        half_out = torch.empty(m, n, device=device, dtype=torch.float16)
        wide_out = torch.empty(m, n, device=device, dtype=torch.float32)

        flops = 2.0 * m * k * n

        # TWO LANES, because cuBLAS and our kernel do not write the same thing.
        #
        #   fp16->fp16  cuBLAS's native form. It accumulates in fp32 internally but stores
        #               fp16, so it moves HALF the output bytes our kernel does. Comparing
        #               against it is the conservative direction -- we are handicapped.
        #   fp16->fp32  the same end-to-end operation ours performs: fp16 operands, fp32
        #               result in memory. cuBLAS needs a second pass to get there, which is
        #               exactly the fusion cost the campaign trades on, so this is the
        #               apples-to-apples number.
        def gemm_half(a=a, b=b, out=half_out):
            torch.matmul(a, b, out=out)

        def gemm_wide(a=a, b=b, half=half_out, out=wide_out):
            torch.matmul(a, b, out=half)
            out.copy_(half)

        for name, fn in (("cublas fp16->fp16", gemm_half),
                         ("cublas fp16->fp32", gemm_wide)):
            try:
                replay = graphed(fn)
                samples = measure(replay)
                lane = "cuda-graph"
            except Exception:
                samples = measure(fn)
                lane = "eager"

            median_us = statistics.median(samples)
            results.append({
                "shape": f"{m}x{k}x{n}",
                "m": m, "k": k, "n": n,
                "method": name,
                "lane": lane,
                "median_us": median_us,
                "p95_us": percentile(samples, 0.95),
                "tflops": flops / median_us / 1_000_000.0,
            })

    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "sm_clock_mhz": sm_clock(),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
