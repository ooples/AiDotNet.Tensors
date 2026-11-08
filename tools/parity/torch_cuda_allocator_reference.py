#!/usr/bin/env python3
"""PyTorch CUDA caching-allocator reference for the device memory release comparison.

Measures what GpuEngineReleaseTests' comparison measures on our side, with the same sizes:
  release: `count` results of x + x (`elements` float32 each) are dropped, then torch.cuda.empty_cache() runs; the
           free device memory (torch.cuda.mem_get_info) before and after it, and the call's duration.
  churn:   allocate and free one float32 tensor of each size in `churn_sizes`, `churn_iterations` times after a
           warm-up, timed per allocate + free pair (torch.empty from Python, as a PyTorch user allocates).

Usage: torch_cuda_allocator_reference.py <case.json> <result.json>
"""
import gc
import json
import sys
import time

import torch


def release(elements, count):
    x = torch.zeros(elements, dtype=torch.float32, device="cuda")
    ys = [x + x for _ in range(count)]
    torch.cuda.synchronize()
    del ys
    gc.collect()
    before = torch.cuda.mem_get_info()[0]
    start = time.perf_counter()
    torch.cuda.empty_cache()
    seconds = time.perf_counter() - start
    after = torch.cuda.mem_get_info()[0]
    del x
    torch.cuda.empty_cache()
    return {"freed_bytes": after - before, "seconds": seconds}


def churn(elements, iterations, warmup):
    for _ in range(warmup):
        t = torch.empty(elements, dtype=torch.float32, device="cuda")
        del t
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        t = torch.empty(elements, dtype=torch.float32, device="cuda")
        del t
    torch.cuda.synchronize()
    return {"elements": elements, "seconds_per_pair": (time.perf_counter() - start) / iterations}


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False (a CPU build of torch, or no device)", file=sys.stderr)
        return 3
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        c = json.load(f)
    result = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "release": release(c["elements"], c["count"]),
        "churn": [churn(e, c["churn_iterations"], c["churn_warmup"]) for e in c["churn_sizes"]],
    }
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
