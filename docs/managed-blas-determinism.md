# ManagedBlas determinism & dispatch modes

AiDotNet.Tensors runs CPU matrix multiplication (GEMM) through its own managed
kernel, **ManagedBlas** (`BlasManaged.Gemm`). There is no native BLAS dependency
(OpenBLAS and Intel MKL were removed). This document describes the two execution
modes, the determinism guarantee, and how dispatch is routed.

## Two modes

Select with a single process-wide switch:

```csharp
AiDotNetEngine.SetDeterministicMode(true);   // reproducible (default)
AiDotNetEngine.SetDeterministicMode(false);  // maximum throughput
```

| | Deterministic (default) | Non-deterministic |
|---|---|---|
| GEMM kernel | ManagedBlas | ManagedBlas, which may also split the K-reduction across threads |
| Result reproducibility | **bit-identical** across runs and thread counts (same machine/ISA) | not guaranteed bit-identical (reduction order may vary) |
| Parallelism | **full** (multi-threaded) | full |
| Use when | training/inference that must reproduce exactly; debugging; regression baselines | latency/throughput is the only goal |

### Why this exceeds the usual tradeoff

PyTorch's deterministic mode (`torch.use_deterministic_algorithms(True)`) sacrifices
parallelism for some ops. ManagedBlas does **not**: deterministic mode is both
**reproducible and fully parallel**. This is possible because every parallel GEMM
path splits work over **disjoint output tiles** (rows / columns / 2-D blocks) and
performs each output element's full K-reduction on a single thread in a fixed order.
The number of threads (i.e. how the tiles are partitioned) therefore never changes
the result. The only split that *would* change the result — partitioning the
K-reduction across threads and summing partials — is disabled in deterministic mode
(`AxisSelector` gates it on non-deterministic mode only).

This contract is pinned by `DeterministicParallelGemmContractTests`, which runs the
same GEMM at 1/2/4/16 threads and asserts the output is bit-identical.

### Scope of the determinism guarantee

- **Guaranteed:** bit-identical across runs and across thread counts **on the same
  machine / instruction set**.
- **Not guaranteed:** bit-identical across *different* instruction sets (AVX-512 vs
  AVX2 vs scalar vs ARM Neon), which reduce at different vector widths. No mainstream
  framework (PyTorch included) guarantees cross-ISA bit-exactness either.

## Dispatch routing

Every CPU GEMM, including the `BlasProvider.Gemm` and `BlasProvider.GemmEx` entry
points, runs the managed `BlasManaged` kernel in both modes. The mode changes only
which splits that kernel may use (the K-reduction split above) and whether the
order-dependent reductions elsewhere run serially (`BlasProvider.SetDeterministicMode`).

## Quick reference

```csharp
using AiDotNet.Tensors.Engines;

// Reproducible + parallel (default).
AiDotNetEngine.SetDeterministicMode(true);

// Max throughput: K-reduction splits allowed, results may vary with thread count.
AiDotNetEngine.SetDeterministicMode(false);
```
