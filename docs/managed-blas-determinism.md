# ManagedBlas determinism & dispatch modes

AiDotNet.Tensors runs CPU matrix multiplication (GEMM) through its own managed
kernel, **ManagedBlas** (`BlasManaged.Gemm`), as the default path. Native OpenBLAS
is kept as an **optional per-shape accelerator**: in non-deterministic mode the
autotune router may dispatch to it for the (typically large, single-stream) shapes
where it still edges the managed kernel. This document describes the two execution
modes, the determinism guarantee, and how dispatch is routed.

## Two modes

Select with a single process-wide switch:

```csharp
AiDotNetEngine.SetDeterministicMode(true);   // reproducible (default)
AiDotNetEngine.SetDeterministicMode(false);  // maximum throughput
```

| | Deterministic (default) | Non-deterministic |
|---|---|---|
| GEMM kernel | ManagedBlas, always | per-shape best of ManagedBlas / native OpenBLAS (autotuned) |
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

All `BlasProvider.TryGemm*` entry points decide managed-vs-native via one helper,
`BlasProvider.ShouldRouteManaged`, in this priority order:

1. **`BlasManaged.PreferManaged`** (default `false`) — force managed everywhere; set
   `true` for supply-chain-hardened deployments that want zero native attack surface.
2. **Deterministic mode** — always managed (the reproducible-and-parallel kernel).
   Deterministic mode never consults the timing-based autotune below, because a
   measurement-chosen kernel would not be reproducible.
3. **`BlasManaged.AutotuneRouting`** (default `true`) — in *non-deterministic* mode,
   `PrefersManagedCache` measures both kernels once per `(shape, hardware)` tuple
   (disk-persisted under `~/.aidotnet/autotune/`) and routes future calls to the
   faster one. Managed where it wins, native where it wins — so the managed default
   never costs throughput.
4. Otherwise → native OpenBLAS.

## Quick reference

```csharp
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;

// Reproducible + parallel (default).
AiDotNetEngine.SetDeterministicMode(true);

// Max throughput; per-shape best-of managed/native.
AiDotNetEngine.SetDeterministicMode(false);

// Hardened: never touch native BLAS, in any mode.
BlasManaged.PreferManaged = true;

// Disable per-shape autotune. Deterministic mode still routes managed; in
// non-deterministic mode dispatch then falls back to native (unless PreferManaged
// is true).
BlasManaged.AutotuneRouting = false;
```
