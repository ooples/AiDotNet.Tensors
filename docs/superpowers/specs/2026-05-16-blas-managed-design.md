# BlasManaged — Full BLIS-style Managed GEMM Kernel

**Date:** 2026-05-16
**Author:** franklinic (with brainstorming assist)
**Issue:** [#358 — ConvTranspose2D L2-shape (M=4096 N=16 K=512 transA=true) runs 50–150× below MKL peak](https://github.com/ooples/AiDotNet.Tensors/issues/358)
**Status:** Draft — pending user review

---

## 1. Background + motivation

Issue #358 documents a pathological GEMM shape — `M=4096, N=16, K=512, transA=true` — that the DCGAN generator at batch=1 hits as its L2 deconvolution layer. Both Intel MKL (~559 ms) and OpenBLAS (~215 ms) run this shape **50–150× below their own DGEMM peak** (~0.44 ms at 75 GFLOPS). The naive 7-nested loop currently lands at ~100 ms — *faster than BLAS* but still 200× off peak.

The shape blocks `DCGANTests.MoreData_ShouldNotDegrade` from passing its 120 s budget and represents a broader CPU-batch=1 pathology that affects DCGAN-style generators, VAE/flow decoders, and diffusion U-Net upsample stages.

PR #357 mitigated similar pathologies for L4/L6/L8 deconv layers via a heuristic that pre-transposes the kernel when `kernelSize ≤ 4 MB AND hw ∈ [32, 256]`. L2's 16 MB kernel and `hw = 16` miss both clauses, so it falls into the slow `transA=true` BLAS path with no escape.

The fix this spec proposes — **a full BLIS-style managed GEMM kernel that replaces the existing `Avx512Sgemm` + `SimdGemm` paths** — closes #358 by hitting hardware peak (~0.5 ms FP64) and unlocks longer-term wins across MatMul, Conv2D, attention, and fused-multi-layer paths.

## 2. Goals + non-goals

### Goals

1. **Close #358.** Get the L2 shape under 1 ms FP64 on x64+AVX-512.
2. **Beat MKL on shapes MKL is bad at** — small-N transposed, tall-thin, weight-reused training loops.
3. **Match or beat the existing `Avx512Sgemm`/`SimdGemm` on every shape they handle.**
4. **Surpass PyTorch CPU on training loops** via weight pre-pack caching and adaptive autotune (CPU PyTorch dispatches to the same MKL we're competing with; we need things PyTorch doesn't do).
5. **Single source of truth for managed GEMM** in the codebase. One dispatcher, one set of microkernels, configurable for power users.

### Non-goals

- **External BLAS replacement.** When native BLAS is available and the shape is in its sweet spot, we still dispatch to it. BlasManaged is for shapes BLAS handles poorly and for the no-native-BLAS fallback path.
- **GPU dispatch.** GPU engines have their own kernels (CUDA/OpenCL/HIP/Metal/Vulkan/WebGPU). BlasManaged is CPU only.
- **INT8 / FP16 / BF16 quantization.** FP32 + FP64 only in this PR. Quantized paths are a follow-up.
- **Direct convolution (Winograd, FFT).** The im2col + GEMM decomposition stays; we just make the GEMM step fast.
- **Sparse weights.** Dense GEMM only.

## 3. Architecture overview

`BlasManaged` — a managed-C# BLIS-style GEMM kernel under `src/AiDotNet.Tensors/Engines/BlasManaged/`. Layered structure top to bottom:

```
┌────────────────────────────────────────────────────────────────────┐
│ Public API: BlasManaged.Gemm<T>(...) + epilogue variants           │  ← all callers
├────────────────────────────────────────────────────────────────────┤
│ Dispatcher:  shape analysis → autotune cache → strategy selection  │
├────────────────────────────────────────────────────────────────────┤
│ Strategy layer (3): PackBoth / PackAOnly / Streaming               │
├────────────────────────────────────────────────────────────────────┤
│ Architecture dispatch (4): Avx512 / Avx2 / Scalar / Neon           │
├────────────────────────────────────────────────────────────────────┤
│ Microkernel pool: hand-written variants + JIT-emitted specializa-  │
│                   tions per (M,N,K,trans,precision,arch) tuple     │
├────────────────────────────────────────────────────────────────────┤
│ Allocator (5 layers): Per-thread pool → ArrayPool → Pack cache →   │
│                       TensorArena → User-supplied workspace        │
├────────────────────────────────────────────────────────────────────┤
│ Autotune cache: AutotuneCache-backed, per-shape decisions          │
└────────────────────────────────────────────────────────────────────┘
```

### Public API (final)

```csharp
namespace AiDotNet.Tensors.Engines.BlasManaged;

public static class BlasManaged {
    // Primary entry point (replaces SimdGemm.Sgemm/Dgemm + Avx512Sgemm.SgemmBlocked).
    public static void Gemm<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c,         int ldc,
        int m, int n, int k,
        in BlasOptions<T> options = default) where T : unmanaged;

    // Optional pre-pack handles for training-loop weight reuse.
    public static WeightPackHandle PrePackA<T>(...) where T : unmanaged;
    public static WeightPackHandle PrePackB<T>(...) where T : unmanaged;

    // Diagnostics.
    public static BlasManagedStats GetStats();
    public static void ClearCaches();
}

public readonly ref struct BlasOptions<T> where T : unmanaged {
    public PackingMode PackingMode { get; init; }   // Auto / ForcePackBoth / ForcePackAOnly /
                                                    // ForceStreaming / DisableAutotune
    public Epilogue<T> Epilogue    { get; init; }   // bias / activation / skip / dropout / scale
    public Span<byte> Workspace    { get; init; }   // optional caller-supplied scratch
    public WeightPackHandle? PackedA { get; init; } // pre-packed A from training loop
    public WeightPackHandle? PackedB { get; init; } // pre-packed B
    public int NumThreads          { get; init; }   // 0 = autotune, -1 = single-thread, positive = pin
    public ulong AutotuneKey       { get; init; }   // 0 = derive from shape; nonzero = caller key
    public long MaxJitCacheBytes   { get; init; }   // 0 = process default (64 MB)
}

public enum PackingMode {
    Auto, ForcePackBoth, ForcePackAOnly, ForceStreaming, DisableAutotune
}

public readonly ref struct Epilogue<T> where T : unmanaged {
    public ReadOnlySpan<T> BiasN     { get; init; } // empty = no bias  (length = N)
    public FusedActivationType Activation { get; init; }
    public ReadOnlySpan<T> SkipMxN   { get; init; } // empty = no skip-connection
    public uint DropoutMask          { get; init; } // 0 = no dropout
    public T OutputScale             { get; init; } // 1 = no scale
}

public sealed class WeightPackHandle : IDisposable {
    public void MarkDirty();                 // optimizer-step path calls this
    public void Dispose();
}
```

### Compatibility shims

`Avx512Sgemm.SgemmBlocked`, `SimdGemm.Sgemm`, `SimdGemm.Dgemm` retain their existing public signatures but their bodies become single-line forwards to `BlasManaged.Gemm`. Marked `[Obsolete]` with a one-release deprecation window. Internal callers all re-point in this PR.

## 4. Microkernel design

The microkernel is the innermost loop — it must saturate the FMA ports. One hand-written kernel per arch×precision combination; JIT layer (Section 7) emits shape-specialized clones.

### Per-arch microkernel widths (`Mr × Nr`)

| Arch + Precision | Mr × Nr | Vector type | Accumulators | Rationale |
|---|---|---|---|---|
| AVX-512 / FP32 | 16 × 16 | `Vector512<float>` | 16 regs (one per row) | 16 lanes × 16 rows = full ZMM register file |
| AVX-512 / FP64 | 8 × 16  | `Vector512<double>` | 16 regs (one per row pair) | 8 lanes × 2 rows packed → 16 active accumulators |
| AVX2 / FP32 | 8 × 8 | `Vector256<float>` | 8 regs | 8 lanes × 8 rows = full YMM register file |
| AVX2 / FP64 | 4 × 8 | `Vector256<double>` | 8 regs | 4 lanes × 8 rows |
| ARM64 Neon / FP32 | 8 × 4 | `Vector128<float>` | 8 regs | 4 lanes × 8 rows; 32 NEON registers |
| ARM64 Neon / FP64 | 4 × 4 | `Vector128<double>` | 8 regs | 2 lanes × 8 rows × 2 vectors |
| Scalar (net471) | 4 × 4 | plain `T` | 16 regs (one per cell) | Non-SIMD hosts; net471 |

### Inner K-loop pattern (AVX-512 / FP64 reference)

```csharp
// Register-resident across the entire K-loop:
//   acc[0..15] : 16 × Vector512<double> = the C tile accumulators (8 rows × 16 cols)
// Each K step does:
//   - 2 vector loads (bRow_lo, bRow_hi from packed-B — 16 doubles total)
//   - 1 packed-A column read (8 doubles)
//   - 8 rows × 2 vector halves = 16 broadcast-FMAs

for (int kk = 0; kk < Kc; kk++) {
    Vector512<double> bRow_lo = LoadPackedB(bPtr, kk, lo: true);
    Vector512<double> bRow_hi = LoadPackedB(bPtr, kk, lo: false);

    acc0  = Avx512F.FusedMultiplyAdd(Vector512.Create(packedA[kk*8+0]), bRow_lo, acc0);
    acc1  = Avx512F.FusedMultiplyAdd(Vector512.Create(packedA[kk*8+0]), bRow_hi, acc1);
    acc2  = Avx512F.FusedMultiplyAdd(Vector512.Create(packedA[kk*8+1]), bRow_lo, acc2);
    // ... 16 FMAs total per K-step ...
    acc15 = Avx512F.FusedMultiplyAdd(Vector512.Create(packedA[kk*8+7]), bRow_hi, acc15);
}
```

### Unroll factor

FMA latency is 4 cycles on Skylake-X, 3 on Zen 4. Unroll the K-loop by 4 to hide latency — issue 4 independent FMA chains, accumulate into the same `acc` register at the end. RyuJIT's register allocator handles this with `[MethodImpl(AggressiveInlining)]`.

### Software prefetch

Each K-block end prefetches the next Kc-block of A and B (4-8 cache lines ahead):

```csharp
Sse.Prefetch0(aPtr + nextKcOffsetA);
Sse.Prefetch1(aPtr + nextKcOffsetA + 64);
Sse.Prefetch0(bPtr + nextKcOffsetB);
```

### Edge / tail handling

When M ≢ 0 (mod Mr) or N ≢ 0 (mod Nr), run "edge microkernels" with masked stores:
- AVX-512: native mask registers (`k0`..`k7`) — one masked-store instruction.
- AVX2: `Avx2.MaskStore` with precomputed mask vector.
- Scalar tail: plain loop.

### Epilogue plumbing

Microkernel's final step (after K-loop ends) optionally applies the fused chain before storing:

```
acc → + bias[n..n+Nr] → activation(acc) → + skip[m..m+Mr, n..n+Nr] →
       × dropoutMask (training) → × outputScale → store to C[m, n]
```

Each stage branches on a precomputed `EpilogueFlags` bitmask. The non-fused case (`flags == 0`) is the hot path — no branches in the inner loop, just K-loop + store.

## 5. Packing + cache hierarchy

Three-level Goto-algorithm loop nest that matches L1/L2/L3 cache sizes:

```
for jc = 0 to N step Nc:                          // L3-resident B panel
    for pc = 0 to K step Kc:                      // L2-resident A panel
        PackB(B[pc:pc+Kc, jc:jc+Nc]) → Bp        // [Kc × Nc] packed
        for ic = 0 to M step Mc:                  // M outer block
            PackA(A[ic:ic+Mc, pc:pc+Kc]) → Ap    // [Mc × Kc] packed
            for jr = 0 to Nc step Nr:             // N-direction microkernel
                for ir = 0 to Mc step Mr:         // M-direction microkernel
                    Microkernel(Ap[ir, :], Bp[:, jr], C[ic+ir, jc+jr])
```

### Per-arch blocking defaults (Mc, Nc, Kc for FP64)

| Arch | L1d | L2 | L3 | Mc | Nc | Kc |
|---|---|---|---|---|---|---|
| Skylake-X / Sapphire Rapids | 32 KB | 1 MB | 35-77 MB | 240 | 3072 | 240 |
| Zen 4 | 32 KB | 1 MB | 32-96 MB | 144 | 4080 | 256 |
| Zen 3 / Cascade Lake | 32 KB | 512 KB | 32-72 MB | 96 | 2040 | 192 |
| Apple M-series (Neon) | 128 KB | 12-24 MB | n/a | 80 | 1024 | 240 |
| Generic AVX2 host | 32 KB | 256 KB | 8-32 MB | 96 | 1024 | 128 |

FP32 doubles each value. Defaults from BLIS's published per-arch parameter tables; shipped in `BlasManaged/BlockingDefaults.cs`. Arch detection via existing `CpuFeatures`.

### Packed-A layout

`Mc × Kc` block reordered into Mr-row stripes, K-contiguous within each stripe:

```
Ap[stripe, k, row_within_stripe] = A[ic + stripe*Mr + row_within_stripe, pc + k]
where stripe ∈ [0, Mc/Mr), row_within_stripe ∈ [0, Mr), k ∈ [0, Kc).
```

Microkernel reads one stripe's worth — Mc/Mr stripes per Mc panel. Stride is 1 in K → unit-stride loads. 64-byte alignment for AVX-512.

### transA baked into PackA

When `transA = true`, logical view is `A[K, M]` but stored `A[M, K]`. PackA reads `A[pc:pc+Kc, ic+row]` (M-stride instead of K-stride) and writes the same stripe layout. The microkernel never knows the difference.

**This is what closes the L2 pathology**: MKL's transA=true slow path is bandwidth-bound by the transposed access pattern at the microkernel level. By doing the transposed access ONCE during cache-blocked SIMD pack, we amortize across `Mc/Mr × Nc/Nr × Kc` microkernel invocations.

### Pack-B layout

`Kc × Nc` block in Nr-column stripes:

```
Bp[stripe, k, col_within_stripe] = B[pc + k, jc + stripe*Nr + col_within_stripe]
```

`transB` absorbed identically.

### Pack routine

- Each pack stripe is independent — parallelizable on the same thread pool as the GEMM.
- AVX-512: transposes 8×8 FP64 tiles via `vshufps + permutexvar` in ~20 cycles per tile.
- AVX2: 4×4 FP64 tiles via 4 unpack instructions per tile.
- ARM64 Neon: 4×4 FP32 tiles via `vtrn1q_f32 / vtrn2q_f32`.
- 64-byte aligned. Pack buffers oversized to handle Mc-tail / Kc-tail without bounds checks.

### Cache-fit verification

At dispatcher entry:
```
sizeof(Ap) = Mc × Kc × sizeof(T)   must fit in L2  (target ≤ 0.6 × L2)
sizeof(Bp[stripe]) = Kc × Nr × sizeof(T)   must fit in L1  (target ≤ 0.4 × L1)
```

If `BlasOptions.Workspace` can't hold these, dispatcher shrinks Mc / Kc until they do (autotune persists the shrunk values).

### Streaming kernel (no-pack variant)

When dispatcher picks `Streaming` mode (small K or in-L1 shape):
- No PackA / PackB. Microkernel reads A and B in native stride.
- Separate microkernel variants per `(transA, transB)` combination — 4 total: `NN, TN, NT, TT`. Hand-written; no shared code with the packed path.
- Used for `K < 32` shapes where pack overhead exceeds GEMM time.

## 6. Parallelism + determinism

Three axes (M, N, K) + 2D grid + sequential — autotune picks per shape.

### Axis selection

| Axis | Mechanism | Wins when | Deterministic? |
|---|---|---|---|
| Sequential | All loops on one thread | M·N·K < ParallelWorkThreshold | Yes |
| M | Outer `ic` split, each thread owns disjoint Mc-block of C rows | M ≥ threadCount × Mr × 2 | Yes — disjoint C writes |
| N | Outer `jc` split, each thread owns disjoint Nc-block of C cols | M small, N large (attention QKᵀ at small batch) | Yes — disjoint C writes |
| K | Outer `pc` split with reduction tree | M and N small but K large (tall-thin) | Only with deterministic tree-reduce |
| MN_2D | (M, N) flattened to 1D work-item index | M·N ≥ procs × Mr × Nr × 4 | Yes — disjoint C writes |

Default heuristic (used until autotune cache warms):

```csharp
if (m * n * k < ParallelWorkThreshold) return Axis.None;
if (m >= procs * Mr * 2 && (k <= 256 || !deterministic)) return Axis.M;
if (n >= procs * Nr * 2) return Axis.N;
if (k >= 512 && !deterministic) return Axis.K;
if (m * n >= procs * Mr * Nr * 4) return Axis.MN_2D;
return Axis.M;
```

### M-axis split (L2 case)

```
M=4096 across 16 threads → each owns Mc-block = 256 rows.
Each thread runs the full (jc, pc, ir, jr microkernel) inner nest.
Pack-A done per-thread (own buffer); Pack-B shared (done once before fork-join).
C writes disjoint; no synchronization.
```

Uses existing `CpuParallelSettings.ParallelForOrSerial` (same scheduler as the rest of the codebase).

### K-axis split (tall-thin shapes)

```
K split across N threads → each handles a Kc-range.
Each thread accumulates into private Cp[Mc, Nc] partial C from per-thread workspace.
After all threads done, reduction tree sums Cp_0 + Cp_1 + ... → C in fixed pairwise order.
```

Reduction tree topology is shape-independent — same accumulation order regardless of completion order → deterministic. Cost: `log₂(threads)` extra Mc·Nc write+read passes. Worth it when K-split throughput gain exceeds reduction cost (autotune learns this).

### Determinism mode (`BlasProvider.SetDeterministicMode(true)`)

1. K-axis effectively disabled (threadCount=1 for K-split → no reduction tree).
2. Thread scheduler uses fixed work-item → thread binding (no work-stealing). New `CpuParallelSettings.DeterministicScheduling: bool` flag.
3. Microkernel FMA order unchanged (already deterministic by construction).
4. Pack-A / Pack-B output bit-identical across thread counts.
5. Autotune fixed to "use cached choice only"; first-call benchmarking disabled.

Net: bit-identical output across threadCount = 1..N, at 10-20% throughput cost vs non-deterministic.

### Single-thread path

When `options.NumThreads == -1` or `deterministic && K_split_picked` or `procs == 1`:
- All four axis-split branches collapse to sequential outer loops.
- Same code path as today's `SimdGemm.SgemmSequential`.

## 7. Allocator + JIT microkernel cache

### Layer 1: per-thread persistent pool

```csharp
internal sealed class PerThreadPool {
    [ThreadStatic] private static PerThreadPool? _instance;
    public static PerThreadPool Current => _instance ??= new();

    private byte[]? _packA;   // sized Mc_max × Kc_max × 8 (FP64)
    private byte[]? _packB;
    private byte[]? _kSplitC; // lazy alloc, only when K-axis active

    public Span<byte> RentPackA(int bytes);
    public Span<byte> RentPackB(int bytes);
    public Span<byte> RentKSplitC(int bytes);
}
```

`[ThreadStatic]` → zero contention. Buffers grow monotonically; ~700 KB per thread at Skylake defaults. 16 threads = 11 MB total. Allocation happens once per thread, on first call.

### Layer 2: ArrayPool overflow

When shape's `(Mc × Kc) > PerThreadPool.MaxPackABytes` (large LLM-scale shapes), fall back to `ArrayPool<byte>.Shared.Rent` with `try/finally Return`.

### Layer 3: Weight pre-pack cache

```csharp
public sealed class WeightPackHandle : IDisposable {
    internal byte[] PackedBuffer;
    internal long Version;
    internal (int Mc, int Kc, bool TransA, PackingMode Mode, Type ElemType) Key;
    public void MarkDirty() => Interlocked.Increment(ref Version);
    public void Dispose() => /* return to pool */;
}
```

- Handle wraps a packed weight buffer; the caller (layer `Initialize()` or compiled-plan setup) creates it once.
- Dispatcher reads `options.PackedA?.Version`; if matches cached "last-packed-version", skip pack step.
- `MarkDirty()` called by fused-optimizer-step path on weight mutation; next call re-packs.
- Inference (no optimizer): handle never dirty → packed once forever.

**This is the PyTorch-surpassing layer.** PyTorch CPU re-packs every call; we pack once per weight per training run.

### Layer 4: TensorArena integration

When `TensorArena.IsActive`, pack buffers allocated from the arena instead of per-thread pool. Frees at iteration end. Falls through to layer 1 when arena inactive.

### Layer 5: caller-supplied workspace

```csharp
public struct BlasOptions {
    public Span<byte> Workspace;
}
```

When `Workspace.Length >= requiredBytes`, dispatcher carves pack-A / pack-B / partial-C sub-spans out of it. Zero allocation in the kernel call. Use cases:
- Benchmarks wanting reproducible alloc-free measurements
- Inference servers with NUMA-local / hugepage memory pools
- AOT-published binaries with predictable memory cost

If supplied workspace too small, fall back to layer 1.

### JIT microkernel cache

`BlasManaged/Jit/JittedKernelCache.cs`:

```csharp
internal static class JittedKernelCache {
    private static readonly ConcurrentDictionary<KernelKey, Delegate> _cache = new();

    public static Action<...>? TryGetJittedKernel(KernelKey key);
    public static Action<...>  GetOrEmit(KernelKey key, Func<KernelKey, Action<...>> emit);
    public static void Clear();   // for testing

    public struct KernelKey : IEquatable<KernelKey> {
        public int M, N, K;
        public int Lda, Ldb, Ldc;
        public bool TransA, TransB;
        public PackingMode Packing;
        public byte EpilogueFlags;
        public Type ElemType;
        public CpuArch Arch;
    }
}
```

**Emission:**
- Builds `DynamicMethod` walking the BLIS loop nest
- Bakes M/N/K/strides as `Ldc_I4` constants → JITted x64 has no parameter loads
- Unrolls K-loop fully when Kc ≤ 32; ×4 otherwise
- Direct calls to `Avx512F.FusedMultiplyAdd(...)`, `Avx.LoadVector256(...)`, etc., via `MethodInfo`
- Skips edge/tail branches when shape is mod-Mr / mod-Nr
- Inlines epilogue chain based on `EpilogueFlags` — no per-call branches
- Emitted delegate takes raw `IntPtr` for buffers — no managed-array bounds checks

**Cost:**
- Emit + JIT: ~1-3 ms per shape (one-off)
- Memory: ~5-15 KB IL per shape
- LRU eviction at 64 MB total emitted IL (configurable via `BlasOptions.MaxJitCacheBytes`)
- Per-process; not persisted

**When fired:**
- After 3+ calls to same shape, dispatcher fires emit on background `Task.Run`
- First calls use hand-written kernel; later calls hit the JIT cache
- One-off shapes never trigger emit

**NativeAOT / Mono fallback:**
- `RuntimeFeature.IsDynamicCodeSupported == false` → JIT layer is a no-op
- Hand-written kernels carry full functionality (5-15% slower on hot shapes)

**Determinism:** JIT-emitted kernels are bit-identical to hand-written ones by construction (same instruction sequence). Determinism mode does not need to disable JIT.

## 8. Autotune integration

Reuses the existing `AutotuneCache` infrastructure (currently used by `SimdGemm.ResolveParallelFromAutotune`).

**Autotune key:** `(M, N, K, transA, transB, precision, arch, hasEpilogue, MachineId)`
**Cached value:** `(strategy, Mc, Nc, Kc, threadAxis, threadCount, dispatchedToJit)`

**First call** to a shape:
- Runs 3-5 strategy candidates back-to-back (10-30 ms total benchmarking overhead)
- Picks fastest
- Persists to `AutotuneCache` (which already has disk persistence via `BuiltInCatalog`)

**Subsequent calls:**
- Cache hit → use cached tuple directly
- After 3+ hits → background JIT emit fires

**MachineId in key:** prevents a cached choice on one CI host from being applied to another with different microarchitecture. Derived from `(CpuFeatures.VendorString, CpuFeatures.FamilyModelStepping, Environment.ProcessorCount, RuntimeInformation.OSArchitecture)`. Stable across reboots; identical across identical hosts; differs across heterogeneous CI runners.

**Override via `BlasOptions.PackingMode = DisableAutotune`:** forces a specific strategy, bypasses cache lookup. Used by perf-debugging and reproducibility tests.

## 9. Caller migration

All ~30 existing callers re-point in this PR. Migration is mechanical — replace call site with `BlasManaged.Gemm<T>(...)` plus relevant `BlasOptions`.

### Caller inventory

| Bucket | Files | Re-point notes |
|---|---|---|
| Direct GEMM (MatMul, BatchMatMul) | `CpuEngine.cs` MatMul paths | `BlasOptions.Default` — autotune learns the shape over iterations |
| Conv2D forward / backward | `CpuEngine.cs`, `Im2ColHelper.cs`, `FusedConvHelper.cs` | `BlasOptions.PackedA` when weights stable; `Epilogue` for fused activation+bias |
| ConvTranspose2D forward / backward | `Im2ColHelper.TryConvTranspose2DWithGemm` (both overloads) | **Delete the existing transA-vs-pre-transpose heuristic at lines 1129, 1255** — `BlasManaged.Gemm` autotune handles it. This is the L2 fix. |
| Attention | `FlashAttention.cs`, `BackwardFunctions.cs` (attention bwd) | `Epilogue.Activation = None` (softmax stays separate — needs full Q·Kᵀ output for max-subtract). The S·V output projection can fuse dropout mask + output scale. |
| Fused multi-layer | `FusedMultiLayerGemm.cs`, `FusedMultiLayerBackward.cs` | Use `PrePackA` handles per layer-weight; pass same handle into each call. **Biggest perf win lands here.** |
| BLAS fallback wrappers | `BlasProvider.TryGemm`, `TryGemmEx` | Thin shims: native BLAS when available; route to `BlasManaged.Gemm` instead of `SimdGemm.Sgemm` for fallback |
| Decomposition utilities | `SvdDecomposition.cs`, `MatrixMultiplyHelper.cs` | `BlasOptions.Default` |
| Compiled-plan inliners | `CompiledTrainingPlan.cs`, `BackwardCSEPass.cs` | `BlasOptions.PackedA/B` from compiled-plan's weight-handle table |

### Files deleted

None. Existing files retain shim bodies. Net new files: `src/AiDotNet.Tensors/Engines/BlasManaged/*` (~25-30 new files).

## 10. Acceptance criteria

### Gate 1: L2 shape ≤ 1 ms FP64 (closes #358)

- New benchmark: `tests/AiDotNet.Tensors.Benchmarks/BlasManagedBenchmarks.ConvTranspose2D_L2_Shape`
- Runs `BlasManaged.Gemm<double>(M=4096, N=16, K=512, transA=true)` post-warmup, 100-iter median
- Thresholds:
  - x64 AVX-512: ≤ 1 ms
  - AVX2-only host: ≤ 5 ms
  - ARM64 Neon: ≤ 5 ms

### Gate 2: No regressions on existing benchmarks

- Existing suite covers: MatMul (5 shapes), Conv2D fwd (3 shapes), Conv2D bwd, attention QKᵀ, FlashAttention
- CI step: run each benchmark before and after kernel swap; assert median time within 5% of baseline (or faster)
- Baselines committed alongside kernel: `tests/AiDotNet.Tensors.Benchmarks/baselines/preBlasManaged.json`

### Gate 3: Bit-exact in deterministic mode

- `BlasManagedDeterminismTests.cs`
- For each `(precision, arch, shape)` triple in a 12-shape representative set:
  - Run with `SetDeterministicMode(true)` at `threadCount = 1, 2, 4, 8, 16`
  - Assert all five outputs bit-identical (`MemoryExtensions.SequenceEqual` on raw bytes)
- Non-deterministic mode: assert `maxDiff < 1e-9` (FP64) / `< 1e-3` (FP32) vs deterministic single-thread reference

### Gate 4: DCGAN test ≤ 60 s

- Test lives in sibling `AiDotNet` repo
- Paired companion PR enables the test with 60 s budget
- Local repro via `dotnet test` in sibling repo against this branch's `AiDotNet.Tensors` build
- This PR merges when sibling PR's CI is green on this branch

### Correctness tests (in addition to perf gates)

- **Per-microkernel unit tests.** For each `(arch, precision, packing, transA, transB)`, 50+ generated shapes (N=1..N=64 to exercise tails). Bit-exact for `Streaming` at threadCount=1; `< 1 ULP` for parallel modes.
- **Fused epilogue tests.** Each `Epilogue` combination compared against unfused two-pass reference. Includes dropout × training ordering.
- **JIT cache correctness.** Same shape 10 times → call 1 (hand-written) and call 4+ (JIT-emitted) produce bit-identical output.
- **NativeAOT smoke test.** Small AOT-published app calls `BlasManaged.Gemm` on a half-dozen shapes; confirms JIT layer bypassed cleanly.
- **Pack-cache invalidation test.** Allocate handle, run GEMM, mutate weight, `MarkDirty()`, run again — second result must reflect mutated weight.

## 11. Risks + mitigations

| Risk | Mitigation |
|---|---|
| 12-week project missing #358's resolution window | Land `BlasProvider.TryGemmEx` shape-detect fix as tiny Phase 0 prelude — when L2-class shape detected, fall through to existing managed `SimdGemm` path (~100 ms vs 215 ms). Unblocks DCGAN in days while BLIS work proceeds. |
| Regression in any of ~30 caller sites | Per-caller benchmark in Gate 2; baseline JSON committed; CI rejects > 5% regression. |
| AOT/Mono breakage | Gate 3 + 4 only require non-AOT; NativeAOT smoke test catches fallback path. Document AOT as "supported, ~10% slower". |
| Autotune cache poisoning across CI runs | `AutotuneCache` already per-CI-arch keyed; add `MachineId` to key. |
| Multi-week PR review burden | Land in atomic commits per layer (microkernel, pack, allocator, JIT, dispatcher, caller migration). PR description has per-commit roadmap. |
| `DynamicMethod` JIT cost on first call to hot shape | Background `Task.Run` for emit. First-call latency unaffected. |
| RyuJIT register spilling at 16 active accumulators | Validate via `[MethodImpl(AggressiveInlining)]` + check JITted assembly with `DOTNET_JitDisasm` at dev time. Fall back to 8 accumulators (smaller microkernel) per arch if needed. |
| ARM64 Neon test coverage in CI | Existing CI runs Linux x64 only. Add an ARM64 GitHub-Actions runner job (`ubuntu-latest-arm64`) — modest budget cost; required by Gate 1's ARM threshold. |
| Weight pre-pack cache invalidation bugs | Test that `MarkDirty()` is called by every code path that mutates weights. Add an `[Conditional("DEBUG")]` write-guard that hashes the source weight tensor on pack and re-hashes on use; mismatch with cached version throws. Production builds skip the hash check (zero overhead). |

## 12. Approaches considered + rejected

Three alternatives were considered before committing to full BLIS-style:

**Approach A: Specialized small-N transA microkernel only.** Just enough to fix L2 — a single hand-rolled kernel for `M ≫ N, K medium, transA=true`. Dispatched from `Im2ColHelper.TryConvTranspose2DWithGemm` when L2-class shape detected. Multi-day effort. **Rejected** because user prioritized hardware peak across the codebase, not just for one shape.

**Approach B: Add transA support to existing `Avx512Sgemm` + `SimdGemm.Dgemm`.** Extend the 3000-LOC existing managed-GEMM files with proper transA paths. **Rejected** because the existing kernels lack BLIS-style packing, so transA support alone wouldn't hit peak — the cache hierarchy work is needed regardless.

**Approach C: Specialized first, then promote to general (phased).** Land Approach A as Phase 1 (unblocks #358 in days); Phase 2 refactors it into the general SimdGemm transA path. **Rejected** because user wanted single-PR delivery and committed to full scope upfront.

**Selected: Approach D — full BLIS-style kernel replacing both `Avx512Sgemm` and `SimdGemm`.** Scope captured in Sections 3-9 above.

## 13. Open questions

None at spec-review time. Resolution captured in Section 14.

## 14. Decisions summary

| # | Question | Answer |
|---|---|---|
| 1 | Perf target | Hardware peak (~0.5 ms FP64 L2) |
| 2 | Kernel scope | Full BLIS-style GEMM, one PR |
| 3 | Replace vs coexist with existing kernels | Replace `Avx512Sgemm` + `SimdGemm` bodies; shims kept one release cycle |
| 4 | Precision priority | FP64 first, FP32 second, both at peak |
| 5 | Determinism | Configurable via `SetDeterministicMode` |
| 6 | CPU targets | AVX-512 + AVX2 + scalar + ARM64 Neon (all four) |
| 7 | Packing strategy | 3 kernels (Pack-Both, Pack-A, Streaming) + autotune cache + user knob |
| 8 | Parallelism | Full 2D + K-split with autotune; det mode forces det-tree reduce |
| 9 | Workspace | 5-layer allocator + JIT-compiled microkernels per shape |
| 10 | Caller migration | Re-point all ~30 callers explicitly |
| 11 | Fused epilogue | Full chain: bias + activation + skip + dropout + output-scale |
| 12 | Release phasing | One PR with everything |
| 13 | Acceptance gates | L2 ≤ 1 ms + no regressions + bit-exact det mode + DCGAN ≤ 60 s |
