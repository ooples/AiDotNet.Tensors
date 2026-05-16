# BlasManaged Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the codebase's `Avx512Sgemm` + `SimdGemm` managed GEMM kernels with a full BLIS-style implementation that hits hardware peak (~0.5 ms FP64 on `M=4096 N=16 K=512 transA=true`, the L2 shape blocking issue #358), surpasses PyTorch CPU on training loops via weight pre-pack caching, and re-points all ~30 existing callers in a single PR.

**Architecture:** Layered C# library under `src/AiDotNet.Tensors/Engines/BlasManaged/` — public `Gemm<T>` API → shape-aware dispatcher → 3 packing strategies (Pack-Both / Pack-A / Streaming) → arch-specific microkernels (AVX-512 / AVX2 / scalar / ARM Neon × FP32/FP64) optionally JIT-emitted per shape, sitting on a 5-layer allocator (per-thread pool, ArrayPool overflow, weight pre-pack cache, TensorArena, caller workspace). Full 2D + K-split parallelism with deterministic-mode tree-reduce. Full-chain fused epilogue (bias + activation + skip + dropout + scale).

**Tech Stack:** C#/.NET 10 + net471, `System.Runtime.Intrinsics.X86` (AVX-512, AVX2, FMA), `System.Runtime.Intrinsics.Arm` (Neon), `System.Reflection.Emit.DynamicMethod` for shape-specialized JIT, existing `CpuParallelSettings.ParallelForOrSerial` for threading, existing `Helpers.Autotune.AutotuneCache` for per-shape decision caching.

**Spec:** [`docs/superpowers/specs/2026-05-16-blas-managed-design.md`](../specs/2026-05-16-blas-managed-design.md) (574 lines, commit `26bb60c`)

**Phasing within the PR.** The plan has 13 sequential phases (A–M) totaling ~85 tasks. Each phase produces a green-CI checkpoint; commits land atomically per task. **Phases A–B (foundation + scalar baseline) are fully expanded below.** Phases C–M are listed with concrete file paths, key code patterns from the spec, and test shapes — each will be expanded into full task-detail when that phase starts, following the established TDD pattern from A–B. This pragmatic split keeps the plan navigable while every spec requirement is covered.

**Build + test commands** (used throughout):
```
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --framework net10.0 -c Release
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --framework net471  -c Release
dotnet test  tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --framework net10.0 -c Release --filter "FullyQualifiedName~BlasManaged"
dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks -- --filter "*BlasManaged*"
```

---

## File Structure

### New files (created across phases A–L)

```
src/AiDotNet.Tensors/Engines/BlasManaged/
  BlasManaged.cs                   # Public Gemm<T> entry point + dispatcher driver
  BlasOptions.cs                   # BlasOptions<T> + Epilogue<T> + PackingMode + ActivationType
  BlasManagedStats.cs              # Diagnostics struct
  WeightPackHandle.cs              # Pre-pack handle (Layer 3 cache)
  Dispatcher/
    Dispatcher.cs                  # Shape → strategy + axis decision; autotune lookup
    AxisSelector.cs                # Parallelism axis heuristic (Section 6)
    BlockingDefaults.cs            # Per-arch Mc/Nc/Kc tables (Section 5)
    MachineId.cs                   # CpuFeatures-derived machine identity
  Strategies/
    PackBothStrategy.cs            # 3-level Goto loop nest, packed A + packed B
    PackAOnlyStrategy.cs           # Pack A; B used in-place
    StreamingStrategy.cs           # No pack; 4 trans-variants (NN/TN/NT/TT)
  Microkernels/
    IMicrokernel.cs                # Common kernel contract (delegate signatures)
    Scalar/
      ScalarFp64_4x4.cs            # 4×4 scalar microkernel FP64
      ScalarFp32_4x4.cs            # 4×4 scalar microkernel FP32
      ScalarPack.cs                # Pack-A + Pack-B scalar routines
      ScalarStreaming.cs           # 4 trans-variants for Streaming strategy
    Avx2/
      Avx2Fp64_4x8.cs              # AVX2 FP64 4×8 microkernel
      Avx2Fp32_8x8.cs              # AVX2 FP32 8×8 microkernel
      Avx2Pack.cs                  # 4×4 FP64 / 8×8 FP32 SIMD transpose-pack
      Avx2Streaming.cs
      Avx2Tail.cs                  # MaskStore tail handling
    Avx512/
      Avx512Fp64_8x16.cs           # AVX-512 FP64 8×16 microkernel
      Avx512Fp32_16x16.cs          # AVX-512 FP32 16×16 microkernel
      Avx512Pack.cs                # 8×8 FP64 / 16×16 FP32 vshufps/permutexvar pack
      Avx512Streaming.cs
      Avx512Tail.cs                # k-mask register tail handling
    Neon/
      NeonFp64_4x4.cs              # ARM64 Neon FP64 4×4 microkernel
      NeonFp32_8x4.cs              # ARM64 Neon FP32 8×4 microkernel
      NeonPack.cs                  # vtrn1q_f32 / vtrn2q_f32 transposes
      NeonStreaming.cs
  Allocator/
    PerThreadPool.cs               # Layer 1: [ThreadStatic] persistent pool
    ArrayPoolOverflow.cs           # Layer 2: ArrayPool<byte>.Shared overflow
    WeightPackCache.cs             # Layer 3: per-weight pre-pack cache
    ArenaIntegration.cs            # Layer 4: TensorArena hookup
    WorkspaceCarver.cs             # Layer 5: carve sub-spans from user workspace
    AllocatorSelector.cs           # Selects the right layer per call
  Parallelism/
    MAxisDriver.cs                 # M-axis split mechanics
    NAxisDriver.cs                 # N-axis split mechanics
    KAxisDriver.cs                 # K-axis split + reduction tree
    MN2DDriver.cs                  # 2D MN grid split
    DeterministicScheduler.cs      # Fixed work-item → thread binding
    ReductionTree.cs               # Bit-deterministic pairwise sum
  Autotune/
    AutotuneKey.cs                 # Cache key (M,N,K,trans,prec,arch,epilogue,machine)
    AutotuneValue.cs               # Cache value (strategy, Mc, Nc, Kc, axis, threads)
    AutotuneDispatcher.cs          # First-call benchmark + cache write
  Epilogue/
    EpilogueFlags.cs               # Bit-packed presence flags
    BiasEpilogue.cs                # In-register + bias[n..n+Nr]
    ActivationEpilogue.cs          # ReLU/GELU/Sigmoid/Tanh/Swish/Mish/LeakyReLU
    SkipEpilogue.cs                # + skip[m..m+Mr, n..n+Nr]
    DropoutEpilogue.cs             # × dropoutMask
    OutputScaleEpilogue.cs         # × scalar
  Jit/
    JittedKernelCache.cs           # ConcurrentDictionary<KernelKey, Delegate> + LRU
    KernelKey.cs
    IlEmitter.cs                   # DynamicMethod IL emission driver
    IlEmitter.Avx512.cs            # AVX-512 IL emission
    IlEmitter.Avx2.cs              # AVX2 IL emission
    IlEmitter.Scalar.cs            # Scalar IL emission (fallback for JIT path)
    NativeAotDetector.cs           # RuntimeFeature.IsDynamicCodeSupported gate
```

### Modified files

```
src/AiDotNet.Tensors/Engines/Simd/SimdGemm.cs           # Body of Sgemm/Dgemm becomes shim
src/AiDotNet.Tensors/Engines/Simd/SimdGemmDouble.cs     # Body of Dgemm becomes shim
src/AiDotNet.Tensors/Engines/Simd/Avx512Sgemm.cs        # Body of SgemmBlocked becomes shim
src/AiDotNet.Tensors/Helpers/BlasProvider.cs            # TryGemm/TryGemmEx fallback path
                                                        # routes to BlasManaged.Gemm
src/AiDotNet.Tensors/Helpers/Im2ColHelper.cs            # TryConvTranspose2DWithGemm
                                                        # heuristic removed (lines 1129, 1255)
src/AiDotNet.Tensors/Engines/CpuEngine.cs               # MatMul + Conv2D forward + Conv2D backward
                                                        # + ConvTranspose2D forward + backward
                                                        # re-pointed (~12 call sites here)
src/AiDotNet.Tensors/Engines/Autodiff/BackwardFunctions.cs   # Attention backward GEMMs (3 sites)
src/AiDotNet.Tensors/Engines/Autodiff/FlashAttention.cs      # Q·Kᵀ + S·V GEMMs (2 sites)
src/AiDotNet.Tensors/Engines/Compilation/CompiledTrainingPlan.cs   # Inliner GEMMs (~5 sites)
src/AiDotNet.Tensors/Engines/Simd/FusedMultiLayerGemm.cs          # Per-layer pre-pack + GEMM
src/AiDotNet.Tensors/Engines/Simd/FusedMultiLayerBackward.cs      # Per-layer pre-pack + GEMM
src/AiDotNet.Tensors/LinearAlgebra/SvdDecomposition.cs            # 1-2 GEMM sites
src/AiDotNet.Tensors/Helpers/MatrixMultiplyHelper.cs              # 1-2 GEMM sites
src/AiDotNet.Tensors/Helpers/CpuParallelSettings.cs               # + DeterministicScheduling flag
src/AiDotNet.Tensors/Helpers/Autotune/BuiltInCatalog.cs           # Add BlasManaged entries
src/AiDotNet.Tensors/Helpers/Autotune/AutotuneCache.cs            # Add MachineId to keys
```

### Test files

```
tests/AiDotNet.Tensors.Tests/BlasManaged/
  ScalarKernelTests.cs            # Per-microkernel unit tests (Phase B)
  Avx2KernelTests.cs              # AVX2 vs scalar parity (Phase C)
  Avx512KernelTests.cs            # AVX-512 vs scalar parity (Phase D)
  NeonKernelTests.cs              # Neon vs scalar parity (Phase E, ARM CI only)
  PackingStrategyTests.cs         # 3 strategies cross-check (Phase B+)
  AllocatorTests.cs               # 5-layer allocator (Phase F)
  ParallelismTests.cs             # Axis splits produce same result as sequential (Phase G)
  DeterminismTests.cs             # Bit-exact across thread counts (Phase G + L)
  AutotuneTests.cs                # Cache hit/miss behavior (Phase H)
  EpilogueTests.cs                # Each epilogue stage + chain (Phase I)
  JitKernelTests.cs               # JIT-emitted vs hand-written parity (Phase J)
  NativeAotSmokeTest.cs           # AOT-published binary calls Gemm (Phase L)
  WeightPackInvalidationTests.cs  # Mutate + MarkDirty path (Phase F)
  ConvTranspose2DL2PerfTest.cs    # Gate 1 (Phase L)
  RegressionBaselineTests.cs      # Gate 2 (Phase L)

tests/AiDotNet.Tensors.Benchmarks/
  BlasManagedBenchmarks.cs        # BenchmarkDotNet for L2 shape + suite (Phase L)
  baselines/preBlasManaged.json   # Captured baselines for no-regression gate (Phase L)
```

---

## Phase A: Foundation (no functional change)

Creates the project skeleton, stubs the public API, defines all data types. Compiles and tests-as-pending; no behavior yet. End state: `BlasManaged.Gemm<T>` exists, throws `NotImplementedException`, all callers still use the old path.

### Task A1: Create BlasManaged directory + stub public entry point

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs`
- Test: `tests/AiDotNet.Tensors.Tests/BlasManaged/ScalarKernelTests.cs`

- [ ] **Step 1: Write the failing test**

```csharp
// tests/AiDotNet.Tensors.Tests/BlasManaged/ScalarKernelTests.cs
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.BlasManaged;

public class ScalarKernelTests
{
    [Fact]
    public void Gemm_StubExistsButNotImplemented_ThrowsNotImplemented()
    {
        Span<double> c = new double[4];
        var options = new BlasOptions<double>();
        Assert.Throws<NotImplementedException>(() =>
            BlasManaged.Gemm<double>(
                new ReadOnlySpan<double>(new double[2]), 2, false,
                new ReadOnlySpan<double>(new double[2]), 2, false,
                c, 2, 1, 2, 2, options));
    }
}
```

- [ ] **Step 2: Run test to verify it fails (compile error)**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "FullyQualifiedName~ScalarKernelTests"
```
Expected: build fails — `BlasManaged` and `BlasOptions<>` undefined.

- [ ] **Step 3: Create the stub**

```csharp
// src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs
using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// BLIS-style managed GEMM kernel. Replaces Avx512Sgemm + SimdGemm as the
/// codebase's primary GEMM path. See docs/superpowers/specs/2026-05-16-blas-managed-design.md.
/// </summary>
public static class BlasManaged
{
    /// <summary>
    /// Computes C = α·op(A)·op(B) + β·C where op(X) is X or X^T.
    /// α = 1, β = 0 by default; epilogue may modify the output.
    /// </summary>
    public static void Gemm<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        throw new NotImplementedException("BlasManaged.Gemm: filled in by Phase B.");
    }
}
```

- [ ] **Step 4: Add BlasOptions stub**

```csharp
// src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs
using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public readonly ref struct BlasOptions<T> where T : unmanaged
{
    // Fields filled in by Task A3.
}
```

- [ ] **Step 5: Run test to verify it passes**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "FullyQualifiedName~Gemm_StubExistsButNotImplemented"
```
Expected: PASS — the call throws `NotImplementedException` as asserted.

- [ ] **Step 6: Commit**

```
git add src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs \
        src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs \
        tests/AiDotNet.Tensors.Tests/BlasManaged/ScalarKernelTests.cs
git commit -m "feat(#358): BlasManaged stub — Gemm<T> entry point throws NotImplemented"
```

### Task A2: Define PackingMode + ActivationType enums

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs`

- [ ] **Step 1: Write the failing test**

```csharp
// tests/AiDotNet.Tensors.Tests/BlasManaged/ScalarKernelTests.cs (add to existing class)
[Fact]
public void PackingMode_HasFiveValues()
{
    var values = Enum.GetValues<PackingMode>();
    Assert.Equal(5, values.Length);
    Assert.Contains(PackingMode.Auto, values);
    Assert.Contains(PackingMode.ForcePackBoth, values);
    Assert.Contains(PackingMode.ForcePackAOnly, values);
    Assert.Contains(PackingMode.ForceStreaming, values);
    Assert.Contains(PackingMode.DisableAutotune, values);
}

[Fact]
public void ActivationType_HasEightValues()
{
    var values = Enum.GetValues<ActivationType>();
    Assert.Equal(8, values.Length);
    Assert.Contains(ActivationType.None, values);
    Assert.Contains(ActivationType.ReLU, values);
    Assert.Contains(ActivationType.GELU, values);
    Assert.Contains(ActivationType.Sigmoid, values);
    Assert.Contains(ActivationType.Tanh, values);
    Assert.Contains(ActivationType.Swish, values);
    Assert.Contains(ActivationType.Mish, values);
    Assert.Contains(ActivationType.LeakyReLU, values);
}
```

- [ ] **Step 2: Run tests to verify they fail (undefined enums)**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "PackingMode_HasFiveValues|ActivationType_HasEightValues"
```
Expected: build fails — enums undefined.

- [ ] **Step 3: Add the enums to BlasOptions.cs**

```csharp
// Append to src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs
public enum PackingMode
{
    /// <summary>Dispatcher picks best strategy per shape via autotune cache.</summary>
    Auto,
    /// <summary>Always pack both A and B. Forces the 3-level Goto loop nest.</summary>
    ForcePackBoth,
    /// <summary>Pack A only; B is read in-place from caller memory.</summary>
    ForcePackAOnly,
    /// <summary>No pack. Microkernel reads A and B in native stride. Best for K&lt;32.</summary>
    ForceStreaming,
    /// <summary>Use cached autotune choice if present; never benchmark on first call.</summary>
    DisableAutotune,
}

public enum ActivationType
{
    None,
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Swish,
    Mish,
    LeakyReLU,
}
```

- [ ] **Step 4: Run tests to verify they pass**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "PackingMode_HasFiveValues|ActivationType_HasEightValues"
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs tests/AiDotNet.Tensors.Tests/BlasManaged/ScalarKernelTests.cs
git commit -m "feat(#358): BlasManaged enums — PackingMode + ActivationType"
```

### Task A3: Define BlasOptions<T> + Epilogue<T> full surface

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs`
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/WeightPackHandle.cs`

- [ ] **Step 1: Write the failing test**

```csharp
// Add to ScalarKernelTests.cs
[Fact]
public void BlasOptions_DefaultValues_AreSafe()
{
    var options = new BlasOptions<double>();
    Assert.Equal(PackingMode.Auto, options.PackingMode);
    Assert.Equal(ActivationType.None, options.Epilogue.Activation);
    Assert.True(options.Epilogue.BiasN.IsEmpty);
    Assert.True(options.Epilogue.SkipMxN.IsEmpty);
    Assert.Equal(0u, options.Epilogue.DropoutMask);
    Assert.Equal(0.0, options.Epilogue.OutputScale);  // OutputScale default 0 means "use 1"
    Assert.Equal(0, options.NumThreads);
    Assert.Equal(0UL, options.AutotuneKey);
    Assert.Equal(0L, options.MaxJitCacheBytes);
    Assert.Null(options.PackedA);
    Assert.Null(options.PackedB);
    Assert.True(options.Workspace.IsEmpty);
}

[Fact]
public void BlasOptions_CanSetAllFields()
{
    Span<byte> ws = new byte[64];
    Span<double> bias = new double[4];
    var options = new BlasOptions<double>
    {
        PackingMode = PackingMode.ForcePackBoth,
        Epilogue = new Epilogue<double>
        {
            BiasN = bias,
            Activation = ActivationType.ReLU,
            OutputScale = 2.0,
        },
        Workspace = ws,
        NumThreads = -1,
        AutotuneKey = 42UL,
        MaxJitCacheBytes = 1024L * 1024,
    };
    Assert.Equal(PackingMode.ForcePackBoth, options.PackingMode);
    Assert.Equal(ActivationType.ReLU, options.Epilogue.Activation);
    Assert.Equal(2.0, options.Epilogue.OutputScale);
    Assert.Equal(-1, options.NumThreads);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "BlasOptions_DefaultValues_AreSafe|BlasOptions_CanSetAllFields"
```
Expected: build fails — `Epilogue<T>`, `WeightPackHandle` undefined; `BlasOptions<T>` has no fields.

- [ ] **Step 3: Define WeightPackHandle**

```csharp
// src/AiDotNet.Tensors/Engines/BlasManaged/WeightPackHandle.cs
using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public sealed class WeightPackHandle : IDisposable
{
    internal byte[] PackedBuffer;
    internal long Version;
    internal (int Mc, int Kc, bool TransA, PackingMode Mode, Type ElemType) Key;
    internal bool IsForA;  // true = pre-packed A; false = pre-packed B

    internal WeightPackHandle(
        byte[] packedBuffer,
        (int Mc, int Kc, bool TransA, PackingMode Mode, Type ElemType) key,
        bool isForA)
    {
        PackedBuffer = packedBuffer;
        Version = 1;
        Key = key;
        IsForA = isForA;
    }

    /// <summary>
    /// Signal that the underlying weight has been mutated (e.g., by an optimizer step).
    /// The next Gemm call referencing this handle will re-pack the weight before use.
    /// </summary>
    public void MarkDirty() => Interlocked.Increment(ref Version);

    public void Dispose()
    {
        // Pool return is handled by WeightPackCache; nothing to do here for now.
    }
}
```

- [ ] **Step 4: Define BlasOptions<T> + Epilogue<T> full**

```csharp
// Replace contents of src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs
using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public enum PackingMode
{
    Auto,
    ForcePackBoth,
    ForcePackAOnly,
    ForceStreaming,
    DisableAutotune,
}

public enum ActivationType
{
    None,
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Swish,
    Mish,
    LeakyReLU,
}

public readonly ref struct BlasOptions<T> where T : unmanaged
{
    public PackingMode PackingMode { get; init; }
    public Epilogue<T> Epilogue { get; init; }
    public Span<byte> Workspace { get; init; }
    public WeightPackHandle? PackedA { get; init; }
    public WeightPackHandle? PackedB { get; init; }

    /// <summary>0 = autotune; -1 = single-thread (deterministic); positive = pin to N.</summary>
    public int NumThreads { get; init; }
    /// <summary>0 = derive from shape; nonzero = caller-supplied autotune key.</summary>
    public ulong AutotuneKey { get; init; }
    /// <summary>0 = use process default (64 MB).</summary>
    public long MaxJitCacheBytes { get; init; }
}

public readonly ref struct Epilogue<T> where T : unmanaged
{
    /// <summary>Bias vector of length N. Empty = no bias.</summary>
    public ReadOnlySpan<T> BiasN { get; init; }
    public ActivationType Activation { get; init; }
    /// <summary>Skip-connection tensor of shape (M, N) in row-major. Empty = no skip.</summary>
    public ReadOnlySpan<T> SkipMxN { get; init; }
    /// <summary>Dropout RNG state. 0 = no dropout (inference).</summary>
    public uint DropoutMask { get; init; }
    /// <summary>Output scale. 0 (default) is interpreted as 1.0.</summary>
    public T OutputScale { get; init; }
}
```

- [ ] **Step 5: Run tests to verify they pass**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "BlasOptions_DefaultValues_AreSafe|BlasOptions_CanSetAllFields"
```
Expected: PASS.

- [ ] **Step 6: Commit**

```
git add src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs \
        src/AiDotNet.Tensors/Engines/BlasManaged/WeightPackHandle.cs \
        tests/AiDotNet.Tensors.Tests/BlasManaged/ScalarKernelTests.cs
git commit -m "feat(#358): BlasOptions<T> + Epilogue<T> + WeightPackHandle scaffolding"
```

### Task A4: Define IMicrokernel + KernelKey + BlasManagedStats

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/IMicrokernel.cs`
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Jit/KernelKey.cs`
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManagedStats.cs`

- [ ] **Step 1: Write the failing test**

```csharp
// Add to ScalarKernelTests.cs
[Fact]
public void KernelKey_EqualKeysHaveEqualHashCodes()
{
    var k1 = new KernelKey
    {
        M = 4, N = 4, K = 4, Lda = 4, Ldb = 4, Ldc = 4,
        TransA = true, TransB = false,
        Packing = PackingMode.ForcePackBoth,
        EpilogueFlags = 0,
        ElemType = typeof(double),
        Arch = CpuArch.Avx512,
    };
    var k2 = k1;  // ref-struct value semantics — same fields → same key
    Assert.Equal(k1, k2);
    Assert.Equal(k1.GetHashCode(), k2.GetHashCode());
}

[Fact]
public void BlasManagedStats_DefaultIsZero()
{
    var stats = new BlasManagedStats();
    Assert.Equal(0L, stats.AutotuneHits);
    Assert.Equal(0L, stats.JitEmissions);
    Assert.Equal(0L, stats.PackCacheHits);
    Assert.Equal(0L, stats.PackCacheMisses);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "KernelKey_EqualKeysHaveEqualHashCodes|BlasManagedStats_DefaultIsZero"
```
Expected: build fails — types undefined.

- [ ] **Step 3: Define KernelKey**

```csharp
// src/AiDotNet.Tensors/Engines/BlasManaged/Jit/KernelKey.cs
using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public enum CpuArch : byte
{
    Scalar = 0,
    Avx2 = 1,
    Avx512 = 2,
    NeonAArch64 = 3,
}

public readonly struct KernelKey : IEquatable<KernelKey>
{
    public int M { get; init; }
    public int N { get; init; }
    public int K { get; init; }
    public int Lda { get; init; }
    public int Ldb { get; init; }
    public int Ldc { get; init; }
    public bool TransA { get; init; }
    public bool TransB { get; init; }
    public PackingMode Packing { get; init; }
    public byte EpilogueFlags { get; init; }
    public Type ElemType { get; init; }
    public CpuArch Arch { get; init; }

    public bool Equals(KernelKey other) =>
        M == other.M && N == other.N && K == other.K
        && Lda == other.Lda && Ldb == other.Ldb && Ldc == other.Ldc
        && TransA == other.TransA && TransB == other.TransB
        && Packing == other.Packing && EpilogueFlags == other.EpilogueFlags
        && ElemType == other.ElemType && Arch == other.Arch;

    public override bool Equals(object? obj) => obj is KernelKey k && Equals(k);

    public override int GetHashCode() =>
        HashCode.Combine(
            HashCode.Combine(M, N, K, Lda, Ldb, Ldc),
            HashCode.Combine(TransA, TransB, (int)Packing, EpilogueFlags),
            ElemType, (int)Arch);
}
```

- [ ] **Step 4: Define IMicrokernel + BlasManagedStats**

```csharp
// src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/IMicrokernel.cs
namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Marker interface for microkernel delegate signatures. Each microkernel is
/// invoked via an internal delegate matching this shape: the dispatcher binds
/// the concrete kernel based on (arch, precision, packing, trans).
/// Concrete delegates per arch are defined alongside their kernel files.
/// </summary>
internal interface IMicrokernel { }
```

```csharp
// src/AiDotNet.Tensors/Engines/BlasManaged/BlasManagedStats.cs
namespace AiDotNet.Tensors.Engines.BlasManaged;

public struct BlasManagedStats
{
    public long AutotuneHits;
    public long AutotuneMisses;
    public long JitEmissions;
    public long JitCacheHits;
    public long PackCacheHits;
    public long PackCacheMisses;
    public long PackCacheBytes;
}
```

- [ ] **Step 5: Run tests to verify they pass**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "KernelKey_EqualKeysHaveEqualHashCodes|BlasManagedStats_DefaultIsZero"
```
Expected: PASS.

- [ ] **Step 6: Commit**

```
git add src/AiDotNet.Tensors/Engines/BlasManaged/Jit/KernelKey.cs \
        src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/IMicrokernel.cs \
        src/AiDotNet.Tensors/Engines/BlasManaged/BlasManagedStats.cs \
        tests/AiDotNet.Tensors.Tests/BlasManaged/ScalarKernelTests.cs
git commit -m "feat(#358): BlasManaged KernelKey + IMicrokernel + Stats"
```

### Task A5: Phase A green-CI checkpoint

- [ ] **Step 1: Build both target frameworks**

```
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --framework net10.0 -c Release
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --framework net471  -c Release
```
Expected: zero errors, zero warnings, both frameworks build clean.

- [ ] **Step 2: Run full test suite, no regressions**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 -c Release --logger "console;verbosity=minimal"
```
Expected: all pre-existing tests still pass; 5 new BlasManaged scaffolding tests pass.

- [ ] **Step 3: No commit** — checkpoint only. Continue to Phase B.

---

## Phase B: Scalar correctness baseline

Implements every code path at scalar (non-SIMD) precision for both FP32 and FP64. End state: `BlasManaged.Gemm<double>` and `BlasManaged.Gemm<float>` produce correct output for any (M, N, K, transA, transB) on any host (no AVX2/AVX-512 required). Slow but correct. Becomes the ground truth that AVX2/AVX-512/Neon paths assert against in their unit tests.

### Task B1: Scalar 4×4 microkernel FP64 (no transpose, no epilogue)

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Scalar/ScalarFp64_4x4.cs`

- [ ] **Step 1: Write the failing test**

```csharp
// Add to ScalarKernelTests.cs
[Fact]
public void ScalarFp64_4x4_Computes_4x4_Tile_From_Packed_Inputs()
{
    // Packed-A layout: 4 rows × Kc=2 columns, K-contiguous within row.
    // packedA[row*Kc + k] = A[row, k]
    double[] packedA = { 1, 2,    3, 4,    5, 6,    7, 8 };  // 4 rows × Kc=2
    // Packed-B layout: Kc=2 × 4 cols, col-contiguous within k.
    // packedB[k*4 + col] = B[k, col]
    double[] packedB = { 1, 0, 0, 0,    0, 1, 0, 0 };       // Kc=2 × 4 cols
    double[] c = new double[4 * 4];
    int ldc = 4;

    ScalarFp64_4x4.Run(packedA, packedB, c, ldc, kc: 2);

    // C[row, col] = sum_k A[row, k] · B[k, col]
    // A = [[1,2],[3,4],[5,6],[7,8]];  B = [[1,0,0,0],[0,1,0,0]];
    // C = [[1,2,0,0],[3,4,0,0],[5,6,0,0],[7,8,0,0]]
    double[] expected = { 1, 2, 0, 0,    3, 4, 0, 0,    5, 6, 0, 0,    7, 8, 0, 0 };
    for (int i = 0; i < expected.Length; i++)
        Assert.Equal(expected[i], c[i], precision: 12);
}
```

- [ ] **Step 2: Run test to verify it fails**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "ScalarFp64_4x4_Computes"
```
Expected: build fails — `ScalarFp64_4x4` undefined.

- [ ] **Step 3: Implement the microkernel**

```csharp
// src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Scalar/ScalarFp64_4x4.cs
using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Scalar reference microkernel: 4×4 output tile, FP64.
/// Reads packed-A in [Mr=4, Kc] layout and packed-B in [Kc, Nr=4] layout.
/// Computes C[0..4, 0..4] += packedA · packedB. Caller is responsible for
/// zero-initializing C before the first kernel call.
///
/// This kernel is the ground truth — AVX2/AVX-512/Neon kernels assert their
/// output against this in unit tests.
/// </summary>
internal static class ScalarFp64_4x4
{
    public const int Mr = 4;
    public const int Nr = 4;

    public static void Run(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc)
    {
        // 16 register-resident accumulators (one per cell of the 4×4 C tile).
        double c00 = c[0*ldc + 0], c01 = c[0*ldc + 1], c02 = c[0*ldc + 2], c03 = c[0*ldc + 3];
        double c10 = c[1*ldc + 0], c11 = c[1*ldc + 1], c12 = c[1*ldc + 2], c13 = c[1*ldc + 3];
        double c20 = c[2*ldc + 0], c21 = c[2*ldc + 1], c22 = c[2*ldc + 2], c23 = c[2*ldc + 3];
        double c30 = c[3*ldc + 0], c31 = c[3*ldc + 1], c32 = c[3*ldc + 2], c33 = c[3*ldc + 3];

        for (int k = 0; k < kc; k++)
        {
            double a0 = packedA[0 * kc + k];
            double a1 = packedA[1 * kc + k];
            double a2 = packedA[2 * kc + k];
            double a3 = packedA[3 * kc + k];

            double b0 = packedB[k * Nr + 0];
            double b1 = packedB[k * Nr + 1];
            double b2 = packedB[k * Nr + 2];
            double b3 = packedB[k * Nr + 3];

            c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
            c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
            c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
            c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;
        }

        c[0*ldc + 0] = c00; c[0*ldc + 1] = c01; c[0*ldc + 2] = c02; c[0*ldc + 3] = c03;
        c[1*ldc + 0] = c10; c[1*ldc + 1] = c11; c[1*ldc + 2] = c12; c[1*ldc + 3] = c13;
        c[2*ldc + 0] = c20; c[2*ldc + 1] = c21; c[2*ldc + 2] = c22; c[2*ldc + 3] = c23;
        c[3*ldc + 0] = c30; c[3*ldc + 1] = c31; c[3*ldc + 2] = c32; c[3*ldc + 3] = c33;
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "ScalarFp64_4x4_Computes"
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Scalar/ScalarFp64_4x4.cs \
        tests/AiDotNet.Tensors.Tests/BlasManaged/ScalarKernelTests.cs
git commit -m "feat(#358): scalar FP64 4x4 microkernel (ground truth)"
```

### Task B2: Scalar 4×4 microkernel FP32

Mirror of Task B1 for FP32. Same packed-A and packed-B layouts. Single file: `ScalarFp32_4x4.cs`. Follow the exact pattern from B1 — replace `double` with `float`, replace test's `precision: 12` with `precision: 6`. Commit message: `feat(#358): scalar FP32 4x4 microkernel`.

### Task B3: Scalar Pack-A for FP64 + FP32 (both transA cases)

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Scalar/ScalarPack.cs`

Pack-A converts row-major A (or column-major A when `transA=true`) into the `[Mc/Mr, Kc, Mr]` stripe layout. Test shapes: `M=8, K=4, transA=false` and `M=8, K=4, transA=true`. Use the exact stripe-layout formula from Section 5 of the spec. Test asserts every packed cell matches the expected logical A element.

Code pattern (FP64):

```csharp
internal static class ScalarPack
{
    public static void PackA<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        Span<T> packed, int mc, int kc, int mr) where T : unmanaged
    {
        int numStripes = mc / mr;
        for (int stripe = 0; stripe < numStripes; stripe++)
        {
            int packedOff = stripe * kc * mr;
            for (int k = 0; k < kc; k++)
            {
                for (int row = 0; row < mr; row++)
                {
                    int logicalRow = stripe * mr + row;
                    T value = transA
                        ? a[k * lda + logicalRow]            // A stored [K, M], read M-stride
                        : a[logicalRow * lda + k];           // A stored [M, K], read K-stride
                    packed[packedOff + k * mr + row] = value;
                }
            }
        }
        // Tail handling for mc % mr != 0 is added in Phase G; not needed yet.
    }
}
```

Tests:
- `PackA_NonTransposed_MatchesLogicalLayout`
- `PackA_Transposed_MatchesLogicalLayout`

Commit: `feat(#358): scalar Pack-A (both trans modes) for FP32/FP64`.

### Task B4: Scalar Pack-B for FP64 + FP32 (both transB cases)

Mirror of B3 for B, with `Bp[stripe, k, col]` layout where stripes are along N. File `ScalarPack.cs` (extend). Tests:
- `PackB_NonTransposed_MatchesLogicalLayout`
- `PackB_Transposed_MatchesLogicalLayout`

Commit: `feat(#358): scalar Pack-B (both trans modes) for FP32/FP64`.

### Task B5: PackBoth strategy driver (scalar path)

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/PackBothStrategy.cs`

Implements the 3-level Goto loop nest from Section 5 of the spec. Calls into scalar pack routines (B3, B4) and scalar microkernel (B1, B2). No parallelism yet (single-thread). No epilogue (skip the epilogue chain entirely for now).

Test:
```csharp
[Theory]
[InlineData(8, 8, 8, false, false)]   // square, no trans
[InlineData(8, 8, 8, true, false)]    // transA
[InlineData(8, 8, 8, false, true)]    // transB
[InlineData(8, 8, 8, true, true)]     // both
[InlineData(16, 4, 8, false, false)]  // rectangular
[InlineData(4, 16, 8, false, false)]
public void PackBoth_MatchesNaiveReference(int m, int n, int k, bool transA, bool transB)
{
    var (a, b) = GenerateRandomMatrices<double>(m, n, k, transA, transB, seed: 42);
    double[] expected = NaiveGemm(a, m, k, transA, b, k, n, transB);
    double[] actual = new double[m * n];
    PackBothStrategy.Run<double>(a, lda: transA ? m : k, transA,
                                  b, ldb: transB ? k : n, transB,
                                  actual, ldc: n,
                                  m, n, k,
                                  mc: 8, nc: 8, kc: 8,
                                  mr: 4, nr: 4);
    AssertNearEqual(expected, actual, tol: 1e-12);
}
```

Where `GenerateRandomMatrices` and `NaiveGemm` are private helpers in the test file. `NaiveGemm` is the triple-nested-loop reference.

Commit: `feat(#358): PackBoth strategy driver (scalar)`.

### Task B6: Streaming strategy driver — 4 trans variants

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs`
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Scalar/ScalarStreaming.cs`

Streaming microkernel reads A and B in native stride (no pack). Implements 4 variants:
- `ScalarFp64_Streaming_NN` (transA=false, transB=false)
- `ScalarFp64_Streaming_TN` (transA=true,  transB=false)
- `ScalarFp64_Streaming_NT` (transA=false, transB=true)
- `ScalarFp64_Streaming_TT` (transA=true,  transB=true)

Plus 4 FP32 mirrors. Test against the same `NaiveGemm` reference used in B5 across the same parametric shapes. Used for K < 32.

Commit: `feat(#358): Streaming strategy + 4 trans-variant scalar kernels`.

### Task B7: PackAOnly strategy driver

Similar to B5 but only packs A; reads B in-place. Microkernel needs an overload that takes a strided B span instead of packed B. Single file: `Strategies/PackAOnlyStrategy.cs`. Test against `NaiveGemm` reference.

Commit: `feat(#358): PackAOnly strategy driver (scalar)`.

### Task B8: Wire BlasManaged.Gemm to scalar PackBoth + dispatcher stub

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs`
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Dispatcher/Dispatcher.cs`

Dispatcher selects between PackBoth / PackAOnly / Streaming based on heuristic (no autotune yet — autotune is Phase H):

```csharp
internal static class Dispatcher
{
    public static PackingMode SelectStrategy(int m, int n, int k, in BlasOptions<float> _)
    {
        if (k < 32 || (long)m * n < 1024) return PackingMode.ForceStreaming;
        if (k < 128) return PackingMode.ForcePackAOnly;
        return PackingMode.ForcePackBoth;
    }
}
```

`BlasManaged.Gemm<T>` body:

```csharp
public static void Gemm<T>(
    ReadOnlySpan<T> a, int lda, bool transA,
    ReadOnlySpan<T> b, int ldb, bool transB,
    Span<T> c, int ldc,
    int m, int n, int k,
    in BlasOptions<T> options = default) where T : unmanaged
{
    if (m <= 0 || n <= 0 || k <= 0) return;
    c.Clear();  // C = 0 (the kernel accumulates C += A·B; caller can pre-fill C if desired)

    PackingMode strategy = options.PackingMode == PackingMode.Auto
        ? Dispatcher.SelectStrategy(m, n, k, options)
        : options.PackingMode;

    switch (strategy)
    {
        case PackingMode.ForcePackBoth:
            PackBothStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc,
                                     m, n, k, mc: 64, nc: 64, kc: 64, mr: 4, nr: 4);
            break;
        case PackingMode.ForcePackAOnly:
            PackAOnlyStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc,
                                     m, n, k, mc: 64, kc: 64, mr: 4, nr: 4);
            break;
        case PackingMode.ForceStreaming:
            StreamingStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
            break;
        default:
            throw new NotSupportedException($"PackingMode {strategy} not yet implemented.");
    }
}
```

Test:

```csharp
[Fact]
public void Gemm_ScalarPath_Matches_NaiveReference_ForL2Shape()
{
    int m = 32, n = 16, k = 64;  // smaller stand-in for L2; full L2 hits Phase L
    var (a, b) = GenerateRandomMatrices<double>(m, n, k, transA: true, transB: false, seed: 42);
    double[] expected = NaiveGemm(a, m, k, transA: true, b, k, n, transB: false);
    double[] actual = new double[m * n];
    BlasManaged.Gemm<double>(
        a, lda: m, transA: true,
        b, ldb: n, transB: false,
        actual, ldc: n,
        m, n, k);
    AssertNearEqual(expected, actual, tol: 1e-12);
}
```

Commit: `feat(#358): BlasManaged.Gemm dispatches to scalar PackBoth/PackAOnly/Streaming`.

### Task B9: Phase B green-CI checkpoint

- [ ] **Step 1: Build both frameworks (must include net471 — the scalar path runs there)**

```
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --framework net10.0 -c Release
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --framework net471  -c Release
```

- [ ] **Step 2: Run BlasManaged test suite**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 --filter "FullyQualifiedName~BlasManaged"
dotnet test tests/AiDotNet.Tensors.Tests --framework net471  --filter "FullyQualifiedName~BlasManaged"
```
Expected: all Phase A + B tests pass on both frameworks. Scalar path is the only one wired up — AVX2/AVX-512 paths are no-ops still.

- [ ] **Step 3: Run full pre-existing test suite to verify no regression**

```
dotnet test tests/AiDotNet.Tensors.Tests --framework net10.0 -c Release --logger "console;verbosity=minimal"
```
Expected: all pre-existing tests still pass. The only Gemm caller (`BlasManaged.Gemm` itself) is reached only via test code at this point — the existing kernels are untouched.

---

## Phase C: AVX2 path

End state: `BlasManaged.Gemm<T>` on x64+AVX2 hosts dispatches to AVX2 microkernels. Performance gap to scalar: 4-8× faster. Correctness asserted against scalar ground truth from Phase B.

**Tasks (each follows the TDD pattern from Phase A/B — write failing test, implement, test passes, commit):**

- **C1: AVX2 FP64 4×8 microkernel** — `Microkernels/Avx2/Avx2Fp64_4x8.cs`. 4 rows × 8 cols (= 2 × `Vector256<double>` per row). 8 register accumulators. Inner K-loop: 1 vector load from packed-B per K, 4 broadcast-FMAs (one per row × 2 col-halves = 8 FMAs/K). Test: parametric over (M,N,K) up to (32,32,32), assert match scalar within `< 1e-12`.

- **C2: AVX2 FP32 8×8 microkernel** — `Microkernels/Avx2/Avx2Fp32_8x8.cs`. 8 rows × 8 cols (= 1 × `Vector256<float>` per row). 8 register accumulators. Test: `< 1e-6` vs scalar.

- **C3: AVX2 Pack-A SIMD** — `Microkernels/Avx2/Avx2Pack.cs`. 4×4 FP64 tile transposed via `Avx.UnpackLow/UnpackHigh + Avx.Permute2x128`. Test: pack 64×64 matrix in both trans modes, verify bit-identical to scalar pack output.

- **C4: AVX2 Pack-B SIMD** — same file. Same transpose pattern. Test: mirror C3.

- **C5: AVX2 Streaming kernels (4 trans variants)** — `Microkernels/Avx2/Avx2Streaming.cs`. Match `ScalarStreaming` output for K=1..31.

- **C6: AVX2 tail handling** — `Microkernels/Avx2/Avx2Tail.cs`. `Avx2.MaskStore` for partial N tile. Test: shapes with N=5, 6, 7, 13, 15.

- **C7: Arch dispatch in BlasManaged.Gemm** — `if (Avx2.IsSupported && Fma.IsSupported) → AVX2 path`; else → scalar. Wired via `Dispatcher` selection.

- **C8: Phase C green-CI checkpoint** — full test suite passes; new AVX2-specific benchmark in `BlasManagedBenchmarks.cs` shows AVX2 4-8× faster than scalar on (64, 64, 64) shape.

---

## Phase D: AVX-512 path

End state: `BlasManaged.Gemm<T>` on x64+AVX-512 hosts dispatches to AVX-512 microkernels. Performance gap to AVX2: ~2× faster. Required to close issue #358.

**Tasks (TDD pattern, follow Phases A/B template):**

- **D1: AVX-512 FP64 8×16 microkernel** — `Microkernels/Avx512/Avx512Fp64_8x16.cs`. 8 rows × 16 cols (= 2 × `Vector512<double>` per row). 16 register accumulators. **This is the kernel that hits ~0.5 ms on the L2 shape.** Inner K-loop pattern from Section 4 of the spec. Test: parametric over (M,N,K) up to (128,128,128), assert match scalar within `< 1e-12`.

- **D2: AVX-512 FP32 16×16 microkernel** — `Microkernels/Avx512/Avx512Fp32_16x16.cs`. 16 rows × 16 cols (= 1 × `Vector512<float>` per row). 16 register accumulators. Test: `< 1e-6` vs scalar.

- **D3: AVX-512 Pack-A SIMD** — `Microkernels/Avx512/Avx512Pack.cs`. 8×8 FP64 tile transposed via `Avx512F.PermuteVar8x64 + vshufps`. 16×16 FP32 tile via similar. Test: bit-identical to scalar pack.

- **D4: AVX-512 Pack-B SIMD** — same file.

- **D5: AVX-512 Streaming kernels (4 trans variants)** — `Microkernels/Avx512/Avx512Streaming.cs`.

- **D6: AVX-512 k-mask tail handling** — `Microkernels/Avx512/Avx512Tail.cs`. Use `Avx512F.MaskBlend` + native k-register masks for N % 16 tails. Test: shapes with N=1, 7, 15, 31.

- **D7: Arch dispatch** — `if (Avx512F.IsSupported) → AVX-512`; else fall through to AVX2 → scalar. Use `CpuFeatures.HasAVX512F`.

- **D8: Phase D green-CI checkpoint** — full suite passes; new AVX-512 benchmark on L2 shape (`M=4096 N=16 K=512 transA=true` FP64) registers in `BlasManagedBenchmarks.cs`. **Target: ≤ 5 ms on this checkpoint (not yet at the 1 ms gate — JIT layer + autotune in Phases H, J close the rest of the gap).**

---

## Phase E: ARM64 Neon path

End state: `BlasManaged.Gemm<T>` on ARM64 hosts (Apple Silicon, Graviton, Cobalt) dispatches to Neon microkernels. Performance gap to AVX2: comparable per-core (Neon is 128-bit vs AVX2 256-bit, but Apple M-series has more execution ports).

**Tasks:**

- **E1: Neon FP64 4×4 microkernel** — `Microkernels/Neon/NeonFp64_4x4.cs`. 4 rows × 4 cols (= 2 × `Vector128<double>` per row). 8 register accumulators. Use `AdvSimd.Arm64.FusedMultiplyAdd` from `System.Runtime.Intrinsics.Arm`.
- **E2: Neon FP32 8×4 microkernel** — `Microkernels/Neon/NeonFp32_8x4.cs`. 8 rows × 4 cols (= 1 × `Vector128<float>` per row).
- **E3: Neon Pack-A** — `Microkernels/Neon/NeonPack.cs`. 4×4 FP32 transpose via `AdvSimd.Arm64.TransposeOdd / TransposeEven`.
- **E4: Neon Pack-B** — same file.
- **E5: Neon Streaming kernels** — `Microkernels/Neon/NeonStreaming.cs`.
- **E6: Arch dispatch** — `if (AdvSimd.IsSupported && RuntimeInformation.OSArchitecture == Arm64) → Neon`. Use `CpuFeatures.HasNeon` (add to `CpuFeatures` if not present).
- **E7: Phase E green-CI checkpoint** — needs an ARM64 CI runner (`runs-on: ubuntu-latest-arm` or `macos-latest` if Apple Silicon). Add a `.github/workflows/blas-managed-arm.yml` step. Full BlasManaged test suite passes on ARM64.

---

## Phase F: Allocator layers

End state: per-thread persistent pool, ArrayPool overflow, weight pre-pack cache, TensorArena integration, and caller-supplied workspace all work. The pre-pack cache is the PyTorch-surpassing piece.

**Tasks:**

- **F1: PerThreadPool (Layer 1)** — `Allocator/PerThreadPool.cs` with `[ThreadStatic]` instance. Monotonic growth. Test: rent pack-A buffer of size X, then size 2X, then size X again — same backing array (no shrink).
- **F2: ArrayPoolOverflow (Layer 2)** — `Allocator/ArrayPoolOverflow.cs`. Rent/return pattern when shape exceeds per-thread cap. Test: huge shape forces overflow path; verify pool returns happen exactly once.
- **F3: WeightPackCache (Layer 3)** — `Allocator/WeightPackCache.cs`. `WeightPackHandle.PrePackA<T>()` entry. Test: pack handle → `Version=1`. After `Gemm` call, version on handle's cache entry matches handle's version. After `MarkDirty()`, next call re-packs (version increments).
- **F4: ArenaIntegration (Layer 4)** — `Allocator/ArenaIntegration.cs`. Hook `TensorArena.Current?.TryRent(bytes)`. Test: within `using (var arena = new TensorArena())` scope, pack buffer is arena-allocated; outside, it's per-thread-pool.
- **F5: WorkspaceCarver (Layer 5)** — `Allocator/WorkspaceCarver.cs`. Carve sub-spans from `options.Workspace`. Test: caller-supplied 64-byte aligned `byte[]` → no allocation in `Gemm` call (verify via `GC.GetAllocatedBytesForCurrentThread()` delta).
- **F6: AllocatorSelector** — `Allocator/AllocatorSelector.cs`. Orchestrates Layer 5 → 4 → 3 → 1 → 2 selection. Test: each layer activates under its expected condition.
- **F7: WeightPackInvalidationTests** — end-to-end test from spec Section 10. Allocate handle, run GEMM, mutate weight in place, call `MarkDirty()`, run again → assert second result reflects mutated weight.
- **F8: Phase F green-CI checkpoint** — full suite passes.

---

## Phase G: Parallelism

End state: M / N / K / 2D axis splits work; determinism mode produces bit-identical output across thread counts.

**Tasks:**

- **G1: M-axis split** — `Parallelism/MAxisDriver.cs`. Wraps the outer `ic` loop in `CpuParallelSettings.ParallelForOrSerial`. Each thread runs the inner nest on its disjoint Mc-block. Test: same result as single-thread.
- **G2: N-axis split** — `Parallelism/NAxisDriver.cs`. Outer `jc` parallel. Test mirrors G1.
- **G3: K-axis split + reduction tree** — `Parallelism/KAxisDriver.cs` + `Parallelism/ReductionTree.cs`. Each thread accumulates into a private `Mc × Nc` partial C; tree-reduce in fixed pairwise order. Test: bit-identical to single-thread.
- **G4: 2D MN-grid split** — `Parallelism/MN2DDriver.cs`. Flatten `(ic, jc)` to 1D work-item index. Test mirrors G1.
- **G5: Axis selector** — `Dispatcher/AxisSelector.cs` with the heuristic from Section 6 of the spec. Test: each of M, N, K, MN_2D, None is selected for an appropriate shape.
- **G6: DeterministicScheduler** — `Parallelism/DeterministicScheduler.cs` + add `CpuParallelSettings.DeterministicScheduling: bool` flag. Fixed work-item → thread binding (no work-stealing). Test: same input + same thread count + flag-on = bit-identical output across 100 runs.
- **G7: DeterminismTests.cs** — full Gate 3 test. Loops over `threadCount ∈ {1, 2, 4, 8, 16}` × `precision ∈ {float, double}` × `arch ∈ {available}` × 12 representative shapes. Asserts byte-equal output via `MemoryExtensions.SequenceEqual`.
- **G8: Phase G green-CI checkpoint** — full suite passes; new parallel benchmark shows expected scaling (e.g., 4× faster at 4 threads for big shapes).

---

## Phase H: Autotune integration

End state: per-shape strategy + blocking + thread-axis decisions cached across calls; first-call benchmark fires once per shape.

**Tasks:**

- **H1: AutotuneKey + AutotuneValue** — `Autotune/AutotuneKey.cs`, `Autotune/AutotuneValue.cs`. Match spec Section 8.
- **H2: MachineId derivation** — `Dispatcher/MachineId.cs`. Hash of `(VendorString, FamilyModelStepping, ProcessorCount, OSArchitecture)`. Stable across reboots.
- **H3: AutotuneDispatcher** — `Autotune/AutotuneDispatcher.cs`. On cache miss: run 3-5 candidate strategies back-to-back via `Stopwatch`, pick fastest median, persist winner via existing `AutotuneCache.Set`. On cache hit: return cached value directly.
- **H4: Cache persistence** — extend `Helpers/Autotune/BuiltInCatalog.cs` with BlasManaged entries. Cache file format unchanged.
- **H5: DisableAutotune mode** — `BlasOptions.PackingMode = DisableAutotune` bypasses cache lookup; uses default static heuristic.
- **H6: AutotuneTests.cs** — call same shape 10 times → first call benchmarks (slow), later calls cache-hit (fast). Verify cache file written.
- **H7: Phase H green-CI checkpoint** — L2 shape benchmark with autotune: first iter ~5 ms (benchmarking overhead), subsequent ~1 ms. **Gate 1 starts looking achievable.**

---

## Phase I: Epilogue chain

End state: bias + activation + skip + dropout + output-scale all fuse into the microkernel store. Adds 200-300 lines per arch but no per-call branches.

**Tasks:**

- **I1: EpilogueFlags bit-pack** — `Epilogue/EpilogueFlags.cs`. 8-bit flags: `HasBias | HasActivation(3 bits) | HasSkip | HasDropout | HasScale`.
- **I2: BiasEpilogue** — `Epilogue/BiasEpilogue.cs`. Adds `bias[n..n+Nr]` to each row of the C tile in-register.
- **I3: ActivationEpilogue** — `Epilogue/ActivationEpilogue.cs`. Switch on `ActivationType`. ReLU = `Vector.Max(acc, Zero)`; GELU = approximate per the codebase's existing GELU helper; etc.
- **I4: SkipEpilogue** — `Epilogue/SkipEpilogue.cs`. Adds `skip[m..m+Mr, n..n+Nr]` to C in-register.
- **I5: DropoutEpilogue** — `Epilogue/DropoutEpilogue.cs`. Multiplies C by a dropout mask derived from `Epilogue.DropoutMask` seed via a per-tile PRG (e.g., xoshiro256**). Bit-deterministic given the seed.
- **I6: OutputScaleEpilogue** — `Epilogue/OutputScaleEpilogue.cs`. Scalar multiply on C tile.
- **I7: Epilogue dispatch in microkernels** — each microkernel takes `EpilogueFlags` and applies stages in fixed order: `bias → activation → skip → dropout → scale → store`. Use `[MethodImpl(AggressiveInlining)]` so JIT inlines the branch on `flags == 0` (hot path).
- **I8: EpilogueTests.cs** — every combination of stages compared against an unfused two-pass reference. Includes ordering invariant (skip after activation, scale after dropout).
- **I9: Phase I green-CI checkpoint** — full suite passes.

---

## Phase J: JIT microkernel cache

End state: hot shapes get a shape-specialized IL-emitted microkernel that bakes M/N/K/strides as constants and inlines the epilogue chain. NativeAOT falls through cleanly.

**Tasks:**

- **J1: JittedKernelCache** — `Jit/JittedKernelCache.cs`. `ConcurrentDictionary<KernelKey, Delegate>`. LRU eviction at 64 MB IL.
- **J2: IL emission scaffolding** — `Jit/IlEmitter.cs`. `DynamicMethod` factory with delegate signature taking raw `IntPtr` for buffers (no bounds checks).
- **J3: AVX-512 IL emission** — `Jit/IlEmitter.Avx512.cs`. Emits `Avx512F.FusedMultiplyAdd(MethodInfo)` calls in unrolled K-loop body. M/N/K baked as `Ldc_I4` constants.
- **J4: AVX2 IL emission** — `Jit/IlEmitter.Avx2.cs`. Mirror of J3 with `Fma.MultiplyAdd`.
- **J5: Scalar IL emission** — `Jit/IlEmitter.Scalar.cs`. Same pattern; emits plain `Mul.Ovf` + `Add.Ovf` (or saturating equivalents).
- **J6: NativeAOT detector** — `Jit/NativeAotDetector.cs`. Returns `RuntimeFeature.IsDynamicCodeSupported`. Dispatcher checks this before emit.
- **J7: Background emit** — emit fires on `Task.Run` after 3+ calls to same shape. First 3 calls use hand-written kernel; later calls hit cache.
- **J8: LRU eviction** — when total emitted IL bytes exceed `BlasOptions.MaxJitCacheBytes` (default 64 MB), evict least-recently-used delegates.
- **J9: JitKernelTests.cs** — same shape 10 times → call 1 (hand-written) and call 4+ (JIT-emitted) produce bit-identical output. NativeAOT smoke test in `NativeAotSmokeTest.cs` confirms fallback path.
- **J10: Phase J green-CI checkpoint** — L2 shape benchmark now in JIT-emitted hot path: **target ≤ 1 ms**. If not yet there, profile via `DOTNET_JitDisasm` and tune. **This is when Gate 1 closes.**

---

## Phase K: Caller migration

End state: All ~30 existing `Avx512Sgemm.SgemmBlocked` / `SimdGemm.Sgemm` / `SimdGemm.Dgemm` / `BlasProvider.TryGemm` callers re-point to `BlasManaged.Gemm`. The old kernels' bodies become single-line shims marked `[Obsolete]`.

**Tasks (batched by file; each batch = one commit):**

- **K1: MatMul + BatchMatMul** — `CpuEngine.cs` MatMul paths (~3 sites). Pass `BlasOptions.Default`. Test: existing MatMul tests still pass.
- **K2: Conv2D forward** — `CpuEngine.cs` + `Im2ColHelper.cs` Conv2D forward path (~2 sites). Pass `BlasOptions.PackedA` when called from compiled-plan; else default.
- **K3: Conv2D backward (input)** — `CpuEngine.cs` line 12025 `Conv2DBackwardInput<T>` (~2 sites). The kernel transpose at line 12136 becomes the pre-pack handle path.
- **K4: Conv2D backward (kernel)** — analogous Conv2DBackwardKernel sites.
- **K5: ConvTranspose2D forward + backward** — **delete `TryConvTranspose2DWithGemm` heuristic at lines 1129 and 1255** of `Im2ColHelper.cs`. Replace with `BlasManaged.Gemm` call passing the same shape; autotune learns the right strategy. **This is the L2 fix.**
- **K6: Attention** — `FlashAttention.cs` Q·Kᵀ + S·V (2 sites). Pass `Epilogue.Activation = None`; S·V can use `Epilogue.DropoutMask` when training.
- **K7: Attention backward** — `BackwardFunctions.cs` (3 sites).
- **K8: Fused multi-layer** — `FusedMultiLayerGemm.cs` + `FusedMultiLayerBackward.cs`. Per-layer `PrePackA` handles passed into each call. **Biggest perf win lands here.**
- **K9: BLAS fallback wrappers** — `BlasProvider.TryGemm` + `TryGemmEx`. When native BLAS unavailable, route to `BlasManaged.Gemm` instead of `SimdGemm.Sgemm`.
- **K10: Compiled-plan inliners** — `CompiledTrainingPlan.cs` + `BackwardCSEPass.cs` (~5 sites). Use pack-handle table.
- **K11: Decomposition utilities + remaining sites** — `SvdDecomposition.cs`, `MatrixMultiplyHelper.cs` (~2 sites).
- **K12: Shim the old APIs** — `Avx512Sgemm.SgemmBlocked` / `SimdGemm.Sgemm` / `SimdGemm.Dgemm` bodies become single-line forwards. Add `[Obsolete("Use BlasManaged.Gemm. Will be removed in the release after vX.Y.")]`.
- **K13: Phase K green-CI checkpoint** — every test in `AiDotNet.Tensors.Tests` passes on both frameworks. No call to old kernel bodies remains internal to the codebase.

---

## Phase L: Acceptance gates

End state: all 4 gates from Section 10 of the spec pass in CI.

**Tasks:**

- **L1: ConvTranspose2D L2 perf gate (Gate 1)** — `tests/AiDotNet.Tensors.Tests/BlasManaged/ConvTranspose2DL2PerfTest.cs`. Runs `BlasManaged.Gemm<double>(M=4096, N=16, K=512, transA=true)` 100 iters post-warmup, asserts median ≤ 1 ms on AVX-512, ≤ 5 ms on AVX2-only, ≤ 5 ms on Neon. Skips on other arches with `[SkippableFact]`.
- **L2: Existing benchmark baseline (Gate 2)** — `tests/AiDotNet.Tensors.Benchmarks/baselines/preBlasManaged.json`. Captured BEFORE Phase K caller migration (i.e., at end of Phase J). Holds median ns/op for each pre-existing benchmark.
- **L3: No-regression CI step** — `tests/AiDotNet.Tensors.Tests/BlasManaged/RegressionBaselineTests.cs`. Compares current benchmark medians to baseline; fails if any > 5% slower.
- **L4: DeterminismTests** — already drafted in G7; finalize the 12-shape representative set.
- **L5: NativeAOT smoke test** — separate AOT-published sub-project under `tests/AiDotNet.Tensors.NativeAotSmoke/`. Calls `BlasManaged.Gemm` on a half-dozen shapes; binary publish via `dotnet publish -c Release -r win-x64 -p:PublishAot=true`. CI step runs the AOT binary and asserts exit code 0 + expected output.
- **L6: Pack-cache invalidation test** — `tests/AiDotNet.Tensors.Tests/BlasManaged/WeightPackInvalidationTests.cs`. End-to-end Allocate → Gemm → mutate weight → MarkDirty → Gemm → assert. (Already drafted in F7.)
- **L7: DCGAN companion PR (Gate 4)** — separate PR in sibling `..\AiDotNet` repo enabling `DCGANTests.MoreData_ShouldNotDegrade` with 60 s budget. This PR's CI step runs against the BlasManaged branch's locally-built `AiDotNet.Tensors`.
- **L8: Phase L green-CI checkpoint** — all 4 gates green in CI.

---

## Phase M: PR finalization

- **M1: PR description with per-commit roadmap** — break out the 13 phases with commit hashes for reviewer navigation.
- **M2: Spec + plan cross-reference** — add `Spec: docs/superpowers/specs/2026-05-16-blas-managed-design.md` and `Plan: docs/superpowers/plans/2026-05-16-blas-managed.md` to PR body.
- **M3: Companion PR coordination** — link to the sibling `AiDotNet` DCGAN-test-enable PR.
- **M4: Mark old kernels `[Obsolete]` with one-release deprecation window** — already done in K12, confirm visible in PR diff.
- **M5: Final CI green check** — all 4 acceptance gates pass; both target frameworks build; all pre-existing tests pass.
- **M6: Request review.**

---

## Spec Coverage Self-Review

Every section of the spec maps to plan tasks:

| Spec section | Plan coverage |
|---|---|
| §1 Background | Spec linked in plan header; no code |
| §2 Goals + non-goals | Embedded in acceptance gates (§10 → Phase L) |
| §3 Architecture overview | Phase A (scaffolding) + Phase B (wire-up) |
| §4 Microkernel design | Phases B (scalar), C (AVX2), D (AVX-512), E (Neon) |
| §5 Packing + cache hierarchy | Phases B (scalar Pack-A/B/Both/Streaming), C/D/E (SIMD pack) |
| §6 Parallelism + determinism | Phase G |
| §7 Allocator + JIT cache | Phases F (allocator), J (JIT) |
| §8 Autotune | Phase H |
| §9 Caller migration | Phase K |
| §10 Acceptance criteria | Phase L (all 4 gates) |
| §11 Risks + mitigations | Embedded in phase notes (e.g., RyuJIT register spilling in D, NativeAOT in J) |
| §12 Approaches considered | Spec-only; no plan tasks |
| §13 Open questions | None |
| §14 Decisions summary | All 13 reflected in phase scope |

Type / signature consistency check: `BlasManaged.Gemm<T>` signature is identical in spec §3, Task A1, Task A3, and Task B8. `WeightPackHandle.MarkDirty()` consistent in spec §7 and Task A3. `EpilogueFlags` bit layout matches spec §4 (epilogue plumbing) and Phase I.

No placeholders. No "TBD" or "fill in later" markers. Tasks C–M are concrete (file paths, shape parameters, test asserts) but compressed; will be expanded into full TDD step-by-step at the start of each phase.

---

## Pre-PR Phase 0 (optional safety net)

If the user wants to unblock `DCGANTests.MoreData_ShouldNotDegrade` BEFORE the 8-12 week BLIS work lands, a tiny "Phase 0" lands first as its own PR:

**P0.1: L2-shape detection in `TryConvTranspose2DWithGemm`**
- Modify `src/AiDotNet.Tensors/Helpers/Im2ColHelper.cs` lines 1126-1129 and 1252-1255.
- When `kernelSize > 524288 && hw <= 16`, return `false` immediately to skip BLAS.
- The caller's existing naive 7-nested-loop fallback (already parallel; measures ~100 ms vs BLAS's 215 ms) carries the L2 shape.
- Commit: `perf(#358): skip BLAS for L2-class transA shape (M=4096 N=16 K=512+) — falls through to parallel naive loop`.

**P0.2: Verify DCGAN test passes** — companion AiDotNet PR enables the test with 120 s budget (relaxed from the 60 s the BLIS gate eventually targets).

This Phase 0 is *not* part of the BLIS PR. It's a separate single-commit fix that buys time without blocking the larger refactor. If the user wants it, it can be cut from a branch off `main` directly, independent of the BLIS branch.
