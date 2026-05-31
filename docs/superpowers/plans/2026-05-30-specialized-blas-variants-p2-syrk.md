# Specialized BLAS Variants — P2 (SYRK) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a bit-deterministic, drop-in `BlasManaged.Syrk<T>` symmetric rank-k update (`C = α·op(A)·op(A)ᵀ + β·C`, writing only the requested triangle), reusing the existing GEMM core, with a tile-skip optimization that computes only the needed triangle.

**Architecture:** Approach A. First correct impl reuses the public `Gemm<T>` (B = A, transB flipped) into a scratch buffer, then writes `C[uplo] = α·scratch + β·C[uplo]`. The optimization then computes only the triangle's tiles (skip the off-`uplo` half) to cut ~half the FLOPs — the real SYRK win over a dense GEMM. TDD: correctness + determinism tests gate both the baseline and the optimization.

**Tech Stack:** C# (net10.0 / net471), xUnit, existing `BlasManaged` infrastructure.

**Spec:** `docs/superpowers/specs/2026-05-30-specialized-blas-variants-design.md` §4 SYRK.

**Branch:** `feature/379-specialized-blas-variants` (continues after P0+P1).

---

## File Structure

**Created:**
- `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Syrk.cs` — `BlasManaged.Syrk<T>`.
- `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/SyrkTests.cs` — correctness (Uplo×Trans, FP32/FP64, α/β).
- `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/SyrkDeterminismTests.cs` — bit-exact across thread counts.

**Modified:**
- `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/Catalog/SpecializedShapeCatalog.cs` — add SYRK shapes.
- `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/SpecializedPerfBar.cs` — add SYRK bar stub.

---

## Task 1: Failing SYRK correctness test (FP64, Lower, NoTrans)

**Files:** Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/SyrkTests.cs`

- [ ] **Step 1: Write the failing test** (oracle = naive triple loop, independent of production)

```csharp
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class SyrkTests
{
    // Reference: C = alpha*op(A)*op(A)^T + beta*C, full dense, then caller checks triangle.
    // trans=false: A is n×k, op(A)=A. trans=true: A is k×n, op(A)=A^T (n×k effective).
    private static double[] ReferenceSyrk(
        bool trans, int n, int k, double alpha, double[] a, int lda, double beta, double[] c, int ldc)
    {
        var outC = (double[])c.Clone();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double dot = 0;
                for (int p = 0; p < k; p++)
                {
                    // op(A)[i,p] and op(A)[j,p]
                    double aip = trans ? a[p * lda + i] : a[i * lda + p];
                    double ajp = trans ? a[p * lda + j] : a[j * lda + p];
                    dot += aip * ajp;
                }
                outC[i * ldc + j] = alpha * dot + beta * c[i * ldc + j];
            }
        return outC;
    }

    [Fact]
    public void Syrk_FP64_LowerNoTrans_MatchesReferenceTriangle()
    {
        const int n = 5, k = 3;
        var rng = new Random(42);
        double[] a = new double[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[n * n];
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 2 - 1;

        double[] expectedFull = ReferenceSyrk(false, n, k, 1.0, a, k, 0.0, c, n);

        double[] actual = (double[])c.Clone();
        BlasManagedLib.Syrk<double>(Uplo.Lower, trans: false, n, k, 1.0, a, k, 0.0, actual, n);

        // Only the lower triangle (incl diagonal) must match; upper is untouched (= original c).
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (j <= i) Assert.Equal(expectedFull[i * n + j], actual[i * n + j], 10);
                else        Assert.Equal(c[i * n + j], actual[i * n + j], 10);
    }
}
```

- [ ] **Step 2: Build the test project — expect compile failure** `'BlasManaged' does not contain a definition for 'Syrk'`.

Run: `dotnet build tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --no-restore -f net10.0`
Expected: `error CS0117 ... 'Syrk'`.

---

## Task 2: Implement Syrk (reuse Gemm into scratch + triangular write)

**Files:** Create `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Syrk.cs`

- [ ] **Step 1: Write the implementation**

```csharp
using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Symmetric rank-k update: C = α·op(A)·op(A)ᵀ + β·C, writing only the
    /// <paramref name="uplo"/> triangle of the n×n matrix C. op(A) is A (trans=false,
    /// A is n×k) or Aᵀ (trans=true, A is k×n). Drop-in for cblas_ssyrk/cblas_dsyrk.
    /// </summary>
    public static void Syrk<T>(
        Uplo uplo, bool trans,
        int n, int k, T alpha,
        ReadOnlySpan<T> a, int lda, T beta,
        Span<T> c, int ldc,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (n <= 0) return;
        var ops = MathHelper.GetNumericOperations<T>();

        // full = op(A)·op(A)ᵀ  (n×n) via the existing GEMM core.
        // trans=false: A(n×k) · A(n×k)ᵀ → Gemm(a=A, transA=false, b=A, transB=true, k=k)
        // trans=true:  Aᵀ(n×k) · A(k×n) → Gemm(a=A, transA=true, b=A, transB=false, k=k)
        T[] full = new T[n * n];
        if (k > 0)
        {
            var gemmOpts = new BlasOptions<T> { NumThreads = options.NumThreads, Mode = options.Mode };
            Gemm<T>(a, lda, trans, a, lda, !trans, full, n, n, n, k, gemmOpts);
        }

        // Write C[uplo] = α·full + β·C[uplo]; leave the other triangle untouched.
        for (int i = 0; i < n; i++)
        {
            int lo = uplo == Uplo.Lower ? 0 : i;
            int hi = uplo == Uplo.Lower ? i : n - 1;
            for (int j = lo; j <= hi; j++)
            {
                int ci = i * ldc + j;
                T scaled = ops.Multiply(alpha, full[i * n + j]);
                c[ci] = ops.Add(scaled, ops.Multiply(beta, c[ci]));
            }
        }
    }
}
```

- [ ] **Step 2: Build src** — `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --no-restore -f net10.0` → `Build succeeded.`
- [ ] **Step 3: Run the test** — `dotnet test ... --filter "FullyQualifiedName~Syrk_FP64_LowerNoTrans_MatchesReferenceTriangle" -f net10.0` → `Passed!`.
- [ ] **Step 4: Commit** — `git commit -m "feat(#379): BlasManaged.Syrk<T> reusing GEMM core + first test"`.

---

## Task 3: Full coverage (Uplo×Trans, FP32, α/β) + determinism

**Files:** Modify `SyrkTests.cs`; create `SyrkDeterminismTests.cs`.

- [ ] **Step 1: Append the parameterized coverage test to `SyrkTests.cs`**

```csharp
    public static System.Collections.Generic.IEnumerable<object[]> Matrix()
    {
        foreach (var uplo in new[] { Uplo.Upper, Uplo.Lower })
        foreach (var trans in new[] { false, true })
            yield return new object[] { uplo, trans };
    }

    [Theory]
    [MemberData(nameof(Matrix))]
    public void Syrk_FP64_Coverage_AlphaBeta_MatchesReference(Uplo uplo, bool trans)
    {
        const int n = 6, k = 4;
        var rng = new Random(321);
        // trans=false → A is n×k (lda=k); trans=true → A is k×n (lda=n).
        int aRows = trans ? k : n, aCols = trans ? n : k, lda = aCols;
        double[] a = new double[aRows * aCols];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[n * n];
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 2 - 1;

        double alpha = 1.7, beta = -0.5;
        double[] expectedFull = ReferenceSyrk(trans, n, k, alpha, a, lda, beta, c, n);
        double[] actual = (double[])c.Clone();
        BlasManagedLib.Syrk<double>(uplo, trans, n, k, alpha, a, lda, beta, actual, n);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                bool inTri = uplo == Uplo.Lower ? j <= i : j >= i;
                if (inTri) Assert.Equal(expectedFull[i * n + j], actual[i * n + j], 9);
                else       Assert.Equal(c[i * n + j], actual[i * n + j], 9);
            }
    }

    [Fact]
    public void Syrk_FP32_LowerNoTrans_MatchesReference()
    {
        const int n = 5, k = 4;
        var rng = new Random(11);
        float[] a = new float[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] c = new float[n * n];
        for (int i = 0; i < c.Length; i++) c[i] = (float)(rng.NextDouble() * 2 - 1);

        double[] a64 = Array.ConvertAll(a, x => (double)x);
        double[] c64 = Array.ConvertAll(c, x => (double)x);
        double[] expectedFull = ReferenceSyrk(false, n, k, 1.0, a64, k, 0.0, c64, n);

        float[] actual = (float[])c.Clone();
        BlasManagedLib.Syrk<float>(Uplo.Lower, false, n, k, 1f, a, k, 0f, actual, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                Assert.Equal(expectedFull[i * n + j], actual[i * n + j], 3);
    }
```

- [ ] **Step 2: Create `SyrkDeterminismTests.cs`**

```csharp
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class SyrkDeterminismTests
{
    [Theory]
    [InlineData(64, 48)]
    [InlineData(128, 96)]
    [InlineData(192, 64)]
    public void Syrk_FP64_BitExactAcrossThreadCounts(int n, int k)
    {
        var rng = new Random(42);
        double[] a = new double[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] c0 = new double[n * n];
        for (int i = 0; i < c0.Length; i++) c0[i] = rng.NextDouble() * 2 - 1;

        double[]? baseline = null;
        foreach (int threads in new[] { 1, 2, 4, 8 })
        {
            double[] actual = (double[])c0.Clone();
            var opts = new BlasOptions<double> { NumThreads = threads, Mode = BlasMode.Deterministic };
            BlasManagedLib.Syrk<double>(Uplo.Lower, false, n, k, 1.3, a, k, 0.7, actual, n, opts);
            if (baseline is null) baseline = actual;
            else for (int i = 0; i < actual.Length; i++) Assert.Equal(baseline[i], actual[i]);
        }
    }
}
```

- [ ] **Step 3: Run both** — `dotnet test ... --filter "FullyQualifiedName~Syrk" -f net10.0` → all pass (4 coverage + 1 fp32 + 1 original + 3 determinism = 9).
- [ ] **Step 4: Commit** — `git commit -m "test(#379): Syrk full coverage + determinism"`.

---

## Task 4: Tile-skip optimization (compute only the requested triangle)

**Files:** Modify `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Syrk.cs`

Replace the full n×n GEMM with blocked tiles, computing only tiles intersecting the requested triangle (~half the FLOPs). Off-triangle tiles are skipped entirely; the diagonal tile is computed and masked.

- [ ] **Step 1: Add the blocked path and route to it for large n**

```csharp
    private const int SyrkBlock = 64;

    private static void SyrkBlocked<T>(
        Uplo uplo, bool trans, int n, int k, T alpha, ReadOnlySpan<T> a, int lda, T beta,
        Span<T> c, int ldc, in BlasOptions<T> options, INumericOperations<T> ops) where T : unmanaged
    {
        var gemmOpts = new BlasOptions<T> { NumThreads = options.NumThreads, Mode = options.Mode };
        for (int i0 = 0; i0 < n; i0 += SyrkBlock)
        {
            int bm = Math.Min(SyrkBlock, n - i0);
            for (int j0 = 0; j0 < n; j0 += SyrkBlock)
            {
                int bn = Math.Min(SyrkBlock, n - j0);
                // Skip tiles wholly outside the requested triangle.
                if (uplo == Uplo.Lower) { if (j0 > i0 + bm - 1) continue; }
                else                    { if (j0 + bn - 1 < i0) continue; }

                // tile = op(A)[i0:i0+bm, :] · op(A)[j0:j0+bn, :]ᵀ  (bm×bn)
                // Row r of op(A) lives at: trans ? column r of A : row r of A.
                T[] tile = new T[bm * bn];
                if (k > 0)
                {
                    // Build the two operands as offset spans into A.
                    // trans=false: op(A) row r = a[r*lda + p]; sub-block rows i0.., j0..
                    //   Gemm(aSub(bm×k), transA=false, bSub(bn×k), transB=true, m=bm,n=bn,k=k)
                    // trans=true:  op(A) row r = a[p*lda + r]; columns i0.., j0..
                    //   Use transA=true on the i-block, transB=false won't line up by stride,
                    //   so fall back to the dense full path for trans=true (correctness first).
                    if (!trans)
                    {
                        var aI = a.Slice(i0 * lda);          // bm rows × k, lda-strided
                        var aJ = a.Slice(j0 * lda);          // bn rows × k, lda-strided
                        Gemm<T>(aI, lda, false, aJ, lda, true, tile, bn, bm, bn, k, gemmOpts);
                    }
                    else
                    {
                        // Column-strided operands: compute this tile with a scalar dot to
                        // stay correct (trans tile-skip uses the same skip pattern, scalar inner).
                        for (int ii = 0; ii < bm; ii++)
                            for (int jj = 0; jj < bn; jj++)
                            {
                                T dot = ops.Zero;
                                for (int p = 0; p < k; p++)
                                    dot = ops.Add(dot, ops.Multiply(a[p * lda + (i0 + ii)], a[p * lda + (j0 + jj)]));
                                tile[ii * bn + jj] = dot;
                            }
                    }
                }

                // Write tile into C[uplo] with alpha/beta, masking the diagonal tile.
                for (int ii = 0; ii < bm; ii++)
                {
                    int gi = i0 + ii;
                    for (int jj = 0; jj < bn; jj++)
                    {
                        int gj = j0 + jj;
                        bool inTri = uplo == Uplo.Lower ? gj <= gi : gj >= gi;
                        if (!inTri) continue;
                        int ci = gi * ldc + gj;
                        c[ci] = ops.Add(ops.Multiply(alpha, tile[ii * bn + jj]), ops.Multiply(beta, c[ci]));
                    }
                }
            }
        }
    }
```

Route at the top of `Syrk`, after computing `ops`:

```csharp
        if (n > SyrkBlock)
        {
            SyrkBlocked(uplo, trans, n, k, alpha, a, lda, beta, c, ldc, options, ops);
            return;
        }
```

- [ ] **Step 2: Build + run the full Syrk suite (correctness + determinism must still pass)**

Run: `dotnet test ... --filter "FullyQualifiedName~Syrk" -f net10.0`
Expected: all pass. Add a large-n case below to force the blocked path.

- [ ] **Step 3: Add a large-n correctness case to `SyrkTests.cs`**

```csharp
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Syrk_FP64_LargeN_BlockedPath_MatchesReference(bool trans)
    {
        const int n = 150, k = 40; // n > SyrkBlock(64)
        var rng = new Random(2025);
        int aRows = trans ? k : n, aCols = trans ? n : k, lda = aCols;
        double[] a = new double[aRows * aCols];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[n * n];
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 2 - 1;

        double alpha = 0.9, beta = 0.3;
        double[] expectedFull = ReferenceSyrk(trans, n, k, alpha, a, lda, beta, c, n);
        double[] actual = (double[])c.Clone();
        BlasManagedLib.Syrk<double>(Uplo.Lower, trans, n, k, alpha, a, lda, beta, actual, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                Assert.Equal(expectedFull[i * n + j], actual[i * n + j], 8);
    }
```

- [ ] **Step 4: Run + commit** — `git commit -m "perf(#379): Syrk tile-skip computes only the requested triangle"`.

---

## Task 5: Catalog + perf-bar entries; multi-TFM build

**Files:** Modify `SpecializedShapeCatalog.cs`, `SpecializedPerfBar.cs`.

- [ ] **Step 1: Add SYRK shapes to `SpecializedShapeCatalog.cs`** (inside the class)

```csharp
    public record SyrkShape(string Name, Uplo Uplo, bool Trans, int N, int K, bool Fp64, int Frequency, string Source);

    public static readonly SyrkShape[] Syrk =
    {
        new("Cov_256x64",  Uplo.Lower, true,  256,  64, true, 30, "workload:covariance"),
        new("Cov_512x128", Uplo.Lower, true,  512, 128, true, 25, "workload:covariance"),
        new("Gram_128x768",Uplo.Lower, false, 128, 768, true, 20, "workload:gram-matrix"),
        new("Cov_FP32_256x64", Uplo.Lower, true, 256, 64, false, 15, "workload:covariance-fp32"),
    };
```

- [ ] **Step 2: Add SYRK bar stub to `SpecializedPerfBar.cs`** (inside the class)

```csharp
    // SYRK vs OpenBLAS ssyrk/dsyrk on the authoritative runner.
    public const int    SyrkMinWinRatePercent = 0;     // TO BE SET after first bench
    public const double SyrkMaxLossMultiple    = 99.0; // TO BE SET after first bench
    public static bool SyrkBarFrozen => SyrkMinWinRatePercent > 0;
```

- [ ] **Step 3: Multi-TFM build + full Syrk suite**

Run: `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --no-restore` → both net10.0 + net471 succeed.
Run: `dotnet test ... --filter "FullyQualifiedName~Syrk" -f net10.0` → all pass.

- [ ] **Step 4: Commit** — `git commit -m "test(#379): SYRK bench catalog + perf-bar stub"`.

---

## Task 6 (tracked, not silently dropped): SYRK call-site rewiring

Covariance/gram call sites (`CpuEngine`, `MultivariateDistributions`) are candidate rewires to `BlasManaged.Syrk`, but each computes `Aᵀ·A`/`A·Aᵀ` inside larger routines and needs individual analysis to preserve exact semantics (centering, normalization, batch handling). This task is **explicitly tracked** for a focused follow-up commit on this branch — it is not part of the kernel deliverable and must not be force-rewired without per-site verification (existing tests are the canary, same discipline as P1 Task 9).

---

## Self-Review

- §4 SYRK reuse-GEMM-core → Task 2 ✓
- §4 tile-skip (only requested triangle) → Task 4 ✓
- §4 triangular-write + α/β → Tasks 2,3 ✓
- §3 determinism contract → Task 3 (bit-exact across threads; GEMM core is deterministic, scalar trans tile is fixed-order) ✓
- §5 catalog + frozen bar → Task 5 ✓
- Placeholder scan: only the documented `0`/`99.0` perf-bar stubs. Type consistency: `Syrk<T>`, `SyrkBlock`, `SyrkBlocked`, `ReferenceSyrk` consistent across tasks.
- Note: trans=true tile-skip uses a scalar inner dot (correct + deterministic); a SIMD trans path is a later micro-opt, flagged here, not silently dropped.
