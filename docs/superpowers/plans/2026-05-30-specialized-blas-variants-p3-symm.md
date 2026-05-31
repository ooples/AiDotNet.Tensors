# Specialized BLAS Variants — P3 (SYMM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Checkbox (`- [ ]`) steps.

**Goal:** Add a bit-deterministic, drop-in `BlasManaged.Symm<T>` symmetric matrix multiply (`C = α·A·B + β·C` for Side.Left, `C = α·B·A + β·C` for Side.Right; A symmetric, stored in the `uplo` triangle), reusing the existing GEMM core.

**Architecture:** Approach A. Materialize the full symmetric A from its `uplo` triangle into a scratch buffer (mirror across the diagonal), run the existing `Gemm<T>` into a result scratch, then write `C = α·result + β·C`. Cost of the symmetric materialization is O(side²), negligible vs the GEMM's O(side²·n). The "mirror-on-pack" micro-opt (avoid the temp-A buffer) only matters for tiny n and is tracked as a follow-up, not part of this deliverable.

**Spec:** `docs/superpowers/specs/2026-05-30-specialized-blas-variants-design.md` §4 SYMM. **Branch:** `feature/379-specialized-blas-variants`.

---

## File Structure
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Symm.cs`
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/SymmTests.cs`, `SymmDeterminismTests.cs`
- Modify: `SpecializedShapeCatalog.cs`, `SpecializedPerfBar.cs`

---

## Task 1: Failing test (FP64, Left, Lower)

Create `SymmTests.cs` with an in-test oracle (`A_sym[i,j] = stored-triangle-or-mirror`) and one `[Fact]` for Side.Left/Uplo.Lower comparing to `BlasManaged.Symm<double>`. Build test project → expect `CS0117 'Symm'`.

## Task 2: Implement Symm (materialize symmetric A + Gemm)

Create `BlasManaged.Symm.cs`:

```csharp
using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Symmetric matrix multiply. Side.Left: C = α·A·B + β·C (A is m×m symmetric).
    /// Side.Right: C = α·B·A + β·C (A is n×n symmetric). A is stored in the
    /// <paramref name="uplo"/> triangle. Drop-in for cblas_ssymm/cblas_dsymm.
    /// </summary>
    public static void Symm<T>(
        Side side, Uplo uplo,
        int m, int n, T alpha,
        ReadOnlySpan<T> a, int lda,
        ReadOnlySpan<T> b, int ldb, T beta,
        Span<T> c, int ldc,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (m <= 0 || n <= 0) return;
        var ops = MathHelper.GetNumericOperations<T>();
        int s = side == Side.Left ? m : n;   // dimension of square symmetric A

        // Materialize full symmetric A (s×s) from the uplo triangle (mirror).
        T[] full = new T[s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
            {
                bool stored = uplo == Uplo.Lower ? j <= i : j >= i;
                full[i * s + j] = stored ? a[i * lda + j] : a[j * lda + i];
            }

        // result = A·B (Left, m×n) or B·A (Right, m×n) via the existing GEMM core.
        T[] result = new T[m * n];
        var gemmOpts = new BlasOptions<T> { NumThreads = options.NumThreads, Mode = options.Mode };
        if (side == Side.Left)
            Gemm<T>(full, s, false, b, ldb, false, result, n, m, n, s, gemmOpts); // (m×m)(m×n)
        else
            Gemm<T>(b, ldb, false, full, s, false, result, n, m, n, s, gemmOpts); // (m×n)(n×n)

        // C = α·result + β·C
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                int ci = i * ldc + j;
                c[ci] = ops.Add(ops.Multiply(alpha, result[i * n + j]), ops.Multiply(beta, c[ci]));
            }
    }
}
```

Build src → succeed. Run first test → pass. Commit.

## Task 3: Coverage (Side×Uplo, FP32, α/β) + determinism

Append parameterized `Side×Uplo` FP64 α/β test + FP32 test to `SymmTests.cs`; create `SymmDeterminismTests.cs` (bit-exact across 1/2/4/8 threads, n large enough to parallelize). Run → all pass. Commit.

## Task 4: Catalog + perf-bar + multi-TFM

Add `SymmShape[]` to `SpecializedShapeCatalog.cs` and `Symm*` bar constants to `SpecializedPerfBar.cs`. Build net10.0+net471, run full Symm suite. Commit.

## Tracked follow-up (not silently dropped)
Mirror-on-pack (avoid temp-A materialization) — micro-opt for tiny n; tracked, deferred.

## Self-Review
§4 SYMM reuse-core + mirror → Task 2 ✓; determinism (GEMM core deterministic, materialization fixed-order) → Task 3 ✓; catalog/bar → Task 4 ✓. Placeholder: only documented bar stubs.
