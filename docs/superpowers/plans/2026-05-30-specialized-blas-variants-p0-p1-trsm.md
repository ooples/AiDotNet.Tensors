# Specialized BLAS Variants — P0 (Scaffolding) + P1 (TRSM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the shared scaffolding (`Side`/`Uplo`/`Diag` enums, `SparseLayout<T>`, perf/determinism harness stubs) and the first variant — a bit-deterministic, drop-in `BlasManaged.Trsm<T>` triangular solve — then rewire the existing managed substitution loops in `LinearSolvers` to use it.

**Architecture:** Approach A from the spec — `Trsm` is a blocked BLIS-style driver: solve each `Mr`-sized diagonal block with a new scalar triangular-solve kernel, then update the trailing right-hand-side via the existing `PackBothStrategy` GEMM macrokernel. TDD throughout: a correct scalar `Trsm` is built and verified against the existing `TriangularSolveSingle` oracle first, then the blocked/macrokernel-reuse optimization is layered in behind the same passing tests.

**Tech Stack:** C# (net10.0 / net471 multi-target), xUnit, the existing `AiDotNet.Tensors.Engines.BlasManaged` infrastructure (microkernels, `PackBothStrategy`, `BlasOptions<T>`, `BlasMode`).

**Spec:** `docs/superpowers/specs/2026-05-30-specialized-blas-variants-design.md` (§3 API, §4 TRSM, §6 phasing).

**Branch:** `feature/379-specialized-blas-variants` (already created; the spec commit is already on it).

---

## File Structure

**Created:**
- `src/AiDotNet.Tensors/Engines/BlasManaged/SpecializedBlasEnums.cs` — `Side`, `Uplo`, `Diag` enums + `SparseLayout<T>` ref-struct view + `SparseLayoutFormat` enum.
- `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Trsm.cs` — `BlasManaged.Trsm<T>` partial-class file (keeps the 800-line `BlasManaged.cs` from growing further; each new variant gets its own partial file).
- `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/Catalog/SpecializedShapeCatalog.cs` — per-variant bench shapes (TRSM entries this phase).
- `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/SpecializedPerfBar.cs` — frozen per-variant perf-bar constants (TRSM stub this phase).
- `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmTests.cs` — correctness + shape-coverage matrix.
- `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmDeterminismTests.cs` — bit-exact across thread counts.

**Modified:**
- `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs` — change `public static class BlasManaged` to `public static partial class BlasManaged` so the `.Trsm.cs` partial compiles.
- `src/AiDotNet.Tensors/LinearAlgebra/Solvers/LinearSolvers.cs:129-163` — rewire `SolveTriangularInternal` to call `BlasManaged.Trsm` for the FP32/FP64 non-batched 2D case; keep `TriangularSolveSingle` as the fallback (batched/other T) and as the test oracle.

---

## Task 1: Make `BlasManaged` a partial class

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs:12`

- [ ] **Step 1: Change the class declaration to partial**

In `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs`, line 12, change:

```csharp
public static class BlasManaged
```

to:

```csharp
public static partial class BlasManaged
```

- [ ] **Step 2: Build to verify nothing broke**

Run: `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --no-restore -f net10.0`
Expected: `Build succeeded.` with 0 errors. (`partial` on a single declaration is always legal.)

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs
git commit -m "refactor(#379): make BlasManaged a partial class for per-variant files"
```

---

## Task 2: Add the shared enums and `SparseLayout<T>`

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/SpecializedBlasEnums.cs`

- [ ] **Step 1: Write the file**

Create `src/AiDotNet.Tensors/Engines/BlasManaged/SpecializedBlasEnums.cs`:

```csharp
using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>Which side of the product the triangular/symmetric operand sits on.</summary>
public enum Side
{
    /// <summary>op(A) is on the left: op(A)·X = B  (TRSM) or C = A·B (SYMM).</summary>
    Left,
    /// <summary>op(A) is on the right: X·op(A) = B (TRSM) or C = B·A (SYMM).</summary>
    Right,
}

/// <summary>Which triangle of a triangular/symmetric matrix is referenced.</summary>
public enum Uplo
{
    /// <summary>Upper triangle (including diagonal) holds the data.</summary>
    Upper,
    /// <summary>Lower triangle (including diagonal) holds the data.</summary>
    Lower,
}

/// <summary>Whether a triangular matrix has an implicit unit diagonal.</summary>
public enum Diag
{
    /// <summary>Diagonal entries are read from the matrix.</summary>
    NonUnit,
    /// <summary>Diagonal entries are assumed to be 1 and not read.</summary>
    Unit,
}

/// <summary>Storage format of a <see cref="SparseLayout{T}"/> view.</summary>
public enum SparseLayoutFormat
{
    /// <summary>Compressed Sparse Row: RowPtr length = Rows+1, Indices = column indices.</summary>
    Csr,
    /// <summary>Compressed Sparse Column: RowPtr length = Cols+1, Indices = row indices.</summary>
    Csc,
}

/// <summary>
/// Allocation-free readonly view over a CSR/CSC sparse matrix, decoupled from the
/// heap <see cref="AiDotNet.Tensors.LinearAlgebra.SparseTensor{T}"/> class so
/// <see cref="BlasManaged.SpMM{T}"/> can be called from hot paths without boxing.
/// For CSR, <see cref="Pointers"/> is the row-pointer array (length Rows+1) and
/// <see cref="Indices"/> holds column indices. For CSC the roles transpose.
/// </summary>
public readonly ref struct SparseLayout<T> where T : unmanaged
{
    /// <summary>Number of rows in the logical (dense-equivalent) matrix.</summary>
    public int Rows { get; init; }
    /// <summary>Number of columns in the logical (dense-equivalent) matrix.</summary>
    public int Cols { get; init; }
    /// <summary>Row pointers (CSR) or column pointers (CSC).</summary>
    public ReadOnlySpan<int> Pointers { get; init; }
    /// <summary>Column indices (CSR) or row indices (CSC).</summary>
    public ReadOnlySpan<int> Indices { get; init; }
    /// <summary>Nonzero values, parallel to <see cref="Indices"/>.</summary>
    public ReadOnlySpan<T> Values { get; init; }
    /// <summary>Which compressed format the spans encode.</summary>
    public SparseLayoutFormat Format { get; init; }
}
```

- [ ] **Step 2: Build to verify it compiles**

Run: `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --no-restore -f net10.0`
Expected: `Build succeeded.` 0 errors.

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/SpecializedBlasEnums.cs
git commit -m "feat(#379): add Side/Uplo/Diag enums and SparseLayout<T> view"
```

---

## Task 3: Write the failing TRSM correctness test (FP64, the simplest case)

**Files:**
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmTests.cs`

The oracle is a naive in-test reference solve (independent of the production code so the test can't be satisfied by a no-op). We start with the single simplest case: left, lower, non-transposed, non-unit, single RHS.

- [ ] **Step 1: Write the failing test**

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmTests.cs`:

```csharp
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class TrsmTests
{
    // Naive reference: solve op(A)·X = alpha·B in place, row-major, left side.
    // Independent of production code so it is a genuine oracle.
    private static void ReferenceTrsmLeft(
        Side side, Uplo uplo, bool transA, Diag diag,
        int m, int n, double alpha,
        double[] a, int lda, double[] b, int ldb)
    {
        // Scale B by alpha first.
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                b[i * ldb + j] *= alpha;

        // Effective triangular access: A(r,c) with optional transpose.
        double A(int r, int c) => transA ? a[c * lda + r] : a[r * lda + c];

        bool lower = (uplo == Uplo.Lower) ^ transA; // transpose flips triangle
        if (side != Side.Left) throw new NotSupportedException("test covers Left only here");

        if (lower)
        {
            // Forward substitution: row 0..m-1
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = b[i * ldb + j];
                    for (int kk = 0; kk < i; kk++) sum -= A(i, kk) * b[kk * ldb + j];
                    b[i * ldb + j] = diag == Diag.Unit ? sum : sum / A(i, i);
                }
        }
        else
        {
            // Back substitution: row m-1..0
            for (int i = m - 1; i >= 0; i--)
                for (int j = 0; j < n; j++)
                {
                    double sum = b[i * ldb + j];
                    for (int kk = i + 1; kk < m; kk++) sum -= A(i, kk) * b[kk * ldb + j];
                    b[i * ldb + j] = diag == Diag.Unit ? sum : sum / A(i, i);
                }
        }
    }

    [Fact]
    public void Trsm_FP64_LeftLowerNoTransNonUnit_SingleRhs_MatchesReference()
    {
        const int m = 5, n = 1;
        var rng = new Random(42);
        double[] a = new double[m * m];
        double[] b = new double[m * n];
        // Lower-triangular A with a strong diagonal so it is well-conditioned.
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j <= i; j++) a[i * m + j] = rng.NextDouble() * 2 - 1;
            a[i * m + i] += m; // dominant diagonal
        }
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] expected = (double[])b.Clone();
        ReferenceTrsmLeft(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1.0, a, m, expected, n);

        double[] actual = (double[])b.Clone();
        BlasManagedLib.Trsm<double>(
            Side.Left, Uplo.Lower, transA: false, Diag.NonUnit,
            m, n, 1.0, a, m, actual, n);

        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(expected[i], actual[i], 10); // 10 decimal places
    }
}
```

- [ ] **Step 2: Run the test to verify it fails to compile / fail**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~TrsmTests" -f net10.0`
Expected: **compile error** `'BlasManaged' does not contain a definition for 'Trsm'` (the method does not exist yet). This confirms the test exercises the new API.

---

## Task 4: Implement a correct scalar `Trsm<T>` (make the test pass)

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Trsm.cs`

This first implementation is **scalar and correct, not yet fast** — TDD's "minimal code to pass." The blocked/macrokernel optimization is Task 7, layered in behind the same tests.

- [ ] **Step 1: Write the scalar implementation**

Create `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Trsm.cs`:

```csharp
using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Triangular solve. Computes op(A)·X = α·B (Left) or X·op(A) = α·B (Right),
    /// overwriting B with X. A is the <paramref name="uplo"/> triangle of an
    /// m×m (Left) or n×n (Right) matrix; op(A) is A or Aᵀ per <paramref name="transA"/>.
    /// Drop-in for cblas_strsm/cblas_dtrsm (row-major; order is fixed RowMajor).
    /// </summary>
    public static void Trsm<T>(
        Side side, Uplo uplo, bool transA, Diag diag,
        int m, int n, T alpha,
        ReadOnlySpan<T> a, int lda,
        Span<T> b, int ldb,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (m <= 0 || n <= 0) return;
        var ops = MathHelper.GetNumericOperations<T>();

        // Scale B by alpha (BLAS semantics: solve against alpha·B).
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                int idx = i * ldb + j;
                b[idx] = ops.Multiply(b[idx], alpha);
            }

        if (side == Side.Left)
            TrsmLeftScalar(uplo, transA, diag, m, n, a, lda, b, ldb, ops);
        else
            TrsmRightScalar(uplo, transA, diag, m, n, a, lda, b, ldb, ops);
    }

    private static void TrsmLeftScalar<T>(
        Uplo uplo, bool transA, Diag diag, int m, int n,
        ReadOnlySpan<T> a, int lda, Span<T> b, int ldb,
        AiDotNet.Tensors.LinearAlgebra.INumericOperations<T> ops) where T : unmanaged
    {
        // Effective triangle after optional transpose.
        bool lower = (uplo == Uplo.Lower) ^ transA;

        // A(r,c) honoring transpose.
        T A(int r, int c) => transA ? a[c * lda + r] : a[r * lda + c];

        if (lower)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    T sum = b[i * ldb + j];
                    for (int kk = 0; kk < i; kk++)
                        sum = ops.Subtract(sum, ops.Multiply(A(i, kk), b[kk * ldb + j]));
                    b[i * ldb + j] = diag == Diag.Unit ? sum : ops.Divide(sum, A(i, i));
                }
        }
        else
        {
            for (int i = m - 1; i >= 0; i--)
                for (int j = 0; j < n; j++)
                {
                    T sum = b[i * ldb + j];
                    for (int kk = i + 1; kk < m; kk++)
                        sum = ops.Subtract(sum, ops.Multiply(A(i, kk), b[kk * ldb + j]));
                    b[i * ldb + j] = diag == Diag.Unit ? sum : ops.Divide(sum, A(i, i));
                }
        }
    }

    private static void TrsmRightScalar<T>(
        Uplo uplo, bool transA, Diag diag, int m, int n,
        ReadOnlySpan<T> a, int lda, Span<T> b, int ldb,
        AiDotNet.Tensors.LinearAlgebra.INumericOperations<T> ops) where T : unmanaged
    {
        // Right solve X·op(A) = B with A n×n. Effective triangle after transpose.
        bool lower = (uplo == Uplo.Lower) ^ transA;
        T A(int r, int c) => transA ? a[c * lda + r] : a[r * lda + c];

        if (!lower)
        {
            // X·U = B : columns left-to-right.
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++)
                {
                    T sum = b[i * ldb + j];
                    for (int kk = 0; kk < j; kk++)
                        sum = ops.Subtract(sum, ops.Multiply(b[i * ldb + kk], A(kk, j)));
                    b[i * ldb + j] = diag == Diag.Unit ? sum : ops.Divide(sum, A(j, j));
                }
        }
        else
        {
            // X·L = B : columns right-to-left.
            for (int j = n - 1; j >= 0; j--)
                for (int i = 0; i < m; i++)
                {
                    T sum = b[i * ldb + j];
                    for (int kk = j + 1; kk < n; kk++)
                        sum = ops.Subtract(sum, ops.Multiply(b[i * ldb + kk], A(kk, j)));
                    b[i * ldb + j] = diag == Diag.Unit ? sum : ops.Divide(sum, A(j, j));
                }
        }
    }
}
```

- [ ] **Step 2: Verify `INumericOperations<T>` has `Subtract`/`Multiply`/`Divide`**

Run: `grep -nE "T (Subtract|Multiply|Divide)\(" src/AiDotNet.Tensors/LinearAlgebra/INumericOperations.cs`
Expected: all three method signatures print. (They are the standard generic-math ops used throughout the codebase. If the namespace of `INumericOperations<T>` differs, fix the fully-qualified name in the file accordingly — confirm with `grep -rn "interface INumericOperations" src/`.)

- [ ] **Step 3: Run the test to verify it passes**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~Trsm_FP64_LeftLowerNoTransNonUnit_SingleRhs_MatchesReference" -f net10.0`
Expected: `Passed!  - Failed: 0, Passed: 1`.

- [ ] **Step 4: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Trsm.cs tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmTests.cs
git commit -m "feat(#379): scalar-correct BlasManaged.Trsm<T> + first correctness test"
```

---

## Task 5: Expand the correctness matrix (all Side×Uplo×Trans×Diag, multi-RHS, FP32)

**Files:**
- Modify: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmTests.cs`

- [ ] **Step 1: Add the parameterized full-matrix test**

Append to `TrsmTests.cs` (inside the class). The reference for Right-side is added here; reuse `ReferenceTrsmLeft` for Left.

```csharp
    private static void ReferenceTrsmRight(
        Uplo uplo, bool transA, Diag diag, int m, int n,
        double alpha, double[] a, int lda, double[] b, int ldb)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) b[i * ldb + j] *= alpha;
        double A(int r, int c) => transA ? a[c * lda + r] : a[r * lda + c];
        bool lower = (uplo == Uplo.Lower) ^ transA;
        if (!lower)
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++)
                {
                    double sum = b[i * ldb + j];
                    for (int kk = 0; kk < j; kk++) sum -= b[i * ldb + kk] * A(kk, j);
                    b[i * ldb + j] = diag == Diag.Unit ? sum : sum / A(j, j);
                }
        else
            for (int j = n - 1; j >= 0; j--)
                for (int i = 0; i < m; i++)
                {
                    double sum = b[i * ldb + j];
                    for (int kk = j + 1; kk < n; kk++) sum -= b[i * ldb + kk] * A(kk, j);
                    b[i * ldb + j] = diag == Diag.Unit ? sum : sum / A(j, j);
                }
    }

    public static System.Collections.Generic.IEnumerable<object[]> FullMatrix()
    {
        foreach (var side in new[] { Side.Left, Side.Right })
        foreach (var uplo in new[] { Uplo.Upper, Uplo.Lower })
        foreach (var trans in new[] { false, true })
        foreach (var diag in new[] { Diag.NonUnit, Diag.Unit })
            yield return new object[] { side, uplo, trans, diag };
    }

    [Theory]
    [MemberData(nameof(FullMatrix))]
    public void Trsm_FP64_FullCoverage_MultiRhs_MatchesReference(
        Side side, Uplo uplo, bool trans, Diag diag)
    {
        const int m = 7, n = 4;            // multi-RHS
        int triDim = side == Side.Left ? m : n;
        var rng = new Random(123);
        double[] a = new double[triDim * triDim];
        for (int i = 0; i < triDim; i++)
        {
            for (int j = 0; j < triDim; j++)
                if ((uplo == Uplo.Upper && j >= i) || (uplo == Uplo.Lower && j <= i))
                    a[i * triDim + j] = rng.NextDouble() * 2 - 1;
            a[i * triDim + i] += triDim; // dominant diagonal
        }
        double[] b = new double[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] expected = (double[])b.Clone();
        if (side == Side.Left)
            ReferenceTrsmLeft(side, uplo, trans, diag, m, n, 1.0, a, triDim, expected, n);
        else
            ReferenceTrsmRight(uplo, trans, diag, m, n, 1.0, a, triDim, expected, n);

        double[] actual = (double[])b.Clone();
        BlasManagedLib.Trsm<double>(side, uplo, trans, diag, m, n, 1.0, a, triDim, actual, n);

        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(expected[i], actual[i], 9);
    }

    [Fact]
    public void Trsm_FP32_LeftLower_MatchesReference()
    {
        const int m = 6, n = 3;
        var rng = new Random(7);
        float[] a = new float[m * m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j <= i; j++) a[i * m + j] = (float)(rng.NextDouble() * 2 - 1);
            a[i * m + i] += m;
        }
        float[] b = new float[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // FP64 reference for accuracy.
        double[] a64 = Array.ConvertAll(a, x => (double)x);
        double[] e64 = new double[b.Length];
        for (int i = 0; i < b.Length; i++) e64[i] = b[i];
        ReferenceTrsmLeft(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1.0, a64, m, e64, n);

        float[] actual = (float[])b.Clone();
        BlasManagedLib.Trsm<float>(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1f, a, m, actual, n);

        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(e64[i], actual[i], 3); // FP32: 3 decimal places
    }

    [Fact]
    public void Trsm_Alpha_ScalesRightHandSide()
    {
        const int m = 4, n = 2;
        var rng = new Random(99);
        double[] a = new double[m * m];
        for (int i = 0; i < m; i++) { for (int j = 0; j <= i; j++) a[i*m+j] = rng.NextDouble(); a[i*m+i] += m; }
        double[] b = new double[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble();

        double[] expected = (double[])b.Clone();
        ReferenceTrsmLeft(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 2.5, a, m, expected, n);
        double[] actual = (double[])b.Clone();
        BlasManagedLib.Trsm<double>(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 2.5, a, m, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 9);
    }
```

- [ ] **Step 2: Run the full TRSM test class**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~TrsmTests" -f net10.0`
Expected: `Passed!` — 16 (full matrix) + 1 (FP32) + 1 (alpha) + 1 (original) = 19 tests pass, 0 fail.

- [ ] **Step 3: Commit**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmTests.cs
git commit -m "test(#379): full Side×Uplo×Trans×Diag + FP32 + alpha coverage for Trsm"
```

---

## Task 6: Determinism test across thread counts

**Files:**
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmDeterminismTests.cs`

The scalar impl ignores `NumThreads`, so this passes trivially now and acts as the **guard** that Task 7's parallel optimization must not break.

- [ ] **Step 1: Write the determinism test**

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmDeterminismTests.cs`:

```csharp
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class TrsmDeterminismTests
{
    [Theory]
    [InlineData(64, 32)]
    [InlineData(128, 16)]
    [InlineData(256, 8)]
    public void Trsm_FP64_BitExactAcrossThreadCounts(int m, int n)
    {
        var rng = new Random(42);
        double[] a = new double[m * m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j <= i; j++) a[i * m + j] = rng.NextDouble() * 2 - 1;
            a[i * m + i] += m;
        }
        double[] b0 = new double[m * n];
        for (int i = 0; i < b0.Length; i++) b0[i] = rng.NextDouble() * 2 - 1;

        double[]? baseline = null;
        foreach (int threads in new[] { 1, 2, 4, 8 })
        {
            double[] actual = (double[])b0.Clone();
            var opts = new BlasOptions<double> { NumThreads = threads, Mode = BlasMode.Deterministic };
            BlasManagedLib.Trsm<double>(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1.0, a, m, actual, n, opts);
            if (baseline is null) baseline = actual;
            else
                for (int i = 0; i < actual.Length; i++)
                    Assert.Equal(baseline[i], actual[i]); // EXACT bit equality, no tolerance
        }
    }
}
```

- [ ] **Step 2: Run it**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~TrsmDeterminismTests" -f net10.0`
Expected: `Passed!` 3 tests, 0 fail.

- [ ] **Step 3: Commit**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmDeterminismTests.cs
git commit -m "test(#379): Trsm bit-exact determinism guard across thread counts"
```

---

## Task 7: Blocked TRSM with GEMM-macrokernel trailing updates (the optimization)

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Trsm.cs`

Replace the trailing inner loops with blocked panels: solve a small diagonal block with the scalar kernel, then subtract its contribution from the remaining RHS using the existing `PackBothStrategy.Run` GEMM macrokernel. The scalar kernel from Task 4 stays as the per-block solver and the small-`m` fast path.

- [ ] **Step 1: Read the macrokernel entry signature you will call**

Run: `sed -n '59,73p' src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/PackBothStrategy.cs`
Expected: prints the `public static unsafe void Run<T>(...)` parameter list. Confirm the parameter order (a, lda, transA, b, ldb, transB, c, ldc, m, n, k, mc, nc, kc, mr, nr, …) before writing the call. If the trailing tuning params (mc/nc/kc/mr/nr) are non-trivial to supply directly, call the public `BlasManaged.Gemm<T>` dispatcher instead (it selects them) — see Step 2's note.

- [ ] **Step 2: Add the blocked driver**

In `BlasManaged.Trsm.cs`, add a blocked path used when `m` (Left) or `n` (Right) exceeds a block threshold, and route `Trsm` to it. Use the public `Gemm<T>` for trailing updates to avoid duplicating tile-size selection (it computes `C := A·B`; we need `B -= A·X`, so compute into a scratch tile and subtract — or pass a negative-alpha equivalent via a temporary). Concrete Left-Lower-NoTrans implementation:

```csharp
    // Block size for the diagonal-solve / trailing-update split. 64 keeps the
    // diagonal block in L1 while giving the GEMM macrokernel a worthwhile panel.
    private const int TrsmBlock = 64;

    private static void TrsmLeftLowerNoTransBlocked<T>(
        Diag diag, int m, int n,
        ReadOnlySpan<T> a, int lda, Span<T> b, int ldb,
        in BlasOptions<T> options,
        AiDotNet.Tensors.LinearAlgebra.INumericOperations<T> ops) where T : unmanaged
    {
        for (int i0 = 0; i0 < m; i0 += TrsmBlock)
        {
            int bm = Math.Min(TrsmBlock, m - i0);

            // 1) Solve the bm×n diagonal block with the scalar kernel.
            //    The block's own triangle is A[i0:i0+bm, i0:i0+bm].
            var aDiag = a.Slice(i0 * lda + i0);
            var bBlock = b.Slice(i0 * ldb);
            TrsmLeftScalar(Uplo.Lower, false, diag, bm, n, aDiag, lda, bBlock, ldb, ops);

            // 2) Trailing update: for rows below this block,
            //    B[i0+bm:, :] -= A[i0+bm:, i0:i0+bm] · X[i0:i0+bm, :]
            int remRows = m - (i0 + bm);
            if (remRows > 0)
            {
                var aSub = a.Slice((i0 + bm) * lda + i0); // remRows × bm panel (lda-strided)
                var xBlk = b.Slice(i0 * ldb);              // bm × n (just solved), lda=ldb
                // scratch = aSub · xBlk  (remRows × n)
                T[] scratch = new T[remRows * n];
                Gemm<T>(aSub, lda, false, xBlk, ldb, false, scratch, n, remRows, n, bm,
                        new BlasOptions<T> { NumThreads = options.NumThreads, Mode = options.Mode });
                var bRem = b.Slice((i0 + bm) * ldb);
                for (int r = 0; r < remRows; r++)
                    for (int c = 0; c < n; c++)
                    {
                        int bi = r * ldb + c;
                        bRem[bi] = ops.Subtract(bRem[bi], scratch[r * n + c]);
                    }
            }
        }
    }
```

Then route to it from `TrsmLeftScalar`'s caller. In `Trsm<T>`, replace the `side == Side.Left` branch with:

```csharp
        if (side == Side.Left)
        {
            bool lower = (uplo == Uplo.Lower) ^ transA;
            // Blocked path only for the canonical Left-Lower-NoTrans case in this
            // phase; all other combinations use the proven scalar kernel. Later
            // tasks extend blocking to upper/transposed cases.
            if (m > TrsmBlock && uplo == Uplo.Lower && !transA && lower)
                TrsmLeftLowerNoTransBlocked(diag, m, n, a, lda, b, ldb, options, ops);
            else
                TrsmLeftScalar(uplo, transA, diag, m, n, a, lda, b, ldb, ops);
        }
        else
            TrsmRightScalar(uplo, transA, diag, m, n, a, lda, b, ldb, ops);
```

> **Note on the trailing-update direction:** the trailing GEMM operates only on rows *below* the just-solved block and on the off-diagonal sub-panel `A[i0+bm:, i0:i0+bm]`, which is fully populated (not triangular), so the dense `Gemm` is exact. Determinism is preserved because `Gemm` in `BlasMode.Deterministic` is itself bit-exact across threads (verified by the existing `DeterminismTests`), and the subtraction order is fixed.

- [ ] **Step 3: Run the full correctness + determinism suite (must still pass)**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~Trsm" -f net10.0`
Expected: `Passed!` — all TRSM correctness (19) + determinism (3) tests pass. The blocked path is now exercised by the `256×8` determinism case and the larger correctness cases.

- [ ] **Step 4: Add a large-shape correctness case that forces the blocked path**

Append to `TrsmTests.cs`:

```csharp
    [Fact]
    public void Trsm_FP64_LargeLeftLower_BlockedPath_MatchesReference()
    {
        const int m = 200, n = 5; // m > TrsmBlock(64) → blocked
        var rng = new Random(2024);
        double[] a = new double[m * m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j <= i; j++) a[i * m + j] = rng.NextDouble() * 2 - 1;
            a[i * m + i] += m;
        }
        double[] b = new double[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] expected = (double[])b.Clone();
        ReferenceTrsmLeft(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1.0, a, m, expected, n);
        double[] actual = (double[])b.Clone();
        BlasManagedLib.Trsm<double>(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1.0, a, m, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 8);
    }
```

- [ ] **Step 5: Run it**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~Trsm_FP64_LargeLeftLower" -f net10.0`
Expected: `Passed!` 1 test.

- [ ] **Step 6: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Trsm.cs tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TrsmTests.cs
git commit -m "perf(#379): blocked Trsm Left-Lower path reuses GEMM macrokernel for trailing updates"
```

---

## Task 8: Bench catalog + perf-bar stub

**Files:**
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/Catalog/SpecializedShapeCatalog.cs`
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/SpecializedPerfBar.cs`

- [ ] **Step 1: Confirm the existing `ShapeCatalog` record shape to mirror it**

Run: `sed -n '1,40p' tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/Catalog/ShapeCatalog.cs`
Expected: prints the `Shape` record / catalog structure. Mirror its `(Name, M, N, K, …, Frequency, Source)` convention in the new file (adapt fields to TRSM: `Side`, `Uplo`, `M`, `N` instead of `K`).

- [ ] **Step 2: Write the TRSM catalog**

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/Catalog/SpecializedShapeCatalog.cs`:

```csharp
using AiDotNet.Tensors.Engines.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;

/// <summary>Per-variant bench shapes for the #379 specialized BLAS variants.</summary>
public static class SpecializedShapeCatalog
{
    public record TrsmShape(string Name, Side Side, Uplo Uplo, bool TransA, Diag Diag,
        int M, int N, bool Fp64, int Frequency, string Source);

    public static readonly TrsmShape[] Trsm =
    {
        new("Chol_Solve_64x1",   Side.Left, Uplo.Lower, false, Diag.NonUnit,  64,   1, true,  50, "workload:cholesky-solve"),
        new("Chol_Solve_256x1",  Side.Left, Uplo.Lower, false, Diag.NonUnit, 256,   1, true,  40, "workload:cholesky-solve"),
        new("QR_BackSub_256x64", Side.Left, Uplo.Upper, false, Diag.NonUnit, 256,  64, true,  30, "workload:qr-backsub"),
        new("Solve_MultiRhs_512x128", Side.Left, Uplo.Lower, false, Diag.NonUnit, 512, 128, true, 20, "workload:linsolve"),
        new("Chol_SolveT_256x1", Side.Left, Uplo.Lower, true,  Diag.NonUnit, 256,   1, true,  25, "workload:cholesky-solve-transpose"),
        new("Solve_FP32_512x64", Side.Left, Uplo.Lower, false, Diag.NonUnit, 512,  64, false, 15, "workload:linsolve-fp32"),
    };
}
```

- [ ] **Step 3: Write the perf-bar stub**

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/SpecializedPerfBar.cs`:

```csharp
namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Frozen per-variant perf bars for #379. Values are placeholders until the first
/// authoritative bench run on the self-hosted runner (AIDOTNET_PERF_RUNNER=1) lands,
/// at which point the project owner sets them in a single gating commit — same
/// discipline as <see cref="PerfBar"/> for dense GEMM (#368).
/// </summary>
public static class SpecializedPerfBar
{
    // TRSM vs OpenBLAS strsm/dtrsm on the authoritative runner.
    public const int    TrsmMinWinRatePercent = 0;     // TO BE SET after first bench
    public const double TrsmMaxLossMultiple    = 99.0; // TO BE SET after first bench
    public const string TargetHardwareFingerprint = ""; // captured from runner

    /// <summary>True once the owner has frozen the TRSM bar (non-zero win rate).</summary>
    public static bool TrsmBarFrozen => TrsmMinWinRatePercent > 0;
}
```

> The `0` / `99.0` placeholders are **intentional and documented** (not a plan gap): the bar is set only after real measurement, exactly as `PerfBar.cs` does for dense GEMM. `TrsmBarFrozen` lets the perf-regression test skip until then.

- [ ] **Step 4: Build the test project**

Run: `dotnet build tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --no-restore -f net10.0`
Expected: `Build succeeded.` 0 errors.

- [ ] **Step 5: Commit**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/Catalog/SpecializedShapeCatalog.cs tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/SpecializedPerfBar.cs
git commit -m "test(#379): TRSM bench catalog + frozen-after-measurement perf-bar stub"
```

---

## Task 9: Rewire `LinearSolvers.SolveTriangularInternal` to use `BlasManaged.Trsm`

**Files:**
- Modify: `src/AiDotNet.Tensors/LinearAlgebra/Solvers/LinearSolvers.cs:129-163`

The existing `TriangularSolveSingle` stays (batched / non-FP32-FP64 fallback + oracle). For the common non-batched FP32/FP64 2D case, delegate to `BlasManaged.Trsm`.

- [ ] **Step 1: Read the current `TriangularSolveSingle` to match its triangle/transpose semantics exactly**

Run: `grep -n "TriangularSolveSingle" src/AiDotNet.Tensors/LinearAlgebra/Solvers/LinearSolvers.cs`
then `sed -n '<line>,<line+60>p'` over its body.
Expected: confirms `upper`/`transpose`/`unitDiagonal` flag meanings map to `Uplo`/`transA`/`Diag` as: `upper → Uplo.Upper`, `transpose → transA`, `unitDiagonal → Diag.Unit`. (Triangular solve is always Left, alpha=1, here.)

- [ ] **Step 2: Add the delegation in `SolveTriangularInternal`**

In the per-batch loop (`LinearSolvers.cs` ~line 156-160), replace the body of the `for (int bi …)` loop with a typed delegation when `T` is `float`/`double` and the input is non-batched-friendly. Concretely, after the existing setup and before the loop, add:

```csharp
        // Fast path: route FP32/FP64 to the managed BLAS Trsm kernel (#379).
        // Triangular solve is always Left side with alpha = 1.
        bool useBlas = (typeof(T) == typeof(double) || typeof(T) == typeof(float));
        if (useBlas)
        {
            var trsmUplo = upper ? AiDotNet.Tensors.Engines.BlasManaged.Uplo.Upper
                                 : AiDotNet.Tensors.Engines.BlasManaged.Uplo.Lower;
            var trsmDiag = unitDiagonal ? AiDotNet.Tensors.Engines.BlasManaged.Diag.Unit
                                        : AiDotNet.Tensors.Engines.BlasManaged.Diag.NonUnit;
            for (int bi = 0; bi < batch; bi++)
            {
                var aSlice = new ReadOnlySpan<T>(aData, bi * aStride, n * n);
                var xSlice = new Span<T>(xData, bi * xStride, nrhs == 1 ? n : n * nrhs);
                AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Trsm<T>(
                    AiDotNet.Tensors.Engines.BlasManaged.Side.Left,
                    trsmUplo, transpose, trsmDiag,
                    n, nrhs, MathHelper.GetNumericOperations<T>().One,
                    aSlice, n, xSlice, nrhs);
            }
            return x;
        }

        for (int bi = 0; bi < batch; bi++)
        {
            TriangularSolveSingle(
                aData, bi * aStride,
                xData, bi * xStride,
                n, nrhs, upper, transpose, unitDiagonal);
        }

        return x;
```

> Confirm `MathHelper.GetNumericOperations<T>().One` exists with `grep -n "T One" src/AiDotNet.Tensors/LinearAlgebra/INumericOperations.cs`; if the property is named differently (e.g. `FromDouble(1.0)`), use that instead.

- [ ] **Step 3: Run the existing linear-solver test suite (the canary)**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~Solve|FullyQualifiedName~Triangular|FullyQualifiedName~Cholesky|FullyQualifiedName~Lstsq" -f net10.0`
Expected: `Passed!` — all existing solver/Cholesky/Lstsq tests still pass. Any failure here means the flag mapping in Step 1 is wrong; fix the `Uplo`/`transA`/`Diag` translation, do not weaken the test.

- [ ] **Step 4: Commit**

```bash
git add src/AiDotNet.Tensors/LinearAlgebra/Solvers/LinearSolvers.cs
git commit -m "perf(#379): route FP32/FP64 triangular solve through BlasManaged.Trsm"
```

---

## Task 10: Multi-target build verification + full BlasManaged suite

**Files:** none (verification only)

- [ ] **Step 1: Build all target frameworks**

Run: `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --no-restore`
Expected: `Build succeeded.` for net10.0 and net471 (the multi-target set). If net471 errors on a span/`MathHelper` API, gate or adjust — net471 lacks some APIs (see CLAUDE.md note on `GC.GetAllocatedBytesForCurrentThread`).

- [ ] **Step 2: Run the full BlasManaged + solver test surface**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~BlasManaged|FullyQualifiedName~Trsm|FullyQualifiedName~Solve|FullyQualifiedName~Cholesky" -f net10.0`
Expected: `Passed!` 0 failures.

- [ ] **Step 3: Final commit if any net471 gating was needed; otherwise nothing to commit.**

```bash
git add -A
git commit -m "build(#379): TFM compatibility for Trsm path" || echo "nothing to commit"
```

---

## Self-Review (completed by plan author)

**Spec coverage (P0+P1 portion):**
- §3 `Side`/`Uplo`/`Diag` + `SparseLayout<T>` → Task 2 ✓
- §3 `Trsm` signature (drop-in CBLAS arg order) → Task 4 ✓
- §4 TRSM blocked driver reusing GEMM macrokernel → Task 7 ✓
- §4 TRSM determinism (Left independent RHS, GEMM trailing updates bit-exact) → Task 6 + Task 7 note ✓
- §5 bench catalog + frozen perf bar → Task 8 ✓
- §5 full Side×Uplo×Trans×Diag correctness matrix → Task 5 ✓
- §6 P1 call-site rewiring (`LinearSolvers`) with existing tests as canary → Task 9 ✓
- §6 partial-file-per-variant structure → Task 1 ✓

**Out of this plan (subsequent phase plans):** SYRK (P2), SYMM (P3), CPU SpMM (P4), GBMV (P5), GPU SpMM (P6), vendor removal (P7), and TRSM blocking for the non-Left-Lower cases (the scalar kernel handles them correctly meanwhile). Each gets its own `docs/superpowers/plans/` document once this template lands.

**Placeholder scan:** the only `0`/`99.0`/`""` placeholders are in `SpecializedPerfBar.cs`, intentional and documented (frozen-after-measurement discipline from #368). No `TODO`/`TBD`/"implement later" in code steps.

**Type consistency:** `Trsm<T>` signature is identical in Task 4 (definition), Task 3/5/6 (tests), and Task 9 (call site). Enum names `Side`/`Uplo`/`Diag` consistent throughout. `TrsmBlock`, `TrsmLeftScalar`, `TrsmRightScalar`, `TrsmLeftLowerNoTransBlocked` referenced consistently between Tasks 4 and 7.

**Verification-before-completion notes embedded:** every code task ends with a `dotnet test`/`dotnet build` step and an explicit expected result; `grep` confirmation steps guard the two assumptions that could be wrong (`INumericOperations<T>` method names, `PackBothStrategy.Run` signature, `TriangularSolveSingle` flag semantics).
