# Specialized BLAS Variants — P5 (GBMV) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Checkbox (`- [ ]`) steps.

**Goal:** Add `BlasManaged.Gbmv<T>` — banded matrix-vector `y = α·op(A)·x + β·y` — drop-in for cblas_sgbmv/cblas_dgbmv, using the standard LAPACK band storage.

**Architecture:** Standalone level-2 kernel (memory-bound, not GEMM). Band storage (LAPACK convention, column-major band, lda ≥ kl+ku+1): the logical element `A(i,j)` lives at `a[j*lda + (ku - j + i)]` for the band `max(0,j-ku) ≤ i ≤ min(m-1,j+kl)`. Each output element is an independent fixed-order dot over the band → deterministic in both modes.

**Spec:** §4 GBMV. **Branch:** `feature/379-specialized-blas-variants`.

## Files
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.Gbmv.cs`, `tests/.../GbmvTests.cs`, `tests/.../GbmvDeterminismTests.cs`
- Modify: `SpecializedShapeCatalog.cs`, `SpecializedPerfBar.cs`

## Tasks
1. Failing test: oracle builds dense banded A + band array consistently, computes `y` densely; non-trans first. Expect `CS0117 'Gbmv'`.
2. Implement `Gbmv` (non-trans + trans, α/β, incx/incy). Build + first test green. Commit.
3. Coverage: trans, FP32, α/β, non-square m≠n, incx/incy≠1; determinism across thread counts. Commit.
4. Catalog (tridiagonal/pentadiagonal) + perf-bar stub; multi-TFM build. Commit.

## Self-Review
§4 banded level-2 + LAPACK storage → Task 2 ✓; trans + strides → Task 3 ✓; determinism (independent fixed-order dots) → Task 3 ✓; catalog/bar → Task 4 ✓.
