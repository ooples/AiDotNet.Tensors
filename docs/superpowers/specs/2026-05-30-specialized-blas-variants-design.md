# Specialized BLAS Variants (TRSM / SYRK / SYMM / GBMV / SpMM) — Design Spec

**Date:** 2026-05-30
**Status:** Approved (pending writing-plans phase)
**Issue:** [#379](https://github.com/ooples/AiDotNet.Tensors/issues/379) — Sparse / banded / symmetric / triangular BLAS variants
**Predecessor:** [#368](https://github.com/ooples/AiDotNet.Tensors/issues/368) — BlasManaged perf sprint (general dense GEMM only)
**Predecessor spec:** [`2026-05-17-blas-managed-perf-sprint-design.md`](./2026-05-17-blas-managed-perf-sprint-design.md)
**Branch:** `feature/379-specialized-blas-variants`

## 1 — Motivation & corrected framing

Issue #379 was tracked as "these specialized variants still route to native BLAS until this issue closes." Investigation of the current tree shows that premise is **inaccurate on the CPU side**:

- `BlasProvider.cs` (CPU) only P/Invokes `cblas_sgemm` / `cblas_dgemm` (+ batch/bf16). There are **no** native `strsm` / `ssyrk` / `ssymm` / `sgbmv` declarations anywhere.
- Triangular solve (TRSM) today = managed substitution loops in `LinearSolvers.cs` and the decomposition classes (Cholesky / QR / LU).
- Sparse SpMM today = fully managed **naive scalar loops** in `CpuSparseEngine.cs` (no SIMD, no parallelism).
- The only *native* sparse bindings (`cuSPARSE` / `rocSPARSE`) are **GPU-only**. The #368 spec (§7) explicitly deferred GPU BLAS.

Therefore there is **no native CPU supply-chain to remove** here. The real work is:

1. Establish **first-class, optimized managed implementations** of each variant inside the BlasManaged framework (microkernels, packing, allocator, `BlasMode` determinism contract), and rewire the existing ad-hoc managed call sites to them.
2. Treat it as a **perf effort** mirroring #368: extend the bench catalog, set frozen per-variant bars against the relevant vendor routine, and beat them.
3. For **GPU SpMM** (the one place a real native dep exists): make the *already-existing* custom CUDA/HIP sparse kernels the default, close the perf gap to parity-or-better vs cuSPARSE/rocSPARSE, then delete the vendor binding.

### Scope (user-confirmed, maximal)

All five families **plus** GPU sparse, in **one phased mega-PR**:
TRSM, SYRK, SYMM, GBMV, CPU SpMM (CSR/CSC), GPU SpMM (cuSPARSE/rocSPARSE replacement).

The size/review cost of a single-PR approach was flagged twice; the maintainer chose one mega-PR with phased, independently-green commits. This spec structures the work so each phase is reviewable on its own.

### Two goals, two correctness gates (inherited from #368)

- **Goal 1 — exceed industry standards** (not just match). See §3 for the levers vendors structurally cannot match.
- **Goal 2 — first-class consolidated APIs** replacing the scattered ad-hoc managed implementations.
- **Gate A — bit-exact determinism** stays the default (`BlasMode.Deterministic`); `BlasMode.Fast` is the opt-in escape hatch.
- **Gate B — per-variant perf bar** vs the relevant vendor routine, frozen after first authoritative bench run on the self-hosted runner.

## 2 — Architecture: Approach A (reuse the GEMM microkernel core)

The three level-3 dense-symmetric/triangular ops are **thin drivers over the existing** packing + microkernel + parallel-driver stack from #366/#368. Only the sparse and banded ops get genuinely new kernels. This matches how production BLAS (BLIS) builds these — level-3 variants are wrappers over the GEMM macrokernel.

```
┌───────────────────────────────────────────────────────────────┐
│ Public API: BlasManaged.{Trsm,Syrk,Symm,Gbmv,SpMM}<T>          │
├───────────────────────────────────────────────────────────────┤
│ BlasOptions<T> (Mode, Epilogue, packing, threads) — unchanged  │
├───────────────────────────────────────────────────────────────┤
│ Variant drivers:                                                │
│   Trsm  → blocked solve + GEMM macrokernel (trailing updates)  │
│   Syrk  → GEMM macrokernel + tile-skip + triangular-write       │
│   Symm  → GEMM macrokernel + mirror-on-pack A-packer            │
│   Gbmv  → NEW standalone banded level-2 kernel                  │
│   SpMM  → NEW CSR/CSC kernel (SIMD over dense-N, row-parallel)  │
├───────────────────────────────────────────────────────────────┤
│ Existing GEMM core: microkernels (AVX512/AVX2/NEON), packing,  │
│ allocator, parallel drivers, reduction tree — REUSED            │
└───────────────────────────────────────────────────────────────┘
```

**New-code surface is intentionally small:** one triangular-solve microkernel (TRSM diagonal tile), one mirror-on-pack routine (SYMM), one triangular-write mask (SYRK), one banded kernel (GBMV), one CSR/CSC SpMM kernel, plus GPU dispatch/perf work. Everything else is existing infrastructure.

### Alternatives considered

- **Approach B — standalone kernel per variant:** maximum per-variant tuning ceiling but 5× the kernel surface to test and keep deterministic; re-solves problems the GEMM core already solved. Rejected as too risky for one PR.
- **Approach C — hybrid:** collapses into A (A already makes SpMM/GBMV standalone).

## 3 — Public API surface

All entry points live on the existing `static class BlasManaged`, mirror `Gemm<T>`'s `ReadOnlySpan<T>` + leading-dimension + `in BlasOptions<T>` shape, and pull `Mode` from `BlasOptions` / `DefaultMode`. Argument order matches the canonical CBLAS reference **exactly** (we drop only the `order` enum, fixed to RowMajor, identical to the existing `Gemm`).

```csharp
// Triangular solve:  op(A)·X = α·B (left) or X·op(A) = α·B (right). In-place on B.
public static void Trsm<T>(
    Side side, Uplo uplo, bool transA, Diag diag,
    int m, int n, T alpha,
    ReadOnlySpan<T> a, int lda,
    Span<T> b, int ldb,
    in BlasOptions<T> options = default) where T : unmanaged;

// Symmetric rank-k update:  C = α·op(A)·op(A)^T + β·C   (writes only `uplo` triangle)
public static void Syrk<T>(
    Uplo uplo, bool trans,
    int n, int k, T alpha,
    ReadOnlySpan<T> a, int lda, T beta,
    Span<T> c, int ldc,
    in BlasOptions<T> options = default) where T : unmanaged;

// Symmetric matrix multiply:  C = α·A·B + β·C  (A symmetric, stored in `uplo` triangle)
public static void Symm<T>(
    Side side, Uplo uplo,
    int m, int n, T alpha,
    ReadOnlySpan<T> a, int lda,
    ReadOnlySpan<T> b, int ldb, T beta,
    Span<T> c, int ldc,
    in BlasOptions<T> options = default) where T : unmanaged;

// Banded matrix-vector:  y = α·op(A)·x + β·y   (A banded, kl sub- / ku super-diagonals)
public static void Gbmv<T>(
    bool transA, int m, int n, int kl, int ku, T alpha,
    ReadOnlySpan<T> a, int lda,
    ReadOnlySpan<T> x, int incx, T beta,
    Span<T> y, int incy,
    in BlasOptions<T> options = default) where T : unmanaged;

// Sparse×dense:  C = α·A_sparse·B + β·C   (CSR or CSC, format read from the view)
public static void SpMM<T>(
    T alpha, SparseLayout<T> a,
    ReadOnlySpan<T> b, int ldb, int n, T beta,
    Span<T> c, int ldc,
    in BlasOptions<T> options = default) where T : unmanaged;
```

**Supporting types** (per CLAUDE.md's no-string-for-closed-sets rule):
`Side { Left, Right }`, `Uplo { Upper, Lower }`, `Diag { NonUnit, Unit }`.
`SparseLayout<T>` — a thin readonly ref-struct view (rowPtr / colInd / values spans + format enum) so `SpMM` stays allocation-free and decoupled from the heap `SparseTensor<T>` class.

### Industry-standard validation

Confirmed against the CBLAS reference ([OpenBLAS `cblas.h`](https://github.com/xianyi/OpenBLAS/blob/develop/cblas.h), [GSL CBLAS](https://www.gnu.org/software/gsl/doc/html/cblas.html)):

| Op | Reference CBLAS arg order | Our signature | Match |
|----|---------------------------|---------------|-------|
| TRSM | `(Side, Uplo, TransA, Diag, M, N, α, A, lda, B, ldb)` | identical | ✓ |
| SYRK | `(Uplo, Trans, N, K, α, A, lda, β, C, ldc)` | identical | ✓ |
| SYMM | `(Side, Uplo, M, N, α, A, lda, B, ldb, β, C, ldc)` | identical | ✓ |
| GBMV | `(TransA, M, N, KL, KU, α, A, lda, x, incx, β, y, incy)` | identical | ✓ |

SpMM has no classic CBLAS; the signature aligns with the modern **MKL Sparse BLAS IE** `mkl_sparse_s_mm` and **cuSPARSE generic** `cusparseSpMM` conventions.

### How we *exceed* the standard (no signature changes — all rides `BlasOptions<T>`)

1. **Fused epilogue (headline win).** Vendor TRSM/SYRK/SYMM/SpMM do one operation; bias/activation needs a separate pass. Our ops inherit `BlasOptions<T>.Epilogue`, so `SpMM + bias + ReLU` is a single fused kernel — exactly the GNN/recsys hot path (the GPU side already proves it with `csr_spmm_bias_relu`). SYRK/SYMM/SpMM/TRSM all honor `options.Epilogue`.
2. **Bit-exact determinism by default.** OpenBLAS/MKL/cuSPARSE are not reproducible across thread counts; our `BlasMode.Deterministic` default is.
3. **One generic `<T>` API** vs the vendors' 4× `s/d/c/z` entry points, with FP32/FP64 specialization under the hood.
4. **FMA-on-FP32 in Fast mode** (per #368 §5) — numerically better than OpenBLAS (which skips FP32 FMA for NumPy compat) *and* faster.

## 4 — Per-variant algorithm & determinism design

### TRSM — reuses GEMM macrokernel; one new tiny kernel
Blocked BLIS-style solve. Partition the triangular dimension into `Mr`-sized diagonal blocks. For each: solve the small `Mr×Nr` diagonal tile with a **new triangular-solve microkernel** (the only genuinely new code), then update the trailing RHS via the **existing GEMM macrokernel** (`B -= A_offdiag · X_solved`). Reuses PackBoth packing.
**Determinism:** substitution is inherently ordered along the triangular axis; rank-k updates use GEMM's existing pairwise reduction; parallelism is over independent RHS columns (`N`) → bit-exact in both modes. No new determinism risk.

### SYRK — `C = α·A·Aᵀ + β·C` — reuses GEMM core; tile-skip + triangular-write
B is just Aᵀ, so packing A yields the B-panel for free. The outer loop **skips microkernel tiles** on the off-`uplo` side of the diagonal (≈half the FLOPs); diagonal tiles apply a **triangular-write mask** in the epilogue so only the `uplo` triangle is stored.
**Determinism:** identical to GEMM (pairwise tree in Deterministic, FMA/free-order in Fast).

### SYMM — `C = α·A·B + β·C`, A symmetric — reuses GEMM core; mirror-on-pack
A is stored in one triangle. The **A-packing routine reflects across the diagonal on the fly** (reads the stored triangle, mirrors into the packed panel). After packing it is a bit-for-bit ordinary GEMM through the existing macro/microkernel.
**Determinism:** identical to GEMM.

### GBMV — `y = α·op(A)·x + β·y`, A banded — new standalone level-2 kernel
Memory-bound GEMV-class, not a GEMM. Band storage `(kl+ku+1)×n`. For each column the band touches rows `[j-ku, j+kl]`; SIMD runs along the band. Rides `BlasOptions` parallelism (N-axis split) with its own kernel.
**Determinism:** non-trans output elements are independent → deterministic. Trans case accumulates into `y`; Deterministic mode uses fixed-order (column-ascending) accumulation, Fast allows reorder.

### CPU SpMM — `C = α·A_sparse·B + β·C` — new kernel (the big optimization vs naive loops)
- **CSR (fast/primary path):** parallelize over sparse rows `i` (M-axis). For each nonzero `(i,k,v)`: `C[i, 0:N] += v · B[k, 0:N]`, **SIMD-vectorized along the dense N dimension** (broadcast `v`, FMA into the C row). Rows are independent → embarrassingly parallel.
- **CSC:** iterate columns, scatter into C. Deterministic mode uses per-thread partials merged in fixed row order; Fast mode allows thread-private accumulation. Autotune may instead transpose-to-CSR once when the matrix is reused.
**Determinism:** CSR row-parallel is **naturally bit-exact** (fixed nonzero traversal order per row, no cross-thread reduction) — a strict improvement over both the current scalar loop and vendor sparse libs.
**Exceed lever:** fuse `+ bias + activation` from `options.Epilogue` into the row write — single pass.

### GPU SpMM — make existing custom kernels the default; close perf gap; remove vendor dep
Custom CUDA kernels already exist (`CudaSparseKernels.cs`: `csr_spmm`, `csr_spmm_warp`, `csr_spmm_bias_relu`, bit-deterministic `*_deterministic` variants). Work: (a) flip `IDirectGpuBackend.CsrSpMM` default from `CuSparseBackend` (native) → custom kernels; (b) add the perf pieces to hit the parity bar (warp-per-row dispatch heuristic, vectorized 128-bit dense loads, FP64 coverage); (c) mirror gaps to HIP/`RocSparse`; (d) delete the cuSPARSE/rocSPARSE P/Invoke + probe after the bar is met.
**Determinism:** Deterministic mode → `*_deterministic` kernels (fixed-order segment sums); Fast mode → atomic-scatter variants.

## 5 — Verification: perf bars, bench catalog, determinism & correctness

Reuses #368 scaffolding (`ShapeCatalog.cs`, `PerfHarness.cs`, `PerfBar.cs`, `PerfBarTest.cs`, `DeterminismTests.cs`).

### Bench catalog extension
Add a `SpecializedShapeCatalog` (or extend `ShapeCatalog`) with per-variant shape sets, each tagged `frequency` + `source`:
- **TRSM:** Cholesky/QR solve shapes (`64×64`, `256×256`, `1024×64` multi-RHS) + ML linear-system solves.
- **SYRK/SYMM:** covariance/gram shapes (`N×K` with K≫N for SYRK; square for SYMM).
- **SpMM:** GNN/recsys shapes `(rows, cols, nnz%, N)` — Cora/PubMed-like 0.1–2% density + a dense-ish stress case.
- **GBMV:** tridiagonal/pentadiagonal band shapes.

### Per-variant frozen perf bars (`SpecializedPerfBar.cs`)
Constants set after the first authoritative bench run on the self-hosted runner; gate-then-frozen discipline from #368 (X/Y/N structure: min win-rate %, max loss multiple, catalog size, target hardware fingerprint).
- **TRSM/SYRK/SYMM:** vs OpenBLAS `strsm/ssyrk/ssymm` — median ≤ OpenBLAS on ≥ X% of shapes.
- **CPU SpMM:** vs MKL Sparse BLAS `mkl_sparse_s_mm` where available, else vs the current naive managed loop (which is the floor — must be a large multiple faster).
- **GPU SpMM:** vs cuSPARSE `cusparseSpMM` — parity-or-better is the explicit gate; the vendor binding is **not deleted until this bar is green** on the runner.

### Determinism gates
Extend `DeterminismTests.cs` with a case per new op asserting bit-exact equality across 1/2/4/8 thread counts in `BlasMode.Deterministic`. **A variant's PR phase does not merge until its determinism coverage is added.** Fast mode may diverge ±1–2 ULP; Deterministic may not.

### Correctness tests
Each variant gets a `*Tests.cs` comparing `BlasManaged.X` to a reference (native `cblas_*`/`cusparseSpMM` where the dep still loads, else a naive reference) within tolerance, with a full shape-coverage matrix: all `Side×Uplo×Trans×Diag` for TRSM, both `Uplo×Trans` for SYRK, CSR+CSC for SpMM, trans/non-trans + band widths for GBMV. Epilogue-fusion paths get their own cases (`SpMM+bias+ReLU == SpMM then bias then ReLU`).

Perf bars stay **skipped on GitHub-hosted runners** and only assert on the self-hosted `AIDOTNET_PERF_RUNNER=1` box, as `PerfBarTest` already does.

## 6 — Phasing, call-site rewiring, and vendor removal

One branch/PR, landing as ordered, independently-green phases (builds + that phase's tests + its determinism coverage pass before the next starts).

| Phase | Deliverable | Gated by |
|-------|-------------|----------|
| **P0** | Shared scaffolding: `Side`/`Uplo`/`Diag` enums, `SparseLayout<T>`, `SpecializedShapeCatalog`, `SpecializedPerfBar` stub, determinism-harness extension points | builds |
| **P1** | **TRSM** — triangular-solve microkernel + blocked driver + tests/determinism + rewire `LinearSolvers`/Cholesky/QR/LU solve loops | P0 |
| **P2** | **SYRK** — tile-skip + triangular-write epilogue + tests/determinism + rewire covariance/gram call sites | P1 |
| **P3** | **SYMM** — mirror-on-pack A-packer + tests/determinism + rewire | P1 |
| **P4** | **CPU SpMM** — CSR row-parallel + CSC + SIMD-over-N + fused epilogue + rewire `CpuSparseEngine.SpMM` (sparse×dense) | P0 |
| **P5** | **GBMV** — banded level-2 kernel + tests/determinism + rewire banded/tridiagonal call sites | P0 |
| **P6** | **GPU SpMM** — flip default to custom kernels, close perf gap (warp-per-row, vectorized loads, FP64), mirror HIP/ROCm | P4 |
| **P7** | **Vendor removal** — delete cuSPARSE/rocSPARSE P/Invoke + probes + package refs, only after P6's parity bar is green | P6 bar green |

TRSM is P1 because it proves the macrokernel-reuse template every later dense variant copies.

### Call-site rewiring discipline
Mechanical and additive (like #368's routing shim): existing managed loops are replaced by `BlasManaged.X` calls; the existing test suite that exercises those paths is the canary. No call site changes its public behavior.

### Escape hatches (carried from #368 §7)
- If a variant can't hit its bar on a class of shapes, autotune routes those shapes to the vendor/native path rather than shipping a regression.
- For GPU, **P7 (vendor deletion) is deferred, not forced**, if P6's bar isn't met — keeping the mega-PR from being held hostage by the single hardest variant. Each deferral is an explicit, visible decision in the issue thread.

## 7 — Out of scope

- LAPACK-level routines (cholesky/qr/svd/lu factorization themselves — only their *triangular-solve* steps are in scope via TRSM).
- FP16/BF16/INT8 specialized kernels (FP32/FP64 only this PR).
- Block-sparse (BSR/BSC) SpMM — `SparseTensor<T>` supports the storage, but blocked SpMM kernels are a follow-up; CSR/CSC are primary here.
- Sparse×sparse (`SpSpMM`) — a different algorithm (sparse×sparse, not sparse×dense) than the `SpMM` kernel this PR adds. `CpuSparseEngine.SpSpMM` is left as-is; only the SIMD-vectorized SpMM (sparse×dense) primitives it internally relies on (if any) benefit indirectly. No new SpSpMM kernel (CPU or GPU) in this PR.

## 8 — Decisions captured

| Decision | Choice |
|----------|--------|
| Architecture | Approach A — reuse GEMM microkernel core for TRSM/SYRK/SYMM; standalone for GBMV/SpMM |
| Goal | Both first-class APIs **and** perf optimization (exceed vendors, not just match) |
| Scope | All 5 CPU families + GPU SpMM |
| Determinism | Same `BlasMode` contract as dense GEMM (Deterministic default, opt-in Fast) |
| GPU sparse | In scope — custom kernels parity-or-better, then remove cuSPARSE/rocSPARSE |
| Delivery | One phased mega-PR, independently-green phases |
| API | Drop-in CBLAS arg order; exceed via fused epilogue + determinism + generic T |

## 9 — Next steps

1. Spec self-review (inline fixes).
2. User reviews spec — gate before writing-plans.
3. writing-plans skill produces the phased implementation plan keyed off this spec.
