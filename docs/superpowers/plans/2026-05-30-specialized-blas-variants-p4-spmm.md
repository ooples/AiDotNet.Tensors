# Specialized BLAS Variants — P4 (CPU SpMM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Checkbox (`- [ ]`) steps.

**Goal:** Add `BlasManaged.SpMM<T>` — sparse×dense `C = α·A·B + β·C` (CSR/CSC) — replacing the cache-hostile naive loop in `CpuSparseEngine.SpMM`, and rewire that call site.

**Architecture:** New standalone kernel (not GEMM-reuse). CSR: pre-scale `C` row by `β`, then for each nonzero `(i,k,v)` accumulate `C[i,0:n] += (α·v)·B[k,0:n]` — B rows read contiguously (cache-friendly), rows independent (deterministic, parallelizable). CSC iterates columns scattering into C. Both fixed-order → bit-exact. Typed-SIMD + row-parallel are tracked perf follow-ups; this deliverable is the correct cache-friendly consolidation + the rewire.

**Spec:** §4 CPU SpMM. **Branch:** `feature/379-specialized-blas-variants`.

## Files
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.SpMM.cs`, `tests/.../SpMMTests.cs`, `tests/.../SpMMDeterminismTests.cs`
- Modify: `src/AiDotNet.Tensors/Engines/CpuSparseEngine.cs:126-167` (rewire), `SpecializedShapeCatalog.cs`, `SpecializedPerfBar.cs`

## Tasks
1. Failing CSR correctness test (oracle = densify A then dense multiply with α/β). Expect `CS0117 'SpMM'`.
2. Implement `SpMM` (CSR + CSC, α/β pre-scale + accumulate). Build + first test green. Commit.
3. Coverage: CSC test, FP32, α/β, empty-row; determinism across thread counts (serial impl → trivially bit-exact, guards future parallelization). Commit.
4. Rewire `CpuSparseEngine.SpMM` → build `SparseLayout` from `ToCsr()` + `dense.AsSpan()`/`result.AsWritableSpan()`; existing sparse tests are the canary. Commit.
5. Catalog + perf-bar (`SpMMShape`/`SpMM*` bar). Multi-TFM build. Commit.

## Tracked follow-ups (not silently dropped)
Typed FP32/FP64 SIMD inner loop + CSR row-parallel (`NumThreads`) + fused `Epilogue` (bias+activation) — the headline exceed-vendor levers; tracked for focused commits on this branch.

## Self-Review
§4 CSR row-local + CSC → Task 2 ✓; determinism → Task 3 ✓; rewire with canary → Task 4 ✓; catalog/bar → Task 5 ✓. SIMD/parallel/epilogue explicitly tracked, not dropped.
