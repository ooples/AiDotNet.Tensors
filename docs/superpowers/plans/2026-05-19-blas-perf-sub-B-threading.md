# Sub-issue B Implementation Plan — Threading wire-up

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Issue:** #370
**Parent:** #368 (mega tracking issue)
**Goal:** Wire the existing `MN2DDriver`, `KAxisDriver`, `ReductionTree` primitives into `PackAOnlyStrategy` and `StreamingStrategy`; introduce `BlasMode { Deterministic, Fast }` so K-axis non-associative reduction can be opt-in.

**Baseline (from PR #381 `artifacts/perf/baseline.json` on x64-amd-avx2-cpu16):**
- 0/54 wins, median 23.5× slower. Threading wire-up alone won't hit 80% bar but should close the gap on Wide-N / Tall-K / mid-square shapes.

## What's already there vs. what's missing

| Primitive | State | This sub-issue |
|-----------|-------|----------------|
| `AxisSelector.Select(...)` → enum `{None, M, N, K, MN_2D}` | Exists | Plumb its result into every strategy |
| `KAxisDriver` | Exists | Wire into `StreamingStrategy` (Fast mode only) |
| `MN2DDriver` | Exists | Wire into `PackBothStrategy` as alternative to M-axis |
| `ReductionTree` | Exists | Used by `KAxisDriver` for deterministic pairwise sum |
| `PackBothStrategy` M-axis Parallel.For | Hardcoded inline | Will route via `AxisSelector` and possibly use 2D grid |
| `PackAOnlyStrategy` | Fully serial | Add N-axis Parallel.For (split `jc` loop) |
| `StreamingStrategy` | Fully serial | Add M-axis split AND K-axis (Fast mode) |
| `BlasMode` enum | Missing | Create |
| `BlasOptions<T>.Mode` field | Missing | Add (default `Deterministic`) |
| `BlasManaged.DefaultMode` | Missing | Add static |

## Tasks

### B.1 — `BlasMode` enum + `BlasOptions<T>.Mode` field + `BlasManaged.DefaultMode`

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasMode.cs`
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs` (add `Mode` field, default `Deterministic`)
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs` (add `DefaultMode` static)
- Test: new unit test in `Tests/Engines/BlasManaged/BlasModeTest.cs`

**Acceptance:**
- Code compiles on net10.0 + net471
- Default is `Deterministic` everywhere (no opt-out by accident)
- Existing tests still pass (no behavior change yet)

### B.2 — N-axis parallel in `StreamingStrategy`

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs`
- Test: perf assertion in `Tests/Engines/BlasManaged/StreamingNAxisParallelTest.cs`

**Approach:** Add a split-on-N wrapper that partitions N into per-thread chunks and dispatches per-chunk Streaming kernels (each thread writes a disjoint column slice of C — no synchronization needed). Gate on `AxisSelector.Select(...) == ParallelismAxis.N`.

**Acceptance:**
- Streaming with wide-N (e.g., M=64, N=4096, K=512) is ≥2× faster at 8 threads than at 1 thread
- Determinism: bit-exact across 1/2/4/8 thread counts (no reduction order changes; each thread is fully independent)

### B.3 — N-axis parallel in `PackAOnlyStrategy`

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/PackAOnlyStrategy.cs`
- Test: extend existing tests

**Approach:** Parallelize the outer `pc` loop's inner `jc` (N) loop when AxisSelector picks N. Pack-A is shared (read-only inside parallel region). Each thread writes a disjoint column slice of C.

**Acceptance:**
- PackAOnly with wide-N shape is ≥2× faster at 8 threads
- Determinism preserved (column slices are independent)

### B.4 — K-axis parallel in `StreamingStrategy` (Fast mode only)

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs`
- Test: `Tests/Engines/BlasManaged/StreamingKAxisFastModeTest.cs`

**Approach:** Partition K into per-thread chunks; each thread computes a partial C; reduce via `KAxisDriver` + `ReductionTree`. Only enabled when `options.Mode == BlasMode.Fast` because the reduction tree's pairwise-sum can produce different ULP results across thread counts.

**Acceptance:**
- StreamingFast tall-K shape (e.g., M=32, N=32, K=4096) is ≥2× faster at 8 threads
- BlasMode.Deterministic refuses K-axis split (stays serial or uses M/N)

### B.5 — 2D MN-grid in `PackBothStrategy`

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/PackBothStrategy.cs`

**Approach:** When `AxisSelector` picks `MN_2D`, route through `MN2DDriver` instead of the hardcoded M-axis Parallel.For. Each thread owns a 2D tile of C and packs its own A panel + B stripe.

**Acceptance:**
- Big-square shape (1024×1024×1024) at 8 threads is ≥1.5× faster than pure M-axis when M is short (e.g., 256×2048×512)

### B.6 — Strategy-side dispatch via `AxisSelector`

**Files:**
- Modify: all three strategy files
- Add helper: `src/AiDotNet.Tensors/Engines/BlasManaged/Parallelism/StrategyDispatch.cs`

**Approach:** Centralized helper that calls `AxisSelector.Select` once and returns the chosen driver, which each strategy then invokes. Replaces ad-hoc per-strategy parallelism with a uniform routing layer.

**Acceptance:**
- All three strategies use the same dispatch path
- Removing/changing AxisSelector heuristic only requires editing one place

### B.5 — 2D MN-grid in PackBothStrategy (deferred to follow-up)

**Decision (2026-05-19):** **Defer** to a follow-up PR after baseline data shows MN_2D is actually a bottleneck. Rationale:

- `AxisSelector.Select` picks `MN_2D` only when both M and N axes individually have too few blocks (`m_blocks * 2 < procs` AND `n_blocks > 1`). For the actual baseline catalog shapes, this is rare:
  - Most BERT/ResNet shapes have M ≥ 256 so M-axis fires
  - LSTM/Embedding/Attention have M=1 but huge N — N-axis fires
  - Genuine 2D-grid cases need M and N both around 32–64, which the catalog lacks
- Implementation cost is high: `PackBothStrategy.RunParallelUnsafe` does M-axis split AFTER packing B once per `jc`; converting to 2D grid requires either (a) packing all B-blocks upfront (~K*N elemsize extra memory) and threading over (jc, ic) tiles, or (b) accepting redundant per-thread B-packing. Either way is a meaningful rewrite of the parallel path.
- The MN2DDriver primitive remains available for a future PR when a real bottleneck surfaces.

`PackBothStrategy` keeps its current hardcoded M-axis parallel path. AxisSelector wire-up is also deferred — current code doesn't consult the selector.

### B.6 — Centralize dispatch via StrategyDispatch helper (deferred to follow-up)

**Decision (2026-05-19):** **Defer**. Three callers with slightly different (mr, nr) constants don't justify a wrapper class. When Sub-F's autotune cache lands, the dispatch becomes a cache lookup, making the centralization moot anyway.

### B.7 — `AxisSelector` autotune backing (deferred follow-up)

**Decision:** **Defer** to a separate sub-PR or to Sub-issue F (the routing shim) which already uses the autotune cache. Wiring autotune into AxisSelector here would entangle B with F prematurely. AxisSelector stays heuristic in this PR; autotune adoption is a 1-PR follow-up.

## Order of execution

```
B.1 (enum)
  ↓
B.2 (Streaming N-axis) ← simplest first, baseline measurement
  ↓
B.3 (PackAOnly N-axis)
  ↓
B.5 (PackBoth 2D grid)
  ↓
B.4 (Streaming K-axis, Fast mode)
  ↓
B.6 (centralize via StrategyDispatch helper)
```

B.4 is later because K-axis is the trickiest (reduction tree + Fast-mode gating). B.6 last because it's a refactor that depends on B.2-B.5 all landing.

## Verification

After each task: `dotnet test --filter Engines.BlasManaged` shows no regressions, plus the new perf assertion for that task passes.

After all tasks: re-run baseline harness, expect mid-N and tall-K shapes to move from 30–100× slower to 5–15× slower (still far from the 1× bar, but the threading lever is fully pulled).
