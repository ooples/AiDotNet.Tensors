# BlasManaged Perf Sprint + Supply-Chain Removal — Design Spec

**Date:** 2026-05-17
**Status:** Approved (pending writing-plans phase)
**Predecessor:** [PR #366](https://github.com/ooples/AiDotNet.Tensors/pull/366) — BlasManaged infrastructure (74 commits, draft)
**Predecessor spec:** [`2026-05-16-blas-managed-design.md`](./2026-05-16-blas-managed-design.md)

## 1 — Motivation & success criteria

PR #366 built the BLIS-style managed GEMM infrastructure (microkernels, packing, autotune, allocator, epilogue chain) but a comparative benchmark against native OpenBLAS shows BlasManaged loses on all 12 measured shapes by 1.5×–69×. The original goals — eliminate the MKL/OpenBLAS supply-chain attack surface AND beat them on perf — are not yet met. This perf sprint exists to close the gap and complete the removal.

### Two non-negotiable goals

1. **Perf parity-or-better against OpenBLAS.** Bar is set against an expanded benchmark catalog (Section 4), not the current 12 shapes. Both the catalog and the bar get codified before optimization work starts so we cannot sandbag later.
2. **Complete supply-chain removal.** All 144 `BlasProvider.*` call sites across 20 files end up calling BlasManaged. The native P/Invoke `cblas_*` entry points get deleted. The csproj no longer references any native BLAS package or DLL.

### Two layered correctness gates that ride along

- **Gate 3 (bit-exact determinism)** stays the *default*. Cross-thread-count identical output is preserved for `BlasMode.Deterministic` (the default).
- **`BlasMode.Fast`** is an opt-in escape hatch that allows non-associative reduction, FMA-on-FP32, and instruction reorders. Perf bench is measured against Fast mode; correctness regression tests stay in Deterministic.

### What success means specifically

- Sub-issue A (Section 3) delivers the expanded bench and a written bar with concrete numerator/denominator (e.g., "BlasManaged median runtime ≤ OpenBLAS median runtime × 1.0 on ≥ X of N shapes; max(BlasManaged/OpenBLAS) ≤ Y on the remaining shapes"). X, Y, N filled in *after* bench data lands.
- Sub-issue G (Section 3) is mergeable only when (a) bar from Sub-issue A is met on Fast mode and (b) Gate 3 still passes on Deterministic mode.

## 2 — Architecture: how the sprint integrates with what exists

The current PR #366 tree already has every primitive this sprint needs as building blocks. The sprint does not add new top-level subsystems — it fills in the orthogonal optimization vectors that the existing infrastructure was scaffolded for but didn't actually exercise.

### The existing layer cake (from PR #366) and what each layer gets in this sprint

```
┌─────────────────────────────────────────────────────────────────┐
│ Public API: BlasManaged.Gemm<T>(...)                            │  Stable — no signature changes
├─────────────────────────────────────────────────────────────────┤
│ BlasOptions<T>                                                  │  + BlasMode { Deterministic, Fast }
├─────────────────────────────────────────────────────────────────┤
│ AutotuneDispatcher                                              │  + per-shape strategy + per-arch tile size
├─────────────────────────────────────────────────────────────────┤
│ Strategies: PackBoth | PackAOnly | Streaming                    │  + N-axis & 2D parallel wired into all 3
├─────────────────────────────────────────────────────────────────┤
│ Parallel drivers: MAxis | NAxis | KAxis | MN2D | AxisSelector   │  Existing primitives → finally used
├─────────────────────────────────────────────────────────────────┤
│ Microkernels: Avx512 | Avx2 | Neon | Scalar  (FP32, FP64)       │  + per-arch tile widths, prefetch, unsafe
├─────────────────────────────────────────────────────────────────┤
│ Allocator: 5 layers (PerThread/Pool/WeightPack/Arena/Carver)    │  + actually adopt WeightPack at call sites
├─────────────────────────────────────────────────────────────────┤
│ Jit: JittedKernelCache + NativeAotDetector  (scaffolded)        │  + emit shape-specialized kernels for hot shapes
└─────────────────────────────────────────────────────────────────┘
```

### Where the supply-chain change lives

A single new layer slots in just above `BlasProvider`:

```
Caller (e.g. CpuEngine.cs, MatrixMultiplyHelper.cs, ...)
   │
   ▼
BlasProvider.TryGemm / TryGemmEx        ← becomes a thin routing shim
   │
   ├─► BlasManaged.Gemm   (when autotune says it wins on this shape)
   │
   └─► native cblas_*     (transitional fallback; deleted in Sub-issue G)
```

By landing the shim in `BlasProvider` itself, none of the 20 caller files get touched during the perf sprint — the routing decision is made once, in one place. Sub-issue G is then a mechanical deletion.

### What is explicitly NOT being rebuilt

- The packing layout (`packedA` vpanel `[Kc, Mr]` order) — fixed in commit `b97c655`, stays as-is.
- The microkernel public interfaces (`IMicrokernel<T>`) — additions only, no breaking changes to existing kernels.
- The autotune cache file format (`~/.aidotnet/autotune/*.json`) — additions only, backward-compatible.
- The strategy hierarchy — three strategies stay (PackBoth/PackAOnly/Streaming); we wire parallelism into the existing two that lack it, not add new strategies.
- The bit-exact determinism contract — same primitives produce same results when `Mode = Deterministic`.

### The "we already have it, we just need to use it" observation

Phase G of PR #366 (#G1–G7) built KAxisDriver, MN2DDriver, AxisSelector, ReductionTree, and all the multi-axis primitives — but only PackBoth uses M-axis split. Sub-issue B is "finish wiring G primitives into PackAOnly + Streaming + multi-axis selection". This is much smaller than building from scratch.

## 3 — Sub-issues, dependencies, and what each PR contains

Seven sub-issues parent off the mega tracking issue.

### Dependency graph

```
                                  ┌──────────────────────────┐
                                  │  #A  Bench catalog       │
                                  │   (gates everything)     │
                                  └────────────┬─────────────┘
                                               │
              ┌────────────────┬───────────────┼───────────────┬────────────────┐
              ▼                ▼               ▼               ▼                ▼
       ┌────────────┐   ┌────────────┐  ┌────────────┐  ┌────────────┐   ┌────────────┐
       │ #B Threading│  │ #C Small-  │  │ #D Per-arch│  │ #E Pre-pack│   │ (review)   │
       │  wire-up   │  │  shape path│  │  microkernel│ │  adoption  │   │            │
       └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘   └────────────┘
             │                │               │                │
             └────────────────┴───────┬───────┴────────────────┘
                                      ▼
                            ┌─────────────────────┐
                            │  #F Routing shim    │
                            │  in BlasProvider    │
                            └──────────┬──────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  #G Native removal  │
                            │  (delete cblas_*)   │
                            └─────────────────────┘
```

### Sub-issue A — Benchmark catalog & measurement infrastructure

- **Output:** `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeCatalog.cs` containing 50–80 shapes with `(name, M, N, K, transA, transB, dtype, frequency, source)`.
- **Two sources merged + deduped:** (a) instrumentation logger added to `BlasProvider.TryGemm/TryGemmEx` that records shapes during the full test suite + integration tests; (b) curated standard-ML shape list (BERT-base FFN/attention, ResNet50 conv-as-GEMM, GPT-2 medium projection, MobileNet depthwise).
- **`PerfHarness.cs`** that runs each shape through both BlasManaged and OpenBLAS, captures median + p95 over warmup + iters, writes JSON output to `artifacts/perf/<git-sha>.json`.
- **Bar codification:** after first full bench run lands, write the bar numbers (X / Y / N from Section 1) into `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBar.cs` as constants that the regression test reads.

### Sub-issue B — Threading wire-up

- Wire N-axis split into `PackAOnlyStrategy` and `StreamingStrategy` using existing `NAxisDriver` (PR #366 only built it, never connected it).
- Wire 2D MN-grid into `PackBothStrategy` as an alternative to pure M-axis when both M and N are large.
- Wire K-axis reduction tree into `StreamingStrategy` for tall-K shapes (where M·N is small but K is huge) — uses existing `ReductionTree` for deterministic pairwise sum.
- `AxisSelector` heuristic gets the autotune cache as a backing store (currently it's pure shape arithmetic).
- **Bar:** at least 3× speedup on shapes where prior single-axis was the bottleneck (Tall-K, Wide-N, mid-square at 8+ threads).

### Sub-issue C — Small-shape streaming path (pack-free)

- New code path `BlasManaged.Gemm` selects when `M·N·K < threshold` (autotune-determined) that **skips packing entirely** and runs the microkernel directly against the user's strided A/B buffers.
- Tail handling reuses the existing `Avx2Tail` / `Avx512Tail` masked-store kernels.
- Eliminates the allocator round-trip that currently dominates Tiny_32sq / Tiny_64sq / WideFat_512×512×64.
- **Bar:** Tiny_32sq, Tiny_64sq, WideFat_512×512×64 each ≤ 1.5× OpenBLAS (currently 30×, 30×, 69×).

### Sub-issue D — Per-arch microkernel tuning + JIT IL emission

- Add architecture-specific tile widths and unrolling:
  - **Zen3/Zen4:** Mr=12, Nr=4 (FP64) / Mr=8, Nr=8 (FP32). Matches AMD's 12-register pressure sweet spot.
  - **Sapphire Rapids / Granite Rapids:** Mr=8, Nr=24 (FP32) using all 32 ZMM registers.
  - **Apple M-series:** Mr=8, Nr=12 (FP32) leveraging 32 Neon-FP regs.
- Add `Sse.Prefetch0/1/2/NTA` calls inside packing loops and microkernel inner loops for memory-latency hiding.
- Convert hot microkernel inner loops to **unsafe + pointer arithmetic** (eliminates Span bounds checks).
- Activate `JittedKernelCache` from PR #366 Phase J for the top-N hot shapes — emit a shape-specialized microkernel with K-loop fully unrolled. Falls back to generic kernel on NativeAOT (already detected).
- **Bar:** on top-10 most-frequent shapes from Sub-issue A catalog, BlasManaged median ≤ OpenBLAS median.

### Sub-issue E — Pre-pack weight cache adoption

- The `WeightPackCache` (PR #366 Phase F, allocator layer 3) exists but no caller uses it yet.
- Identify inference call sites where B is invariant across calls (linear layer projections, embedding lookups, output projections). Add a `PrePackB` warmup pass in those paths.
- Add `BlasManaged.PrePackB(...)` adoption to: `CpuEngine.cs` MatMul, `CompiledTrainingPlan.cs` inference path, `BackwardFunctions.cs` weight-gradient accumulation.
- **Bar:** inference-shape calls (FFN_128×768×768, batched MatMul) show ≥1.5× speedup vs uncached path.

### Sub-issue F — BlasProvider routing shim + autotune dispatch

- Inside `BlasProvider.TryGemm` and `BlasProvider.TryGemmEx`, before native dispatch: query `AutotuneDispatcher.PrefersManaged(shape)`. If yes, route to `BlasManaged.Gemm`. If no, fall through to native.
- The autotune dispatcher learns from runtime measurements: when both paths are exercised at startup (warmup), record which won.
- A `BlasOptions.PreferManaged` global escape hatch (set to `true` for testing or supply-chain-conscious deployments) forces all calls through BlasManaged regardless of autotune.
- **Bar:** Sub-issue F merges only when the bar from Sub-issue A is met on Fast mode with `PreferManaged = true`.

### Sub-issue G — Native removal

- Delete the native P/Invoke declarations in `BlasProvider.NativeMethods.cs`.
- Delete the `BlasProvider.TryLoadOpenBlas` / `TryLoadMkl` discovery logic.
- Delete the csproj `<PackageReference>` for any native BLAS package; delete any `runtime.linux-x64.native.*` references.
- Delete the native conditional code paths in `BlasProvider.TryGemm` / `TryGemmEx`; they now unconditionally call `BlasManaged.Gemm`.
- Update CI: remove any "install OpenBLAS" step. Verify the library builds clean on a host with NO native BLAS installed.
- **Bar:** `dotnet build` and full test suite pass on a freshly-provisioned host with no MKL/OpenBLAS DLLs/.so/.dylib present.

## 4 — Benchmark methodology

### Where measurements run

- Local dev machine: any contributor can run the bench, but local numbers are advisory only.
- Self-hosted runner `ns107444.ip-51-81-109.us` (the user's existing Ubuntu runner): **authoritative**. CPU/AVX features captured in `HardwareFingerprint` and stamped into every result JSON. Bar numbers are set against this hardware.
- GitHub-hosted runners: a *non-perf* smoke test of the bench harness runs there (verify it compiles and produces output), but no perf assertion gates on GitHub-hosted because their CPU is non-deterministic and varies by VM placement.

### Measurement protocol per shape

- Fresh random A, B with seed 42 (already what `ComparativeBenchmark.cs` does — keep this convention).
- Warmup: 3 iters.
- Measure: 10 iters if `M·N·K > 10⁸`, 30 if > 10⁷, 50 otherwise. Same scaling as current `ComparativeBenchmark.cs`.
- Report median (not mean) — robust to outliers. Also capture p95 for tail analysis.
- Stopwatch high-resolution timer; results in milliseconds.
- Run order is randomized per JSON output to defeat any thermal/cache ordering bias.

### The catalog file (Sub-issue A output)

`ShapeCatalog.cs` produces a `Shape[]` with 50–80 entries:

```csharp
public record Shape(
    string Name,           // e.g., "Bert_FFN_3072x768x768"
    int M, int N, int K,
    bool TransA, bool TransB,
    DType Dtype,           // FP32 | FP64
    int Frequency,         // from instrumentation: how often this shape was seen
    string Source);        // "instrumented:CpuEngine.MatMul" | "workload:BERT-base" | "workload:ResNet50"
```

### The bar codification step

After the first full bench run lands, the result JSON is committed to `artifacts/perf/baseline.json`. A test `PerfRegressionTest` reads `PerfBar.cs` constants. **The values below are illustrative placeholders** — actual numbers are filled in by the project owner after Sub-issue A produces real data:

```csharp
public static class PerfBar
{
    public const int    MinWinRatePercent      = 80;   // X%  — TO BE SET after Sub-issue A
    public const double MaxLossMultiple        = 1.20; // Y×  — TO BE SET after Sub-issue A
    public const int    CatalogShapeCount      = 64;   // N   — must match committed catalog size
    public const string TargetHardwareFingerprint = "<runner-fingerprint>"; // captured from authoritative runner
}
```

These constants are written **after** Sub-issue A produces real data, by the project owner, in a single commit that gates the rest of the sprint. The number lands and never moves (except via the escape hatches in Section 7, which require an explicit user-approved commit).

### The regression test (`tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBarTest.cs`)

- Skips on non-self-hosted runners (checks `Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1"`).
- Skips on hosts where `HardwareFingerprint.Current != PerfBar.TargetHardwareFingerprint`.
- Runs the full catalog, computes win-rate and max-loss-multiple, asserts both against `PerfBar`.
- Output JSON is uploaded as a CI artifact every run (so the trend over time is visible).

### Determinism gate (orthogonal to perf gate)

`DeterminismTests.cs` already exists from PR #366 Phase G (#G7). It runs across 1/2/4/8 thread counts on Deterministic mode and asserts bit-exact equality. The perf sprint must not weaken Gate 3 for default mode — only Fast mode is allowed to diverge cross-thread-count.

### What "we hit the bar" looks like in CI output

```
PerfBar: target 80% win rate, no shape > 1.20× OpenBLAS
  64 shapes measured on AMD EPYC 7763 / AVX-512
  BlasManaged wins:        54 / 64  (84.4%)   ✓ ≥ 80%
  Worst loss multiple:     1.14×    ✓ ≤ 1.20×
  Gate 3 (determinism):    PASS     ✓
  → Bar met. Sub-issue G unblocked.
```

## 5 — Determinism, Fast mode, and the `BlasMode` contract

### The enum and its meaning

```csharp
public enum BlasMode
{
    /// <summary>
    /// Default. Output is bit-exact identical across thread counts and across
    /// repeated calls on the same hardware. Allowed transforms: deterministic
    /// pairwise reduction trees only; FMA disabled for FP32 (OpenBLAS-compatible);
    /// no instruction reordering that affects rounding.
    /// </summary>
    Deterministic = 0,

    /// <summary>
    /// Opt-in. Output may differ by ±1-2 ULP across thread counts due to
    /// non-associative reduction order. FMA enabled for FP32 (single rounding
    /// per (a*b+c) instead of two). Instruction reordering by the JIT is allowed.
    /// Numerical accuracy is at least as good as Deterministic mode; the only
    /// thing that changes is bit-exact reproducibility.
    /// </summary>
    Fast = 1,
}
```

### Where Mode lives

`BlasOptions<T>` (the existing ref struct from PR #366) gets a `Mode` field. Default value `Deterministic`. Callers who care set it via `BlasManaged.Gemm(..., new BlasOptions<T> { Mode = BlasMode.Fast })`.

### Codebase-wide default

One global default at `BlasManaged.DefaultMode` (a static property, settable once at startup). Sub-issue F sets this to `Deterministic` so that everything routed through the `BlasProvider` shim stays bit-exact by default; users opting into Fast must do it explicitly.

### What each mode forbids/permits inside the kernels

| Optimization                          | Deterministic | Fast |
|---------------------------------------|---------------|------|
| Pairwise reduction tree (deterministic) | required   | allowed |
| Sequential left-fold accumulation     | forbidden     | allowed |
| FMA on FP32 inside microkernel        | forbidden     | required |
| FMA on FP64 inside microkernel        | allowed (OpenBLAS does it) | required |
| 2D grid parallel with cross-tile sums | requires fixed-order merge | allowed any order |
| K-axis parallel reduction             | requires pairwise-tree merge | allowed |
| Streaming pack-free path              | allowed       | allowed |
| Per-arch tile reorder                 | allowed       | allowed |
| JIT IL-emitted kernels                | allowed (if deterministic-flagged at emit time) | allowed |

The "fixed-order merge" requirement on Deterministic mode means: when M-axis or 2D-grid produces partial C tiles from N threads, those tiles must always be reduced into the final C in the same order (e.g., always thread-id-ascending) regardless of which finishes first. This costs a barrier but preserves bit-exactness.

### Why FMA-on-FP32 is the high-leverage Fast-mode toggle

Without FMA, FP32 GEMM does `tmp = a*b` (round to FP32) → `c += tmp` (round to FP32) — two roundings per macc. With FMA, one rounding per macc — same number of operations dispatched but ~1 ULP better numerically and avoids one rounding-stall on the CPU pipeline. OpenBLAS does not use FMA for FP32 (historical compatibility with older NumPy), so enabling FMA in Fast mode means BlasManaged is numerically *better* than OpenBLAS while running faster.

### How Sub-issues B/C/D adhere to the contract

- Each new optimization PR includes a `BlasMode`-aware test: run the kernel both in Deterministic and Fast, assert Deterministic still passes Gate 3 across thread counts.
- The `DeterminismTests.cs` suite is extended to cover the new code paths added by each sub-issue. Per-sub-issue gate: if the PR introduces a code path that's not covered by Gate 3, it must add the coverage.
- Non-trivial perf optimizations (e.g., 2D grid in PackBoth) get TWO implementations gated by `if (mode == Fast) ... else ...`. This avoids the trap of "we got it fast but determinism is broken so we ship behind a feature flag forever".

## 6 — Migration mechanics: how the routing shim works

Sub-issue F is the smallest-LOC sub-issue but the highest-leverage: it changes the behavior of all 144 call sites without touching any of them.

### Pre-shim state (today)

```csharp
public static bool TryGemm(int m, int n, int k, ...) {
    if (!IsAvailable) return false;
    // ... P/Invoke into cblas_sgemm / cblas_dgemm ...
    return true;
}
```

### Post-shim state (after Sub-issue F)

```csharp
// Illustrative; actual BlasProvider has separate non-generic Single/Double overloads.
// Each overload calls the dtype-specific PrefersManaged variant.
public static bool TryGemm(int m, int n, int k, /* float[]… */ ...) {
    // 1. Decide: managed or native?
    if (BlasOptions.PreferManaged || AutotuneDispatcher.PrefersManagedSingle(m, n, k, transA, transB)) {
        BlasManaged.Gemm<float>(...);   // pulls Mode from BlasManaged.DefaultMode
        return true;
    }
    // 2. Native fallback (deleted in Sub-issue G).
    if (!IsAvailable) return false;
    // ... P/Invoke into cblas_sgemm ...
    return true;
}
```

### How `AutotuneDispatcher.PrefersManaged` learns

It already exists from PR #366 Phase H. Sub-issue F adds one method:

```csharp
// One method per dtype to keep the call hot; pseudocode shows the FP32 path.
public bool PrefersManagedSingle(int m, int n, int k, bool ta, bool tb)
{
    var key = ShapeKey.For(m, n, k, ta, tb, DType.Single, HardwareFingerprint.Current);

    if (_cache.TryGet(key, out var decision)) return decision.PrefersManaged;

    // First-call measurement is amortized once per (shape, hardware) tuple.
    var managedMs = MeasureManagedSingle(m, n, k, ta, tb);
    var nativeMs  = BlasProvider.IsAvailable ? MeasureNativeSingle(m, n, k, ta, tb) : double.MaxValue;
    var prefersManaged = managedMs <= nativeMs;
    _cache.Put(key, new Decision { PrefersManaged = prefersManaged, ManagedMs = managedMs, NativeMs = nativeMs });
    return prefersManaged;
}
```

### Why this is safe to land before perf is perfect

When BlasManaged is slower for a given shape, autotune sees that on its first measurement and routes that shape to native. The codebase keeps running at native speed for shapes BlasManaged hasn't caught up on yet. The perf sprint progressively moves shapes from the "native wins" column to the "managed wins" column. **No regression risk** from landing Sub-issue F early.

### The `BlasOptions.PreferManaged` global toggle

Set this to `true` in two contexts:
1. In the `PerfBarTest` so the bench actually exercises BlasManaged regardless of autotune.
2. In supply-chain-conscious deployments (the original motivation) where the user wants *no* native call regardless of perf.

When `PreferManaged = true`, autotune is bypassed and every call goes through `BlasManaged.Gemm`.

### Caller-side behavior diff

Zero. The 144 call sites continue to call `BlasProvider.TryGemm` with the same signature. They neither know nor care that the bytes are now flowing through managed code. Git blame on those call sites does not show this sprint.

### The Sub-issue G mechanical removal

Once the bar is met:
1. Delete `BlasProvider.NativeMethods.cs` (the P/Invoke declarations).
2. Keep `BlasProvider.IsAvailable` returning `true` permanently — the existing if-branches just always take the success path. Zero caller edits.
3. Delete `<PackageReference Include="OpenBLAS.Native" ... />` (or whatever the native dep is) from the csproj.
4. Delete the CI "install OpenBLAS" step.
5. The diff for Sub-issue G is small: maybe 200 lines deleted, 0 added. Reviewer's job is to confirm "nothing references native BLAS anymore" via `grep -r cblas_ src/`.

### The receipt that supply-chain is actually gone

- `grep -ri "openblas\|mkl\|cblas_" src/ --include="*.cs"` returns zero matches.
- `dotnet list package --include-transitive` shows no native BLAS package.
- A fresh container with `dotnet build` and `dotnet test` succeeds — no `LoadLibrary`/`dlopen` of any native BLAS happens at runtime (verified by a startup-trace test).

## 7 — Out of scope, risks, and recovery

### Explicitly out of scope (deferred to follow-up issues, not this mega-issue)

- **LAPACK-level routines** (cholesky, qr, svd, lu). `SvdDecomposition.cs` calls into native LAPACK functions, not just `cblas_gemm`. This sprint only removes the *BLAS* dependency. A follow-up issue tracks LAPACK replacement.
- **GPU BLAS** (cuBLAS, rocBLAS, MPS). The DirectGpu subsystem has its own native bindings — supply-chain concerns there are tracked separately and not bundled in.
- **FP16/BF16/INT8 microkernels.** Current scope is FP32/FP64 only. Mixed-precision is a different sprint with its own performance characteristics.
- **Sparse GEMM, banded GEMM, symmetric/triangular variants** (sspr, ssyr, strmm). Only general-matrix dense GEMM is in scope.
- **Tensor-core / AMX path.** Sapphire Rapids has AMX tiles for FP32/BF16/INT8 GEMM at 2-8× peak FMA throughput. AMX requires Linux 5.16+ kernel and OS XSAVE state enable. Out of scope; tracked as follow-up.
- **Auto-vectorization via the C# JIT alone.** Every microkernel is hand-coded with intrinsics.

### Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sub-issue A bench catalog turns out to need 200+ shapes to be defensible | Medium | Adds a week to Sub-issue A | Cap at 80 shapes; explicitly state "representative not exhaustive"; defer rare shapes to follow-up |
| Microkernel hand-tuning hits a ceiling below OpenBLAS on some arch | Medium | Bar not met on that arch | Drop the win-rate bar for that arch alone; route to native for affected shapes via autotune (the shim makes this safe) |
| JIT IL emission breaks NativeAOT or trimming | Low (detector exists) | NativeAOT users can't use BlasManaged | `NativeAotDetector` already gates IL emission off on AOT; verified by `NativeAotCompatibilityTest.cs` from PR #366 |
| OS-level affinity P/Invoke creates platform-specific bugs | Medium | One-off bugs per OS | Affinity is opt-in via `BlasOptions.PinThreads = true`, default off; reference impl is per-OS conditional |
| Determinism gate breaks during 2D-grid wire-up | Medium | Sub-issue B blocked | Sub-issue B's PR must add 2D-grid coverage to `DeterminismTests.cs` before merge; reviewer enforces |
| Routing shim adds measurable per-call overhead | Low | Small-shape regression | Autotune cache lookup is `O(1)` dictionary hit; benchmarked in Sub-issue F's PR; fast-path inlined |
| Self-hosted runner becomes unavailable | Medium | Perf bar can't be measured | Bar references `TargetHardwareFingerprint`; an equivalent runner spec is documented so a replacement can be provisioned |
| Caller migration via shim silently regresses an obscure code path | Low | Functional bug shipped | Existing test suite continues to call `BlasProvider.TryGemm`; if any test fails after Sub-issue F lands, that's the canary |
| The full set of 144 call sites grows during the sprint | Medium | Migration target moves | The shim handles it transparently — new callers just call `BlasProvider.TryGemm` and get auto-routed |

### The "we get stuck" escape hatch

If after a sub-issue lands and we genuinely cannot meet the bar on a class of shapes, we have three legal moves:

1. Narrow the bar (lower the win-rate constant in `PerfBar.cs`) — but only by explicit user approval in the mega-issue comments.
2. Add a `BlasManaged.UnsupportedShapes` allowlist that the bench excludes and that always routes to native — narrow, deliberate scope reduction.
3. Defer Sub-issue G entirely. Keep the routing shim (Sub-issue F) shipped; supply chain stays present but only as a fallback. Followup issue addresses the remaining gap.

Each escape hatch is an explicit, visible decision in the issue thread — not a silent redefinition after-the-fact.

### Out-of-band events that pause the sprint

- A security CVE in OpenBLAS or MKL with active exploit. → Sub-issue G gets fast-tracked even with partial bar.
- A net-new BLAS-using feature lands on `main`. → It's auto-handled by the shim, no sprint impact.
- A breaking change in `INumericOperations<T>` or `Vector<T>` runtime API. → Sub-issue D may need rework; tracked as a sub-issue blocker.

## 8 — Decisions captured (from brainstorming)

For traceability, the user-approved decisions that shaped this design:

| Decision | Choice |
|----------|--------|
| Scope shape | One mega-issue, all phases as sub-issues |
| Win bar | Expand bench first, then set bar against bigger set |
| Bench shape sources | Combine: instrument test suite + standard ML workload shapes |
| Phase order | All three optimization vectors in parallel (sub-issues, separate PRs) |
| Determinism | Bit-exact stays default; opt-in Fast mode |
| Migration strategy | Routing shim in BlasProvider itself (zero caller-side edits) |
| Code patterns in scope | Unsafe + pointer arithmetic; prefetch intrinsics; OS-level affinity P/Invoke; DynamicMethod IL emission |

## 9 — Next steps

1. **Spec self-review** (immediately, inline fixes).
2. **User reviews spec** — gate before writing-plans skill runs.
3. **Writing-plans skill** produces a phased implementation plan keyed off this spec.
4. **Mega tracking issue** created on `ooples/AiDotNet.Tensors` with this spec as the body, sub-issues linked.
5. **PR #366 stays draft** until at least Sub-issues A, F land and unblock its perf gate.
