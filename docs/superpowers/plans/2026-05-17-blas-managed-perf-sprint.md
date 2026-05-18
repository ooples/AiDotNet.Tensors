# BlasManaged Perf Sprint + Supply-Chain Removal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the perf gap between BlasManaged and OpenBLAS on a defensible expanded benchmark catalog, then remove all third-party BLAS dependencies (libopenblas + MKL) from the AiDotNet.Tensors library.

**Architecture:** Seven sub-issues run mostly in parallel after the bench catalog lands (Sub-issue A). Sub-issues B/C/D are orthogonal optimization vectors targeting different shape classes. Sub-issue E adopts an existing pre-pack cache at inference call sites. Sub-issue F adds a routing shim inside `BlasProvider` so the 144 caller sites don't have to change. Sub-issue G mechanically deletes the native P/Invoke once the perf bar is met. Detailed task lists for B–G are written AFTER A produces real bench numbers, since the win bar in `PerfBar.cs` depends on actual data.

**Tech Stack:** C# 13 / .NET 10 (and net471/net8.0 multi-targeting), `System.Runtime.Intrinsics`, BLIS-style packed GEMM (PR #366), xUnit test framework, GitHub Actions + self-hosted Ubuntu runner.

**Spec:** [`docs/superpowers/specs/2026-05-17-blas-managed-perf-sprint-design.md`](../specs/2026-05-17-blas-managed-perf-sprint-design.md)

**Predecessor PR:** [#366](https://github.com/ooples/AiDotNet.Tensors/pull/366) (stays draft until at least Sub-issues A, F land)

---

## File structure across all sub-issues

This map identifies every file the sprint creates or modifies. Each sub-issue owns a subset.

### New files (created during sprint)

| File | Owned by | Responsibility |
|------|----------|----------------|
| `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeCatalog.cs` | A | Static list of 50–80 (M,N,K,trans,dtype,freq,source) records |
| `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/PerfHarness.cs` | A | Median+p95 measurement runner; writes JSON to `artifacts/perf/` |
| `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeInstrumenter.cs` | A | Logs shapes seen during test suite to JSON dedupe file |
| `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/WorkloadShapes.cs` | A | Curated BERT/ResNet/GPT/MobileNet shape lists |
| `artifacts/perf/baseline.json` | A | First full-bench result; committed as the reference point |
| `artifacts/perf/instrumented-shapes.json` | A | Deduped shapes from running the test suite with instrumenter |
| `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBar.cs` | A | Win-rate / max-loss constants written by project owner |
| `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBarTest.cs` | A | Asserts bench JSON meets PerfBar constants |
| `src/AiDotNet.Tensors/Engines/BlasManaged/BlasMode.cs` | B | `BlasMode { Deterministic, Fast }` enum |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Dispatcher/PrefersManagedCache.cs` | F | Autotune cache keyed by `(shape, hardware)` |
| `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/RoutingShimTest.cs` | F | Verifies dispatch decision matches autotune |
| `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/NoNativeBlasTest.cs` | G | Asserts no `cblas_*` symbols / no DllImport "libopenblas" |

### Existing files modified

| File | Owned by | Change |
|------|----------|--------|
| `src/AiDotNet.Tensors/Engines/BlasManaged/BlasOptions.cs` | B | Add `BlasMode Mode { get; init; }` |
| `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs` | B | Add `static BlasMode DefaultMode` + `static bool PreferManaged` |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/PackAOnlyStrategy.cs` | B | Wire `NAxisDriver` for N-axis parallel |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs` | B | Wire `NAxisDriver` + `KAxisDriver` for tall-K shapes |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/PackBothStrategy.cs` | B | Wire `MN2DDriver` for 2D grid alternative |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Dispatcher/AxisSelector.cs` | B | Use autotune cache as backing store |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs` | C | Add pack-free path for `M·N·K < threshold` |
| `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs` | C | Dispatcher selects pack-free Streaming for small shapes |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Avx512/Avx512Fp32_16x16.cs` | D | Per-arch tile size variants + unsafe pointer hot loop |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Avx512/Avx512Fp64_8x16.cs` | D | Per-arch tile size variants + prefetch hints |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Avx2/Avx2Fp32_8x8.cs` | D | Zen3/Zen4 Mr=8,Nr=8 variant |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Avx2/Avx2Fp64_4x8.cs` | D | Zen3/Zen4 Mr=12,Nr=4 variant |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Neon/NeonFp32_8x4.cs` | D | Apple M-series Mr=8,Nr=12 variant |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Jit/JittedKernelCache.cs` | D | Emit shape-specialized K-loop-unrolled IL |
| `src/AiDotNet.Tensors/Engines/CpuEngine.cs` | E | Adopt `PrePackB` at inference MatMul call sites |
| `src/AiDotNet.Tensors/Engines/Compilation/CompiledTrainingPlan.cs` | E | Pre-pack weights at plan-compile time |
| `src/AiDotNet.Tensors/Engines/Autodiff/BackwardFunctions.cs` | E | Pre-pack constant weight tensors in backward |
| `src/AiDotNet.Tensors/Helpers/BlasProvider.cs` | F | Routing shim: prefer managed when autotune says so |
| `src/AiDotNet.Tensors/Helpers/BlasProvider.cs` | G | Delete DllImport declarations + MKL fallback |
| `src/AiDotNet.Tensors/AiDotNet.Tensors.csproj` | G | Delete `<PackageReference>` for OpenBLAS-native + MKL |
| `.github/workflows/*.yml` | G | Delete "install OpenBLAS" steps |

### Files explicitly NOT touched

| File | Why not |
|------|---------|
| Microkernel public interfaces (`IMicrokernel.cs`) | Stable contract from PR #366 |
| `BlasManagedStatsTracker.cs` | Already correct |
| `WeightPackHandle.cs` / `WeightPackCache.cs` | Already correct |
| `HardwareFingerprint.cs` | Already adequate |
| `DeterminismTests.cs` | Extended in B/C/D PRs only — no breaking changes |
| Existing tests in `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/` (Scalar/Stats/L1Perf/Regression/AOT) | Stay as-is; new tests added alongside |

---

## Sub-issue summaries (for GitHub issue bodies)

Each section below is the body of one sub-issue. Detailed task lists appear in this plan for Sub-issue A only; B–G get detailed task lists after A's bench data lands and `PerfBar.cs` constants are committed.

---

### Sub-issue A — Benchmark catalog & measurement infrastructure

**Status:** Detailed task list below. Executes first; gates everything else.

**Goal:** Produce a defensible 50–80 shape benchmark catalog, the harness to measure it, and the committed `PerfBar.cs` constants that the rest of the sprint asserts against.

**Acceptance:**
- `ShapeCatalog.cs` contains 50–80 unique `Shape` records sourced from (a) instrumentation of the existing test suite and (b) curated standard ML workloads.
- `PerfHarness.Run()` produces a deterministic JSON output capturing median + p95 per shape per backend (BlasManaged, OpenBLAS).
- `artifacts/perf/baseline.json` committed with results from the authoritative self-hosted runner.
- `PerfBar.cs` constants written based on baseline data (project-owner commit).
- `PerfBarTest.cs` skips on non-authoritative hosts and asserts win-rate ≥ `MinWinRatePercent` and max-loss ≤ `MaxLossMultiple`.

**Verification:** `dotnet test --filter PerfBarTest` passes on the self-hosted runner; baseline JSON is committed and visible in `artifacts/perf/`.

**Detailed task list:** see "Sub-issue A detailed tasks" below.

---

### Sub-issue B — Threading wire-up

**Goal:** Wire the existing `NAxisDriver`, `KAxisDriver`, and `MN2DDriver` primitives (built in PR #366 Phase G but only PackBoth uses M-axis) into `PackAOnlyStrategy` and `StreamingStrategy`. Add the `BlasMode` enum so determinism stays the default while allowing Fast mode at the kernel level.

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasMode.cs`
- Modify: `BlasOptions.cs`, `BlasManaged.cs`, `PackAOnlyStrategy.cs`, `StreamingStrategy.cs`, `PackBothStrategy.cs`, `AxisSelector.cs`
- Add tests: cross-thread-count determinism for all three strategies on Deterministic mode; perf test demonstrating 3× speedup vs M-axis-only on TallK and WideN shapes

**Acceptance:**
- `PackAOnlyStrategy` and `StreamingStrategy` use N-axis split when N ≥ threshold (configurable, autotuned).
- `StreamingStrategy` uses K-axis reduction-tree when K ≥ threshold AND M·N is small.
- `PackBothStrategy` uses 2D MN-grid when both M and N exceed thresholds.
- `DeterminismTests` extended to all three strategies × all three axes; Gate 3 still passes.
- On TallK_64×64×4096 shape, Sub-issue B alone achieves ≥3× speedup vs PR #366 baseline.

**Detailed tasks:** TBD — written after Sub-issue A lands (depends on which shapes the bench catalog flags as threading-bottlenecked).

---

### Sub-issue C — Small-shape pack-free streaming path

**Goal:** Eliminate the 30×–69× gap on Tiny_32sq / Tiny_64sq / WideFat_512×512×64 by adding a pack-free fast path that runs the microkernel directly against caller buffers when `M·N·K < threshold`.

**Files:**
- Modify: `StreamingStrategy.cs` (new code path: pack-free + masked-store tail)
- Modify: `BlasManaged.cs` (dispatcher selects pack-free path for small shapes)
- Add tests: correctness vs. PackBoth output on the small shapes; perf assertion that small shapes are ≤1.5× OpenBLAS

**Acceptance:**
- Small shapes (`M·N·K < threshold`) skip packing entirely.
- Existing `Avx2Tail` / `Avx512Tail` kernels handle the M/N tails.
- Tiny_32sq, Tiny_64sq, WideFat_512×512×64 ≤ 1.5× OpenBLAS (currently 30×, 30×, 69×).
- No regression on existing tests; allocator stats show zero pack allocations for selected shapes.

**Detailed tasks:** TBD — written after Sub-issue A lands.

---

### Sub-issue D — Per-arch microkernel tuning + JIT IL emission

**Goal:** Match OpenBLAS hand-tuning on the top-10 most-frequent shapes by introducing arch-specific tile widths, software prefetch, unsafe pointer hot loops, and shape-specialized IL-emitted microkernels.

**Files:**
- Modify: all six microkernel files under `Microkernels/Avx512/`, `Microkernels/Avx2/`, `Microkernels/Neon/`
- Modify: `JittedKernelCache.cs` (activate IL emission for hot shapes)
- Add tests: per-arch tile size verification; IL-emitted kernel correctness vs generic kernel; NativeAOT compatibility (already covered by existing test)

**Acceptance:**
- Each microkernel file exposes arch-specific tile-size variants gated by `HardwareFingerprint`.
- Software prefetch (`Sse.Prefetch0/1/NTA`) added inside packing and microkernel inner loops.
- Hot inner loops converted to `unsafe` + raw pointer arithmetic; benchmarked vs Span<T> version.
- `JittedKernelCache` emits shape-specialized kernels for top-10 frequency shapes; falls back to generic kernel on NativeAOT.
- On top-10 catalog shapes, BlasManaged median ≤ OpenBLAS median.

**Detailed tasks:** TBD — written after Sub-issue A lands (depends on which arches the runner exposes and which shapes are "top-10 by frequency").

---

### Sub-issue E — Pre-pack weight cache adoption

**Goal:** Activate the existing `WeightPackCache` (allocator layer 3 from PR #366) by adopting `PrePackB` at inference call sites where B is invariant across calls.

**Files:**
- Modify: `CpuEngine.cs` (MatMul/MatMulTransposed adopt PrePackB)
- Modify: `CompiledTrainingPlan.cs` (pre-pack weights at plan-compile time)
- Modify: `BackwardFunctions.cs` (pre-pack constant-weight backward paths)
- Add tests: cache-hit assertion via `BlasManagedStatsTracker.PrePackHits`; perf comparison showing ≥1.5× speedup on repeated-shape inference

**Acceptance:**
- `BlasManaged.PrePackB` called at plan-compile time / model-load time at the three target call sites.
- Stats counter `WeightPackCacheHits` increments on repeat calls.
- FFN_128×768×768 batched inference shows ≥1.5× speedup vs uncached path.
- No correctness regression on existing tests.

**Detailed tasks:** TBD — written after Sub-issue A lands.

---

### Sub-issue F — BlasProvider routing shim + autotune dispatch

**Goal:** Land the routing shim inside `BlasProvider.TryGemm` / `TryGemmEx` so callers transparently get the better of {BlasManaged, native}. Zero caller edits required.

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Dispatcher/PrefersManagedCache.cs`
- Modify: `src/AiDotNet.Tensors/Helpers/BlasProvider.cs` (add routing shim at top of TryGemm/TryGemmEx)
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs` (expose `static bool PreferManaged`)
- Add tests: `RoutingShimTest.cs` verifies dispatch decision matches autotune choice; existing test suite continues to pass without changes (canary)

**Acceptance:**
- `BlasProvider.TryGemm` queries `PrefersManagedCache` before falling through to native.
- First call per (shape, hardware) measures both paths and caches the winner.
- `BlasOptions.PreferManaged = true` forces all calls through BlasManaged regardless of autotune.
- `PerfBarTest` passes with `PreferManaged = true`.
- All existing tests pass — the shim is invisible to callers.

**Detailed tasks:** TBD — written after Sub-issue A lands.

---

### Sub-issue G — Native BLAS removal

**Goal:** Delete every DllImport against `libopenblas` and `MKL`, delete the `<PackageReference>` to native BLAS NuGets, delete CI install steps for native BLAS. Verify the library builds and tests pass on a host with no native BLAS installed.

**Files:**
- Modify (heavy deletes): `src/AiDotNet.Tensors/Helpers/BlasProvider.cs`
- Modify: `src/AiDotNet.Tensors/AiDotNet.Tensors.csproj` (remove native BLAS package refs)
- Modify: `.github/workflows/*.yml` (remove "install OpenBLAS" steps)
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/NoNativeBlasTest.cs` (asserts no `cblas_` strings remain; verifies LoadLibrary is not called for libopenblas/MKL)

**Acceptance:**
- `grep -ri "cblas_\|openblas\|mkl" src/ --include="*.cs"` returns zero matches.
- `dotnet list package --include-transitive` shows no native BLAS package.
- Full test suite passes on a freshly-provisioned Linux/Windows/macOS host with no MKL/OpenBLAS DLLs/.so/.dylib present.
- PR #366's L3 no-regression test continues to pass.

**Detailed tasks:** TBD — written after Sub-issue F lands.

---

## Sub-issue A detailed tasks

This is the only sub-issue with full task detail in this plan. B–G get detailed task lists in follow-up plans after A's bench data lands.

### Task A.1: Create the empty ShapeCatalog scaffold

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeCatalog.cs`

- [ ] **Step 1: Write the failing test**

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/ShapeCatalogTest.cs`:

```csharp
using AiDotNet.Tensors.Benchmarks.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class ShapeCatalogTest
{
    [Fact]
    public void Catalog_Has_Between_50_And_80_Shapes()
    {
        var shapes = ShapeCatalog.All;
        Assert.InRange(shapes.Count, 50, 80);
    }

    [Fact]
    public void All_Shapes_Have_Positive_Dimensions()
    {
        foreach (var s in ShapeCatalog.All)
        {
            Assert.True(s.M > 0, $"{s.Name}: M={s.M}");
            Assert.True(s.N > 0, $"{s.Name}: N={s.N}");
            Assert.True(s.K > 0, $"{s.Name}: K={s.K}");
        }
    }

    [Fact]
    public void All_Shape_Names_Are_Unique()
    {
        var names = ShapeCatalog.All.Select(s => s.Name).ToList();
        Assert.Equal(names.Count, names.Distinct().Count());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "ShapeCatalogTest" --logger "console;verbosity=normal"`
Expected: FAIL — `ShapeCatalog` type doesn't exist.

- [ ] **Step 3: Create the minimal catalog**

Create `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeCatalog.cs`:

```csharp
using System.Collections.Generic;

namespace AiDotNet.Tensors.Benchmarks.BlasManaged;

public enum DType { Single, Double }

public record Shape(
    string Name,
    int M, int N, int K,
    bool TransA, bool TransB,
    DType Dtype,
    int Frequency,
    string Source);

public static class ShapeCatalog
{
    // Initial scaffold — populated by Tasks A.2 (instrumentation) and A.3 (workload shapes).
    // Until those land, return an empty list — the count assertion will fail, but that's
    // the intentional gate that forces those tasks to land before downstream sub-issues.
    public static IReadOnlyList<Shape> All { get; } = new List<Shape>();
}
```

- [ ] **Step 4: Run test to verify the type exists (catalog will be empty, count assertion fails)**

Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "ShapeCatalogTest.All_Shape_Names_Are_Unique"`
Expected: PASS (empty list has unique names trivially).
Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "ShapeCatalogTest.Catalog_Has_Between_50_And_80_Shapes"`
Expected: FAIL — 0 not in range 50–80. This is the gate task A.2 + A.3 close.

- [ ] **Step 5: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeCatalog.cs tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/ShapeCatalogTest.cs
git commit -m "feat(#358-A): ShapeCatalog scaffold + count/uniqueness tests

Catalog starts empty; count assertion fails intentionally until
instrumentation (A.2) and workload shapes (A.3) populate it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A.2: ShapeInstrumenter — log shapes seen during test suite

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeInstrumenter.cs`
- Modify: `src/AiDotNet.Tensors/Helpers/BlasProvider.cs` (add an optional shape-logging hook)

- [ ] **Step 1: Write the failing test**

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/ShapeInstrumenterTest.cs`:

```csharp
using AiDotNet.Tensors.Benchmarks.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class ShapeInstrumenterTest
{
    [Fact]
    public void Instrumenter_Captures_Shape_When_Enabled()
    {
        ShapeInstrumenter.Reset();
        ShapeInstrumenter.Enabled = true;
        try
        {
            // Drive a single BLAS call.
            float[] a = new float[64 * 32];
            float[] b = new float[32 * 16];
            float[] c = new float[64 * 16];
            BlasProvider.TryGemm(m: 64, n: 16, k: 32,
                                 a: a, lda: 32, transA: false,
                                 b: b, ldb: 16, transB: false,
                                 c: c, ldc: 16);

            var shapes = ShapeInstrumenter.Snapshot();
            Assert.Contains(shapes, s => s.M == 64 && s.N == 16 && s.K == 32 && s.Dtype == DType.Single);
        }
        finally
        {
            ShapeInstrumenter.Enabled = false;
            ShapeInstrumenter.Reset();
        }
    }

    [Fact]
    public void Instrumenter_Deduplicates_Identical_Shapes()
    {
        ShapeInstrumenter.Reset();
        ShapeInstrumenter.Enabled = true;
        try
        {
            float[] a = new float[64 * 32];
            float[] b = new float[32 * 16];
            float[] c = new float[64 * 16];
            for (int i = 0; i < 3; i++)
                BlasProvider.TryGemm(m: 64, n: 16, k: 32,
                                     a: a, lda: 32, transA: false,
                                     b: b, ldb: 16, transB: false,
                                     c: c, ldc: 16);

            var shapes = ShapeInstrumenter.Snapshot();
            var match = shapes.Single(s => s.M == 64 && s.N == 16 && s.K == 32);
            Assert.Equal(3, match.Frequency);
        }
        finally
        {
            ShapeInstrumenter.Enabled = false;
            ShapeInstrumenter.Reset();
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "ShapeInstrumenterTest"`
Expected: FAIL — `ShapeInstrumenter` doesn't exist.

- [ ] **Step 3: Implement ShapeInstrumenter**

Create `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeInstrumenter.cs`:

```csharp
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace AiDotNet.Tensors.Benchmarks.BlasManaged;

public static class ShapeInstrumenter
{
    public static bool Enabled { get; set; } = false;

    private static readonly ConcurrentDictionary<ShapeKey, int> _counts = new();

    private record struct ShapeKey(int M, int N, int K, bool TransA, bool TransB, DType Dtype);

    public static void Record(int m, int n, int k, bool transA, bool transB, DType dtype)
    {
        if (!Enabled) return;
        var key = new ShapeKey(m, n, k, transA, transB, dtype);
        _counts.AddOrUpdate(key, 1, (_, count) => count + 1);
    }

    public static IReadOnlyList<Shape> Snapshot()
    {
        return _counts
            .Select(kv => new Shape(
                Name: $"Instrumented_{kv.Key.M}x{kv.Key.N}x{kv.Key.K}_{(kv.Key.TransA ? "TA" : "NA")}_{(kv.Key.TransB ? "TB" : "NB")}_{kv.Key.Dtype}",
                M: kv.Key.M, N: kv.Key.N, K: kv.Key.K,
                TransA: kv.Key.TransA, TransB: kv.Key.TransB,
                Dtype: kv.Key.Dtype,
                Frequency: kv.Value,
                Source: "instrumented:test-suite"))
            .OrderByDescending(s => s.Frequency)
            .ToList();
    }

    public static void Reset() => _counts.Clear();

    public static void DumpToJson(string path)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(Snapshot(), new JsonSerializerOptions { WriteIndented = true }));
    }
}
```

- [ ] **Step 4: Hook into BlasProvider**

Open `src/AiDotNet.Tensors/Helpers/BlasProvider.cs`. At the top of each `TryGemm` / `TryGemmEx` overload, add (right after entry):

```csharp
// Shape instrumentation hook — no-op when disabled, zero-cost in hot path
// because the static field check inlines and ShapeInstrumenter is in the
// test/benchmark assembly. The internal hook is invoked via reflection
// at test-run time so the production code has no reference to it.
ShapeLogHook?.Invoke(m, n, k, transA, transB, /*dtype*/ typeof(float));
```

Add to BlasProvider class:

```csharp
/// <summary>
/// Test-only hook for shape instrumentation. Production code never sets this.
/// </summary>
internal static Action<int, int, int, bool, bool, Type>? ShapeLogHook;
```

In `ShapeInstrumenterTest.cs` setup, wire the hook:

```csharp
static ShapeInstrumenterTest()
{
    BlasProvider.ShapeLogHook = (m, n, k, ta, tb, t) =>
        ShapeInstrumenter.Record(m, n, k, ta, tb, t == typeof(float) ? DType.Single : DType.Double);
}
```

- [ ] **Step 5: Run test to verify pass**

Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "ShapeInstrumenterTest"`
Expected: PASS (2/2).

- [ ] **Step 6: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeInstrumenter.cs \
        tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/ShapeInstrumenterTest.cs \
        src/AiDotNet.Tensors/Helpers/BlasProvider.cs
git commit -m "feat(#358-A): ShapeInstrumenter — record GEMM shapes hit during tests

Internal hook in BlasProvider invoked only when ShapeLogHook is set
(test-only). Zero overhead in production. Deduplicates by shape key,
records frequency. Snapshot returns DESC by frequency for top-N selection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A.3: WorkloadShapes — curated standard ML shapes

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/WorkloadShapes.cs`

- [ ] **Step 1: Write the failing test**

Add to `ShapeCatalogTest.cs`:

```csharp
[Fact]
public void WorkloadShapes_Cover_Bert_Resnet_Gpt_MobileNet()
{
    var w = WorkloadShapes.All;
    Assert.Contains(w, s => s.Source.Contains("BERT"));
    Assert.Contains(w, s => s.Source.Contains("ResNet"));
    Assert.Contains(w, s => s.Source.Contains("GPT"));
    Assert.Contains(w, s => s.Source.Contains("MobileNet"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "WorkloadShapes_Cover_Bert_Resnet_Gpt_MobileNet"`
Expected: FAIL — `WorkloadShapes` doesn't exist.

- [ ] **Step 3: Implement WorkloadShapes**

Create `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/WorkloadShapes.cs`:

```csharp
using System.Collections.Generic;

namespace AiDotNet.Tensors.Benchmarks.BlasManaged;

public static class WorkloadShapes
{
    public static IReadOnlyList<Shape> All { get; } = new[]
    {
        // BERT-base (hidden=768, intermediate=3072, heads=12, head_dim=64, seq=128, batch=8)
        new Shape("BERT_FFN_up_1024x3072x768",     1024, 3072, 768, false, false, DType.Single, 0, "workload:BERT-base FFN expansion"),
        new Shape("BERT_FFN_down_1024x768x3072",   1024, 768, 3072, false, false, DType.Single, 0, "workload:BERT-base FFN contraction"),
        new Shape("BERT_Attn_QKV_1024x768x768",    1024, 768, 768,  false, false, DType.Single, 0, "workload:BERT-base attention QKV proj"),
        new Shape("BERT_Attn_score_96x128x64",     96,   128, 64,   false, false, DType.Single, 0, "workload:BERT-base attention score"),
        new Shape("BERT_Attn_ctx_96x64x128",       96,   64,  128,  false, false, DType.Single, 0, "workload:BERT-base attention context"),

        // GPT-2 medium (hidden=1024, intermediate=4096)
        new Shape("GPT2med_FFN_up_512x4096x1024",  512,  4096, 1024, false, false, DType.Single, 0, "workload:GPT-2 medium FFN up"),
        new Shape("GPT2med_FFN_down_512x1024x4096",512,  1024, 4096, false, false, DType.Single, 0, "workload:GPT-2 medium FFN down"),
        new Shape("GPT2med_Attn_proj_512x1024x1024",512, 1024, 1024, false, false, DType.Single, 0, "workload:GPT-2 medium attention proj"),

        // ResNet50 (im2col-ed conv shapes, FP32 forward)
        new Shape("ResNet50_conv1_3136x64x147",    3136, 64,  147,  false, false, DType.Single, 0, "workload:ResNet50 conv1 7x7"),
        new Shape("ResNet50_layer1_3136x64x64",    3136, 64,  64,   false, false, DType.Single, 0, "workload:ResNet50 layer1 1x1"),
        new Shape("ResNet50_layer2_784x128x128",   784,  128, 128,  false, false, DType.Single, 0, "workload:ResNet50 layer2 3x3"),
        new Shape("ResNet50_layer3_196x256x256",   196,  256, 256,  false, false, DType.Single, 0, "workload:ResNet50 layer3 3x3"),
        new Shape("ResNet50_layer4_49x512x512",    49,   512, 512,  false, false, DType.Single, 0, "workload:ResNet50 layer4 3x3"),
        new Shape("ResNet50_fc_1x1000x2048",       1,    1000,2048, false, false, DType.Single, 0, "workload:ResNet50 FC head"),

        // MobileNetV2 (depthwise + pointwise; pointwise is GEMM)
        new Shape("MobileNetV2_pw_3136x32x32",     3136, 32,  32,   false, false, DType.Single, 0, "workload:MobileNetV2 PW 1x1"),
        new Shape("MobileNetV2_pw_784x144x24",     784,  144, 24,   false, false, DType.Single, 0, "workload:MobileNetV2 PW expand"),
        new Shape("MobileNetV2_pw_196x96x96",      196,  96,  96,   false, false, DType.Single, 0, "workload:MobileNetV2 PW mid"),
        new Shape("MobileNetV2_fc_1x1000x1280",    1,    1000,1280, false, false, DType.Single, 0, "workload:MobileNetV2 FC head"),

        // Mixed precision (FP64) — scientific workloads
        new Shape("FP64_Linreg_4096x1024x1024",    4096, 1024,1024, false, false, DType.Double, 0, "workload:linear regression"),
        new Shape("FP64_PCA_512x512x4096",         512,  512, 4096, false, false, DType.Double, 0, "workload:PCA covariance"),
        new Shape("FP64_QR_2048x2048x256",         2048, 2048,256,  false, false, DType.Double, 0, "workload:QR decomposition panel"),

        // Backward-pass shapes (transposed)
        new Shape("BERT_FFN_bwd_dW_3072x768x1024", 3072, 768, 1024, true,  false, DType.Single, 0, "workload:BERT FFN backward dW"),
        new Shape("ResNet50_bwd_dW_64x147x3136",   64,   147, 3136, true,  false, DType.Single, 0, "workload:ResNet50 conv1 backward"),

        // Tiny shapes (LSTM cells, embedding lookups, small batch)
        new Shape("LSTM_cell_1x256x256",           1,    256, 256,  false, false, DType.Single, 0, "workload:LSTM cell per-timestep"),
        new Shape("Embedding_proj_8x768x768",      8,    768, 768,  false, false, DType.Single, 0, "workload:Embedding projection"),
    };
}
```

- [ ] **Step 4: Run test to verify pass**

Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "WorkloadShapes_Cover_Bert_Resnet_Gpt_MobileNet"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/BlasManaged/WorkloadShapes.cs \
        tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/ShapeCatalogTest.cs
git commit -m "feat(#358-A): WorkloadShapes — BERT/GPT-2/ResNet50/MobileNetV2

25 curated shapes covering FFN expand/contract, attention QKV/score/context,
im2col-ed conv forward/backward, FC heads, and FP64 scientific workloads.
Frequency starts at 0 — instrumentation data (A.2) provides the frequency
weight; workload shapes are kept regardless of frequency to force coverage.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A.4: Merge instrumented + workload into ShapeCatalog.All

**Files:**
- Modify: `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeCatalog.cs`
- Create: `artifacts/perf/instrumented-shapes.json` (committed by a one-time test-run)

- [ ] **Step 1: Run the test suite with instrumenter ON to collect shapes**

Add to `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/Program.cs` (or create a small console runner) the following:

```csharp
// One-shot shape harvester. Invoke once at the local dev machine; commit the JSON.
public static void HarvestShapes()
{
    BlasProvider.ShapeLogHook = (m, n, k, ta, tb, t) =>
        ShapeInstrumenter.Record(m, n, k, ta, tb, t == typeof(float) ? DType.Single : DType.Double);
    ShapeInstrumenter.Enabled = true;

    // Run the full test suite programmatically — invoke each test assembly.
    // Or simpler: ask the user to run `dotnet test` with an env-var that flips Enabled on.
}
```

Instead of programmatic invocation, simpler: gate `ShapeInstrumenter.Enabled` on env var `AIDOTNET_INSTRUMENT_SHAPES=1`. Then:

```bash
AIDOTNET_INSTRUMENT_SHAPES=1 dotnet test --no-restore
```

During the test run, `ShapeInstrumenter` records every shape. After the test run, a separate fixture's `Dispose` (or `[AssemblyFixture]`) calls `ShapeInstrumenter.DumpToJson("artifacts/perf/instrumented-shapes.json")`.

Implementation:

Modify `ShapeInstrumenter.cs`:

```csharp
public static bool Enabled { get; set; } =
    Environment.GetEnvironmentVariable("AIDOTNET_INSTRUMENT_SHAPES") == "1";
```

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/InstrumenterAssemblyFixture.cs`:

```csharp
using System;
using AiDotNet.Tensors.Benchmarks.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[CollectionDefinition("ShapeInstrumenter")]
public class ShapeInstrumenterCollection : ICollectionFixture<ShapeInstrumenterFixture> { }

public class ShapeInstrumenterFixture : IDisposable
{
    public ShapeInstrumenterFixture()
    {
        if (ShapeInstrumenter.Enabled)
        {
            BlasProvider.ShapeLogHook = (m, n, k, ta, tb, t) =>
                ShapeInstrumenter.Record(m, n, k, ta, tb, t == typeof(float) ? DType.Single : DType.Double);
        }
    }

    public void Dispose()
    {
        if (ShapeInstrumenter.Enabled)
        {
            var outPath = Environment.GetEnvironmentVariable("AIDOTNET_INSTRUMENT_OUT")
                ?? "artifacts/perf/instrumented-shapes.json";
            ShapeInstrumenter.DumpToJson(outPath);
        }
    }
}
```

- [ ] **Step 2: Harvest the shapes**

Run from repo root:

```bash
mkdir -p artifacts/perf
AIDOTNET_INSTRUMENT_SHAPES=1 \
AIDOTNET_INSTRUMENT_OUT=$PWD/artifacts/perf/instrumented-shapes.json \
dotnet test tests/AiDotNet.Tensors.Tests --no-restore --logger "console;verbosity=minimal"
```

Expected output: `artifacts/perf/instrumented-shapes.json` contains a JSON array of `Shape` records with non-zero `Frequency`.

- [ ] **Step 3: Commit the harvested JSON**

```bash
git add artifacts/perf/instrumented-shapes.json
git commit -m "data(#358-A): harvested shapes from test-suite instrumentation

Run via:
  AIDOTNET_INSTRUMENT_SHAPES=1 dotnet test tests/AiDotNet.Tensors.Tests

Frequencies reflect how often each shape was hit during a full test
run. ShapeCatalog.All loads this JSON + WorkloadShapes.All to produce
the final benchmark catalog.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Update ShapeCatalog.All to merge sources**

Replace the body of `ShapeCatalog.cs`:

```csharp
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace AiDotNet.Tensors.Benchmarks.BlasManaged;

public enum DType { Single, Double }

public record Shape(
    string Name,
    int M, int N, int K,
    bool TransA, bool TransB,
    DType Dtype,
    int Frequency,
    string Source);

public static class ShapeCatalog
{
    public static IReadOnlyList<Shape> All { get; } = LoadCatalog();

    private static IReadOnlyList<Shape> LoadCatalog()
    {
        var result = new List<Shape>();

        // 1. Always include workload shapes (forces coverage of BERT/ResNet/GPT/MobileNet).
        result.AddRange(WorkloadShapes.All);

        // 2. Add top-N instrumented shapes by frequency. Cap so we stay in 50-80 range.
        var instrumentedPath = FindInstrumentedJson();
        if (instrumentedPath != null)
        {
            var json = File.ReadAllText(instrumentedPath);
            var instrumented = JsonSerializer.Deserialize<List<Shape>>(json) ?? new List<Shape>();
            int slots = 80 - result.Count;
            result.AddRange(instrumented.Take(slots));
        }

        // 3. Dedupe by (M,N,K,TransA,TransB,Dtype). When both workload and instrumented
        //    contain the same shape, keep the workload version (richer Source).
        var deduped = result
            .GroupBy(s => (s.M, s.N, s.K, s.TransA, s.TransB, s.Dtype))
            .Select(g => g.OrderBy(s => s.Source.StartsWith("workload:") ? 0 : 1).First())
            .ToList();

        return deduped;
    }

    private static string? FindInstrumentedJson()
    {
        var candidates = new[]
        {
            "artifacts/perf/instrumented-shapes.json",
            "../../artifacts/perf/instrumented-shapes.json",
            "../../../artifacts/perf/instrumented-shapes.json",
            "../../../../artifacts/perf/instrumented-shapes.json",
        };
        return candidates.FirstOrDefault(File.Exists);
    }
}
```

- [ ] **Step 5: Run ShapeCatalogTest to confirm all tests pass**

Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "ShapeCatalogTest"`
Expected: PASS (all 4 tests including `Catalog_Has_Between_50_And_80_Shapes`).

- [ ] **Step 6: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/BlasManaged/ShapeCatalog.cs
git commit -m "feat(#358-A): ShapeCatalog merges WorkloadShapes + instrumented JSON

ShapeCatalog.All produces deduped Shape[] in 50-80 range. Workload
shapes always included (force coverage of BERT/ResNet/GPT/MobileNet).
Top-N instrumented shapes by frequency fill remaining slots.

When same (M,N,K,TransA,TransB,Dtype) appears in both sources, the
workload version wins (richer Source field).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A.5: PerfHarness — median+p95 measurement runner

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/PerfHarness.cs`

- [ ] **Step 1: Write the failing test**

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfHarnessTest.cs`:

```csharp
using System.IO;
using AiDotNet.Tensors.Benchmarks.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class PerfHarnessTest
{
    [Fact]
    public void Run_Single_Shape_Produces_Result()
    {
        var shape = new Shape("Test_64x64x64", 64, 64, 64, false, false, DType.Single, 0, "test");
        var result = PerfHarness.RunShape(shape);
        Assert.Equal(shape.Name, result.ShapeName);
        Assert.True(result.BlasManagedMedianMs > 0);
        // Native may be unavailable on some hosts; harness must tolerate.
        Assert.True(result.NativeMedianMs >= 0);
        Assert.True(result.BlasManagedP95Ms >= result.BlasManagedMedianMs);
    }

    [Fact]
    public void Run_Writes_Json_With_All_Shapes()
    {
        var tmpPath = Path.Combine(Path.GetTempPath(), $"perfharness-{Guid.NewGuid():N}.json");
        try
        {
            var shapes = new[]
            {
                new Shape("Tiny_32x32x32", 32, 32, 32, false, false, DType.Single, 0, "test"),
                new Shape("Tiny_64x64x64", 64, 64, 64, false, false, DType.Single, 0, "test"),
            };
            PerfHarness.RunAll(shapes, tmpPath);
            Assert.True(File.Exists(tmpPath));
            var json = File.ReadAllText(tmpPath);
            Assert.Contains("Tiny_32x32x32", json);
            Assert.Contains("Tiny_64x64x64", json);
            Assert.Contains("HardwareFingerprint", json);
            Assert.Contains("GitSha", json);
        }
        finally
        {
            if (File.Exists(tmpPath)) File.Delete(tmpPath);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "PerfHarnessTest"`
Expected: FAIL — `PerfHarness` doesn't exist.

- [ ] **Step 3: Implement PerfHarness**

Create `tests/AiDotNet.Tensors.Benchmarks/BlasManaged/PerfHarness.cs`:

```csharp
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Helpers.Autotune;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.BlasManaged;

public record ShapeResult(
    string ShapeName,
    int M, int N, int K,
    bool TransA, bool TransB,
    string Dtype,
    double BlasManagedMedianMs,
    double BlasManagedP95Ms,
    double NativeMedianMs,
    double NativeP95Ms,
    bool NativeAvailable,
    double RatioBmOverNative);

public record HarnessOutput(
    string GitSha,
    string HardwareFingerprint,
    string TimestampUtc,
    IReadOnlyList<ShapeResult> Shapes);

public static class PerfHarness
{
    public static ShapeResult RunShape(Shape s)
    {
        long workEst = (long)s.M * s.N * s.K;
        int iters = workEst > 100_000_000L ? 10 : workEst > 10_000_000L ? 30 : 50;
        const int Warmup = 3;

        if (s.Dtype == DType.Single)
        {
            return MeasureSingle(s, iters, Warmup);
        }
        else
        {
            return MeasureDouble(s, iters, Warmup);
        }
    }

    public static void RunAll(IEnumerable<Shape> shapes, string outputPath)
    {
        var results = shapes.Select(RunShape).ToList();
        var output = new HarnessOutput(
            GitSha: GetGitSha(),
            HardwareFingerprint: HardwareFingerprint.Current.ToString(),
            TimestampUtc: DateTime.UtcNow.ToString("u"),
            Shapes: results);

        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(outputPath, JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static ShapeResult MeasureSingle(Shape s, int iters, int warmup)
    {
        int aRows = s.TransA ? s.K : s.M;
        int aCols = s.TransA ? s.M : s.K;
        int bRows = s.TransB ? s.N : s.K;
        int bCols = s.TransB ? s.K : s.N;
        var rng = new Random(42);
        var a = new float[aRows * aCols];
        var b = new float[bRows * bCols];
        var c = new float[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        for (int i = 0; i < warmup; i++)
            BlasManagedLib.Gemm<float>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);

        var bmTimes = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            BlasManagedLib.Gemm<float>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);
            sw.Stop();
            bmTimes[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(bmTimes);
        double bmMedian = bmTimes[iters / 2];
        double bmP95 = bmTimes[Math.Min(iters - 1, (int)(iters * 0.95))];

        double nativeMedian = -1, nativeP95 = -1;
        bool nativeOk = BlasProvider.IsAvailable;
        if (nativeOk)
        {
            try
            {
                for (int i = 0; i < warmup; i++)
                    BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);

                var nTimes = new double[iters];
                for (int i = 0; i < iters; i++)
                {
                    sw.Restart();
                    BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);
                    sw.Stop();
                    nTimes[i] = sw.Elapsed.TotalMilliseconds;
                }
                Array.Sort(nTimes);
                nativeMedian = nTimes[iters / 2];
                nativeP95 = nTimes[Math.Min(iters - 1, (int)(iters * 0.95))];
            }
            catch
            {
                nativeOk = false;
            }
        }

        double ratio = nativeOk && nativeMedian > 0 ? bmMedian / nativeMedian : 0.0;

        return new ShapeResult(
            s.Name, s.M, s.N, s.K, s.TransA, s.TransB, s.Dtype.ToString(),
            bmMedian, bmP95, nativeMedian, nativeP95, nativeOk, ratio);
    }

    private static ShapeResult MeasureDouble(Shape s, int iters, int warmup)
    {
        // Same as MeasureSingle but with double[] — body inlined for clarity.
        int aRows = s.TransA ? s.K : s.M;
        int aCols = s.TransA ? s.M : s.K;
        int bRows = s.TransB ? s.N : s.K;
        int bCols = s.TransB ? s.K : s.N;
        var rng = new Random(42);
        var a = new double[aRows * aCols];
        var b = new double[bRows * bCols];
        var c = new double[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        for (int i = 0; i < warmup; i++)
            BlasManagedLib.Gemm<double>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);

        var bmTimes = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            BlasManagedLib.Gemm<double>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);
            sw.Stop();
            bmTimes[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(bmTimes);
        double bmMedian = bmTimes[iters / 2];
        double bmP95 = bmTimes[Math.Min(iters - 1, (int)(iters * 0.95))];

        double nativeMedian = -1, nativeP95 = -1;
        bool nativeOk = BlasProvider.IsAvailable;
        if (nativeOk)
        {
            try
            {
                for (int i = 0; i < warmup; i++)
                    BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);

                var nTimes = new double[iters];
                for (int i = 0; i < iters; i++)
                {
                    sw.Restart();
                    BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);
                    sw.Stop();
                    nTimes[i] = sw.Elapsed.TotalMilliseconds;
                }
                Array.Sort(nTimes);
                nativeMedian = nTimes[iters / 2];
                nativeP95 = nTimes[Math.Min(iters - 1, (int)(iters * 0.95))];
            }
            catch
            {
                nativeOk = false;
            }
        }

        double ratio = nativeOk && nativeMedian > 0 ? bmMedian / nativeMedian : 0.0;

        return new ShapeResult(
            s.Name, s.M, s.N, s.K, s.TransA, s.TransB, s.Dtype.ToString(),
            bmMedian, bmP95, nativeMedian, nativeP95, nativeOk, ratio);
    }

    private static string GetGitSha()
    {
        try
        {
            var psi = new ProcessStartInfo("git", "rev-parse HEAD")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var p = Process.Start(psi);
            if (p == null) return "unknown";
            string sha = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(2000);
            return sha;
        }
        catch
        {
            return "unknown";
        }
    }
}
```

- [ ] **Step 4: Run test to verify pass**

Run: `dotnet test tests/AiDotNet.Tensors.Tests --filter "PerfHarnessTest"`
Expected: PASS (2/2).

- [ ] **Step 5: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/BlasManaged/PerfHarness.cs \
        tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfHarnessTest.cs
git commit -m "feat(#358-A): PerfHarness — median+p95 GEMM measurement runner

RunShape(Shape): single-shape measurement, FP32 or FP64. Returns
BlasManaged and native medians + p95s + ratio. Tolerates native
unavailable (sets NativeAvailable=false, ratio=0).

RunAll(shapes, outPath): full catalog sweep, writes JSON with
GitSha + HardwareFingerprint + timestamp envelope.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A.6: Produce baseline.json on the authoritative runner

**Files:**
- Create: `artifacts/perf/baseline.json` (committed)

- [ ] **Step 1: Run full catalog sweep on self-hosted runner**

This step is run **by the project owner** on the self-hosted runner via MCP console automation (per CLAUDE.md "use MCP console automation, not direct SSH").

Connect via the saved profile, then run from the runner's checkout of the branch:

```bash
mkdir -p artifacts/perf
AIDOTNET_PERF_RUNNER=1 \
dotnet run --project tests/AiDotNet.Tensors.Benchmarks --configuration Release -- \
  --bench-perf-baseline \
  --output artifacts/perf/baseline.json
```

(Add a `--bench-perf-baseline` switch to `tests/AiDotNet.Tensors.Benchmarks/Program.cs` that loads `ShapeCatalog.All` and calls `PerfHarness.RunAll(...)`.)

Expected: `artifacts/perf/baseline.json` produced, ~64 shape entries, hardware fingerprint captured.

- [ ] **Step 2: Inspect ratios to decide PerfBar.cs constants**

Open `artifacts/perf/baseline.json`. Compute:
- Win count (ratio < 1.0)
- Tied count (1.0 ≤ ratio < 1.2)
- Loss count (ratio ≥ 1.2)
- Worst-loss multiple

These numbers are the **starting point**, not the bar. The bar is what we commit to *after* the optimization work. The user decides:
- `MinWinRatePercent` — target win rate after sprint complete (e.g., 80).
- `MaxLossMultiple` — ceiling for the worst remaining loss (e.g., 1.20).

- [ ] **Step 3: Commit baseline.json**

```bash
git add artifacts/perf/baseline.json
git commit -m "data(#358-A): baseline.json — pre-sprint BlasManaged vs OpenBLAS

Hardware: <captured from runner fingerprint>
Catalog size: <N> shapes
Pre-sprint state: <win count> / <N> wins, worst loss <X>x

This file is the reference point against which Sub-issues B-F's
progress is measured. PerfBar.cs constants are set in A.7 based on
inspecting this data.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A.7: Commit PerfBar.cs constants + PerfBarTest gate

**Files:**
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBar.cs`
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBarTest.cs`

- [ ] **Step 1: Project owner decides bar values**

After inspecting `baseline.json`, the project owner picks:
- `MinWinRatePercent` — final win-rate target. Realistic: 80% if pre-sprint shows ~0% wins. Aggressive: 90%.
- `MaxLossMultiple` — ceiling for worst remaining loss. Realistic: 1.20 (within noise floor). Aggressive: 1.10.
- `CatalogShapeCount` — exact size of the catalog (must match `ShapeCatalog.All.Count`).
- `TargetHardwareFingerprint` — captured from the runner's first baseline.json.

- [ ] **Step 2: Write PerfBar.cs**

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBar.cs`:

```csharp
namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// The codified perf bar for the BlasManaged supply-chain-removal sprint.
/// These constants are set ONCE after baseline.json lands and do not move
/// without an explicit user-approved commit (see spec Section 7 escape hatches).
/// </summary>
public static class PerfBar
{
    /// <summary>
    /// Target win rate: BlasManaged must beat OpenBLAS on at least this percentage
    /// of catalog shapes. Set based on baseline.json inspection.
    /// </summary>
    public const int MinWinRatePercent = 80;  // SET BY PROJECT OWNER after baseline.json

    /// <summary>
    /// Ceiling for the worst remaining loss. No shape may be more than this
    /// multiple of OpenBLAS slower.
    /// </summary>
    public const double MaxLossMultiple = 1.20;  // SET BY PROJECT OWNER after baseline.json

    /// <summary>
    /// Catalog size at the time the bar was set. PerfBarTest asserts the
    /// current ShapeCatalog.All.Count matches this; if shapes are added
    /// post-sprint, the bar must be re-evaluated.
    /// </summary>
    public const int CatalogShapeCount = 64;  // SET BY PROJECT OWNER after baseline.json

    /// <summary>
    /// Hardware fingerprint of the authoritative runner. The perf test skips
    /// when running on a different host (numbers aren't comparable).
    /// </summary>
    public const string TargetHardwareFingerprint = "<runner-fingerprint>";  // SET BY PROJECT OWNER
}
```

- [ ] **Step 3: Write PerfBarTest.cs**

Create `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBarTest.cs`:

```csharp
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Benchmarks.BlasManaged;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class PerfBarTest
{
    [Fact]
    public void Catalog_Size_Matches_PerfBar()
    {
        Assert.Equal(PerfBar.CatalogShapeCount, ShapeCatalog.All.Count);
    }

    [Fact]
    public void WinRate_And_MaxLoss_Meet_PerfBar()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") != "1")
        {
            // Not the authoritative runner — skip (numbers aren't comparable).
            return;
        }

        if (HardwareFingerprint.Current.ToString() != PerfBar.TargetHardwareFingerprint)
        {
            // Wrong hardware — skip.
            return;
        }

        // Run the full catalog through PerfHarness; assert against PerfBar.
        var tmpPath = Path.Combine(Path.GetTempPath(), $"perfbar-{Guid.NewGuid():N}.json");
        try
        {
            PerfHarness.RunAll(ShapeCatalog.All, tmpPath);
            var output = JsonSerializer.Deserialize<HarnessOutput>(File.ReadAllText(tmpPath))!;

            int wins = output.Shapes.Count(r => r.NativeAvailable && r.RatioBmOverNative > 0 && r.RatioBmOverNative < 1.0);
            int total = output.Shapes.Count(r => r.NativeAvailable);
            int winPct = total > 0 ? (wins * 100) / total : 0;

            double worstLoss = output.Shapes
                .Where(r => r.NativeAvailable && r.RatioBmOverNative > 0)
                .Select(r => r.RatioBmOverNative)
                .DefaultIfEmpty(0)
                .Max();

            Assert.True(
                winPct >= PerfBar.MinWinRatePercent,
                $"Win rate {winPct}% < bar {PerfBar.MinWinRatePercent}% ({wins}/{total} wins)");

            Assert.True(
                worstLoss <= PerfBar.MaxLossMultiple,
                $"Worst loss {worstLoss:F2}x > bar {PerfBar.MaxLossMultiple:F2}x");
        }
        finally
        {
            if (File.Exists(tmpPath)) File.Delete(tmpPath);
        }
    }
}
```

- [ ] **Step 4: Verify on authoritative runner**

Run on the self-hosted runner:

```bash
AIDOTNET_PERF_RUNNER=1 dotnet test tests/AiDotNet.Tensors.Tests --filter "PerfBarTest" --logger "console;verbosity=normal"
```

Expected at this point: `WinRate_And_MaxLoss_Meet_PerfBar` **FAILS** (because no optimization has happened yet). The failure is the intended state until B/C/D/E land. The test is the gate Sub-issue G must satisfy before merging.

`Catalog_Size_Matches_PerfBar` should PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBar.cs \
        tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/PerfBarTest.cs
git commit -m "feat(#358-A): PerfBar.cs constants + PerfBarTest gate

PerfBar constants set against baseline.json on the authoritative
self-hosted runner. PerfBarTest skips on non-authoritative hosts
(env AIDOTNET_PERF_RUNNER) and on wrong hardware (fingerprint check).

Currently failing — that's the intended state. Sub-issues B/C/D/E
move shapes from native-wins to managed-wins until the bar is met.
Sub-issue G can only merge when this test passes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A.8: Document Sub-issue A complete; open Sub-issues B–G

**Files:**
- None (GitHub issue creation only)

- [ ] **Step 1: Update Sub-issue A's GitHub issue with sign-off comment**

Post a comment summarizing:
- Catalog size: N (must match PerfBar.CatalogShapeCount).
- Workload coverage: list of architectures represented.
- Instrumented coverage: source files that contributed shapes.
- Pre-sprint state: win count, worst loss, top-3 shapes by gap.
- Bar values committed: MinWinRatePercent, MaxLossMultiple, TargetHardwareFingerprint.

- [ ] **Step 2: Create Sub-issues B–G**

Each sub-issue is created with the body from the corresponding section of this plan ("Sub-issue B — Threading wire-up" etc.). All seven sub-issues link back to the mega tracking issue.

- [ ] **Step 3: Write detailed task lists for B–G**

For each of B–G, write a new plan file in `docs/superpowers/plans/`:
- `2026-05-XX-blas-perf-sprint-sub-B-threading.md`
- `2026-05-XX-blas-perf-sprint-sub-C-small-shape.md`
- `2026-05-XX-blas-perf-sprint-sub-D-microkernel.md`
- `2026-05-XX-blas-perf-sprint-sub-E-prepack.md`
- `2026-05-XX-blas-perf-sprint-sub-F-routing-shim.md`
- `2026-05-XX-blas-perf-sprint-sub-G-removal.md`

Each plan keys off the bench results from `baseline.json` so the task lists target real bottlenecks.

- [ ] **Step 4: Close Sub-issue A**

Mark Sub-issue A as complete on GitHub. The mega tracking issue updates its checkbox accordingly.

---

## Self-review (run after writing this plan)

1. **Spec coverage check.** Every section of the spec maps to either a sub-issue summary above or to Sub-issue A's detailed task list:
   - Spec §1 (motivation, win bar, gates) → Sub-issue A tasks A.6/A.7 codify the bar.
   - Spec §2 (architecture, layer cake, what's NOT rebuilt) → File structure table above maps every existing file.
   - Spec §3 (seven sub-issues) → Each has a section in this plan.
   - Spec §4 (benchmark methodology) → Sub-issue A tasks A.1–A.7.
   - Spec §5 (BlasMode contract) → Sub-issue B summary (introduces BlasMode.cs).
   - Spec §6 (routing shim mechanics) → Sub-issue F summary.
   - Spec §7 (out of scope, risks, escape hatches) → Acknowledged in Acceptance sections; escape hatches in spec apply uniformly.

2. **Placeholder scan.** All sub-issues B–G acknowledge "Detailed tasks: TBD — written after Sub-issue A lands." This is **intentional structural deferral**, not lazy placeholder work — the detailed tasks for those sub-issues depend on real numbers from `baseline.json`. The plan explicitly creates follow-up plan files in Task A.8.

3. **Type consistency check.** Methods and types referenced consistently:
   - `Shape` record: defined Task A.1, used in A.2/A.3/A.4/A.5/A.7. Signature unchanged throughout.
   - `BlasMode` enum: introduced only in Sub-issue B summary; no Sub-issue A task references it.
   - `BlasManaged.Gemm<T>`: used in Task A.5 with same signature as PR #366 implementation.
   - `BlasProvider.TryGemmEx`: signature `(m, n, k, a, aOffset, lda, transA, b, bOffset, ldb, transB, c, cOffset, ldc)` — matches usage in A.5 and Sub-issue F summary.
   - `ShapeInstrumenter.Record(int, int, int, bool, bool, DType)`: signature consistent A.2 → A.4.
   - `PerfHarness.RunShape(Shape)` → `ShapeResult`: signature consistent A.5 → A.7.
   - `PerfBar.CatalogShapeCount` / `MinWinRatePercent` / `MaxLossMultiple`: defined A.7, asserted in A.7 test.

4. **Coverage of "other stuff the user forgot about."** The spec's Section 7 lists out-of-scope items that need tracking issues:
   - LAPACK (cholesky/QR/SVD/LU) supply-chain removal — separate follow-up issue.
   - GPU BLAS (cuBLAS/rocBLAS/MPS) supply-chain removal — separate follow-up issue.
   - FP16/BF16/INT8 microkernels — separate follow-up issue.
   - Sparse/banded/symmetric/triangular BLAS variants — separate follow-up issue.
   - AMX (Sapphire Rapids tensor cores) path — separate follow-up issue.
   These five "tracked but out of scope" issues should be opened alongside the mega tracking issue.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-17-blas-managed-perf-sprint.md`. Sub-issue A is fully task-detailed; B–G have summaries suitable for GitHub issue bodies with detailed task lists deferred until A produces real bench data.

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration. Best fit for the long-running parallel sub-issues B–F.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints. Best fit for the linear Sub-issue A work (A.1 → A.7 are sequential).

Which approach?
