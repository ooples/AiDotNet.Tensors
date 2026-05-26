# BlasManaged streaming-pool domination — design spec

**Date**: 2026-05-26
**Issue**: #375 (Sub-G: native BLAS removal) + #368 (Mega: perf sprint)
**Status**: design — awaiting user approval before implementation plan

---

## 1. Goal & success criterion

> **Easily dominate PyTorch in head-to-head comparison — outside measurement noise, not within it.**

"Dominate" means a per-shape per-metric speedup of **≥1.5×** (i.e., **at least 50 % faster**, comfortably outside the ±20 % noise floor on shared dev boxes). The strict gate is: `BlasManaged.{Median, P95, P99, AllocBytes} ≤ PyTorch.{...} × 0.67` on every shape, every metric, against both `torch.eager` and `torch.compile`.

The 54-shape `BlasManaged` perf catalog (PR #460 expanded it to 76) drives the head-to-head. PyTorch baselines are taken on the same hardware fingerprint (`x64-amd-avx2-cpu16` today; will add an AVX-512 runner once CI provisioning is settled).

**Why this goal is hard**: PyTorch's CPU GEMM path is Intel MKL, with hand-tuned AVX-512 microkernels and a mature persistent thread pool (OpenMP). To beat them by 50 % we need (a) better microkernel quality OR (b) advantages they structurally can't have — process startup, deterministic mode, multi-tenant per-call thread budget, pre-packed weights.

## 2. Scope

Seven layers, each independently measurable:

| Layer | What | Drives |
|---|---|---|
| **A** | New `StreamingWorkerPool` — Streaming-only, lock-free, spin-then-park | 64³-family tiny shapes (today 10× behind) |
| **B** | M-axis + MN_2D fallback in Streaming | Same family + any tiny-shape gap |
| **C** | Microkernel hardening (AVX-512 verification, specialized paths) | Compute-bound large shapes (BERT, GPT FFN) |
| **D** | Differentiator benchmarks: cold-start, determinism, per-call threads, frozen-weight inference | Areas where PyTorch structurally can't compete |
| **E** | PyTorch head-to-head harness (TorchSharp + Python subprocess) + strict gate | Domination claim itself |
| **F** | Per-shape autotune cache learning | Cache the empirical winner; replace static heuristics |
| **G** | NUMA awareness (worker pinning + node-local arenas) | Multi-socket production servers |

## 3. Layer A — Tight Streaming pool

### 3.1 Why a new pool (not reuse PPE)

`PersistentParallelExecutor` exists and is already pretty good (`ManualResetEventSlim`-based wakeup). But it uses `lock(_executeLock)` per call for safety. At Streaming-call rates (a 64³ FP64 GEMM costs ~16 µs at AVX2 peak), the lock acquisition is single-digit % of overhead but the wakeup latency (sub-µs) plus completion-event Wait (another sub-µs) plus the serialization itself is meaningful for sub-20 µs operations. A Streaming-only pool can be more aggressive: lock-free producer queue, atomic dispatch, no completion event (caller polls a counter).

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ StreamingWorkerPool (singleton)                                     │
│                                                                     │
│   workers: Thread[N-1]      (pinned, started lazily on first Dispatch)│
│   slots:   WorkerSlot[N-1]  (cache-line-aligned)                    │
│                                                                     │
│   Each WorkerSlot:                                                  │
│     - volatile long seq          (dispatch generation counter)      │
│     - volatile Action<int>? body (current dispatch body)            │
│     - volatile int chunkStart    (assigned chunk start)             │
│     - volatile int chunkEnd      (exclusive)                        │
│     - volatile int parkPending   (0=hot, 1=parked)                  │
│     - ManualResetEventSlim park  (back-stop for long idle)          │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
   Worker loop:
     while (true) {
       long mySeq = Volatile.Read(slot.seq);
       int spin = 0;
       while (Volatile.Read(slot.seq) == mySeq) {
         if (++spin < SpinThreshold) Thread.SpinWait(1);
         else if (spin < ParkThreshold) Thread.Yield();
         else { slot.parkPending = 1; slot.park.Wait(); slot.park.Reset(); break; }
       }
       // New work: snapshot body, chunkStart, chunkEnd; run
       slot.body!(slot.chunkStart);  // single-chunk dispatch — caller iterates if needed
       Interlocked.Decrement(ref _remaining);
     }

   Dispatch (caller):
     1. _remaining = numWorkers
     2. for each slot: write chunkStart, chunkEnd, body; increment seq
     3. for each parked slot: park.Set()
     4. main thread runs chunk 0 (overlap)
     5. spin-then-park wait on _remaining == 0
```

### 3.3 Why this matches/beats MKL OpenMP

OpenMP's wakeup latency on Linux glibc is ~1-3 µs. Our spin-then-park hot path is **~50-200 ns** (Volatile.Read loop at 1 cycle/iteration; cache-line bouncing is the only cost on tightly-coupled cores). Worst case (cold workers parked) ~1-2 µs (MRE.Set + MRE.Wait wakeup) — comparable to OpenMP cold start.

### 3.4 Public API

```csharp
internal static class StreamingWorkerPool
{
    public static void Dispatch(int numChunks, Action<int> action);
    public static void Dispatch(int numChunks, long totalWork, Action<int> action); // serial-fallback below grain
}
```

Streaming's `RunNParallel`, `RunKParallel`, and the new `RunMParallel` all call this in place of `Parallel.For`.

### 3.5 Risks

- **Spin CPU**: idle workers spin for ~1 µs before yielding. Mitigated by spin-then-park — workers park within ~50 µs of last dispatch.
- **Cache-line bouncing**: the seq counter is read by all workers. Cache-line-aligned struct + per-worker padding (`[StructLayout(LayoutKind.Explicit, Size = 64)]`) prevents false sharing.
- **Reentrancy**: nested Dispatch falls back to serial (per-thread `[ThreadStatic] _isExecuting`).

## 4. Layer B — M-axis + MN_2D fallback in Streaming

Re-introduces the `RunMParallel` from PR #462's reverted commit, but now dispatching through `StreamingWorkerPool` instead of `Parallel.For`. Should deliver actual speedup on 64³ since dispatch latency drops from ~10 µs to ~200 ns per task.

`Streaming.Run` decision tree:

```
axis = AxisSelector.Select(...)
if axis == K && !deterministic && procs > 1   → RunKParallel
if axis == N && n >= procs * Nr               → RunNParallel
if axis == M && m >= procs * Mr               → RunMParallel       // NEW
if axis == MN_2D                                                    // NEW fallback
   if m >= n                                  → RunMParallel
   else                                       → RunNParallel
otherwise                                     → RunSerial
```

## 5. Layer C — Microkernel hardening

**Targeted, not speculative.** Driven by which shapes still fail the strict gate after Layers A+B.

Likely targets (from #460's worst-loss list):
- **BERT_FFN_up 1024×3072×768 FP32**: 9% of MKL's per-core throughput. Need to verify AVX-512 microkernels fire on AVX-512 hosts (they exist in code; gate by `Avx512F.IsSupported`). If they fire and we're still 10× behind, the gap is microkernel quality vs MKL's hand-tuned assembly — a multi-week investigation, not realistic in this PR. Honest position: document the gap, plan for follow-up sprint.
- **GPT2med_FFN_up 512×4096×1024 FP32**: same family.
- **128×768×768 FP64**: A/B showed PackAOnly@8thr does 4.4× better than the current dispatcher pick. May close with Layer F (autotune learning) alone.

## 6. Layer D — Differentiator benchmarks

New directory: `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/`

| File | Measures | Why PyTorch can't match |
|---|---|---|
| `ColdStartBench.cs` | process start → first GEMM call wall-clock | Python interpreter + torch import overhead (~250 ms minimum). We're <50 ms with NativeAOT. |
| `DeterminismBench.cs` | 1000 GEMM calls, assert bit-identical across thread counts 1/2/4/8/16 | PyTorch documents non-determinism in multi-threaded reductions. Our `Deterministic` mode guarantees bit-exact. |
| `PerCallThreadsBench.cs` | Two concurrent `Gemm` calls with different `NumThreads` (4 vs 8) on the same process | `torch.set_num_threads()` is process-global. Multi-tenant servers can't split. |
| `FrozenWeightInferenceBench.cs` | 1000-call inference loop with weights pinned via `FrozenWeightRegistry.RegisterFrozen` | PyTorch re-packs B per call (the packing is internal to the kernel call). We cache the packed B. |

Each bench produces a Markdown table; a top-level `PyTorchComparisonReport.cs` runs all four and emits the combined report.

## 7. Layer E — PyTorch head-to-head + strict gate

### 7.1 Two harnesses, two views

1. **TorchSharp** (in-process via libtorch P/Invoke) — eager-mode parity, microsecond-accurate, no IPC pollution. Extends the existing `AutogradComparisonBenchmarks.cs` pattern.
2. **Python subprocess** — eager AND `torch.compile`. Extends existing `BaselineRunners/`. IPC adds ~ms per call, so the harness amortizes by batching N calls per subprocess invocation.

### 7.2 Strict gate

For each shape × each metric (median ms, p95 ms, p99 ms, alloc bytes/call):

```
PASS  iff  BlasManaged_metric ≤ PyTorch_metric × 0.67   (≥1.5× faster)
WARN  iff  BlasManaged_metric ≤ PyTorch_metric × 1.05   (parity within noise)
FAIL  otherwise                                          (clearly slower)
```

### 7.3 Report

`artifacts/perf/pytorch-comparison.md`: per-shape PASS/WARN/FAIL per metric × per harness. Plus a summary row: "Dominates N/54 vs torch.eager, M/54 vs torch.compile."

## 8. Layer F — Per-shape autotune cache

### 8.1 Cache

```csharp
internal sealed class AutotuneCache
{
    record ShapeKey(int M, int N, int K, bool TransA, bool TransB, DType Dtype);
    record Entry(
        PackingMode Mode,
        int NumThreads, int Mc, int Nc, int Kc, int Mr, int Nr,
        double MeasuredMs, int SampleCount, long LastReTune);

    ConcurrentDictionary<ShapeKey, Entry> _cache = new();
    ConcurrentDictionary<ShapeKey, byte> _warmup = new();   // shapes currently in warm-up sweep
}
```

### 8.2 Warmup sweep

First 3 calls of a new shape: run the static-dispatcher pick + 2 alternatives (e.g., Streaming, PackAOnly @ 8 threads, PackBoth @ 16 threads). Time each. Pick min. Cache it.

Single-shape-locked during warmup so concurrent callers serialize on the first 3 calls (prevents thundering-herd autotune). After warmup, all callers read the cache lock-free.

### 8.3 Re-tune trigger

If observed call median exceeds `cache.MeasuredMs × 1.5` for 10 consecutive calls, mark the entry stale and re-sweep. Catches thermal throttling, noisy-neighbor changes, NUMA migrations.

### 8.4 Persistence

Optional JSON dump to `~/.aidotnet/blasmanaged-autotune-{HardwareFingerprint}.json`. Loaded on first use. Disabled in `BlasMode.Deterministic` (timing-based decisions leak non-determinism).

## 9. Layer G — NUMA awareness

### 9.1 Detection

Cross-platform NUMA topology detection:
- **Windows**: `GetNumaHighestNodeNumber` + `GetNumaProcessorNode` (kernel32).
- **Linux**: `/sys/devices/system/node/node*/cpulist` parse.
- Fallback: assume single-node when detection fails.

### 9.2 Worker pinning

`StreamingWorkerPool` workers pinned per NUMA node. Workers 0..(N₀-1) → node 0; N₀..(N₀+N₁-1) → node 1; etc. Pinning via `Thread.SetIdealProcessor` (Windows) or `sched_setaffinity` P/Invoke (Linux).

### 9.3 Per-node arenas (deferred)

NUMA-local allocation (`VirtualAllocExNuma` / `numa_alloc_onnode`) is **deferred to a follow-up PR**. Reason: requires cross-platform native interop for memory placement, not standard .NET. Layer G in THIS PR delivers worker pinning + per-node work splitting (using the existing managed allocator); node-local allocation is the next iteration.

### 9.4 Single-socket fallback

When detection finds 1 node (most dev boxes, most CI runners), Layer G is a no-op. Workers stay unpinned, no per-node split logic engaged. Zero overhead.

## 10. Implementation order (suggested)

1. **A + B together** — pool + Streaming wiring (Layers A & B are tightly coupled; ship as one commit).
2. **D** — differentiator benchmarks (parallel work; no dependencies on A-B).
3. **E** — PyTorch harness + strict gate (needs A-B and D done to produce the report).
4. **F** — autotune cache (independent of pool; can be parallel with A-B).
5. **G** — NUMA worker pinning (independent; bolts onto A's pool).
6. **C** — microkernel hardening (data-driven from E's failure list).

## 11. Correctness budget

- **All 320 BlasManaged + ScalarKernel + Determinism + PackBoth + PackAOnly + Streaming + WeightRegistry tests must pass** at every commit.
- **New tests**: `StreamingWorkerPool` correctness (dispatch, reentrancy, exception propagation), `AutotuneCache` correctness (warmup convergence, re-tune trigger), `NumaTopology` correctness (cross-platform detection).
- **Bit-exact assertion**: `DeterminismBench` doubles as a correctness test — assert across N runs.

## 12. Performance budget

- Layer A+B: 64³ FP64 goes from 4 GFLOPS to ≥20 GFLOPS (5× win, closes the OpenBLAS gap to within 1.5×).
- Layer C (if microkernel work happens): BERT_FFN_up FP32 within 2× of MKL on AVX-512 host (current: 10×).
- Layer F: stale-cache shapes still pick correctly across thermal/load regimes.
- Layer G: 2-socket NUMA host shows ~1.5-1.8× speedup on large shapes vs unpinned baseline.
- **Final strict gate**: aim for ≥40/54 PASS vs `torch.eager`; ≥30/54 vs `torch.compile`. Anything less than this and the "dominate" claim is honest about its limits.

## 13. Out of scope

- GPU comparison — different cluster of issues, tracked separately.
- `torch.compile`'s Triton kernels reaching equivalent FP16/BF16 paths — we don't have FP16/BF16 microkernels at all yet.
- LAPACK operations (cholesky, QR, SVD) — separate sprint (#376).
- Cross-platform NUMA *allocation* (just pinning in this PR).

## 14. Risks

- **Microkernel gap unresolvable in this PR**: BERT-class compute-bound shapes may still fail the strict gate. The PR ships with a honest report rather than blocking on this.
- **PPE divergence**: introducing `StreamingWorkerPool` while keeping PPE means two pools coexist. If they fight for CPU (e.g., a `LightweightParallel` call followed by a `Streaming` call), the second pool's workers may not be hot. Mitigation: both pools share the same set of pinned cores when NUMA is enabled.
- **Autotune cache cold start**: first 3 calls of every shape pay the sweep cost. Long-tail of unique shapes (e.g., dynamic seq-len attention) churns the cache. Mitigation: persistence (Section 8.4) warm-starts the cache.
- **NUMA detection fragility**: Linux `/sys/devices` parsing assumes standard kernel layout. Container environments may hide topology. Fallback to single-node when detection fails.

## 15. References

- `src/AiDotNet.Tensors/Helpers/PersistentParallelExecutor.cs` — existing pool, design reference.
- `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs` — wiring target.
- `src/AiDotNet.Tensors/Engines/BlasManaged/Dispatcher/AxisSelector.cs` — parallelism-axis heuristic.
- `tests/AiDotNet.Tensors.Benchmarks/AutogradComparisonBenchmarks.cs` — existing TorchSharp pattern.
- `tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/PythonBaselineRunner.cs` — existing Python subprocess driver.
- PR #460 (`feat/369-refresh-perf-catalog-and-baseline`) — refreshed shape catalog + metrics.
- PR #462 (`perf/blas-managed-fp64-small-square-microkernel`) — dispatcher rebalance + reverted-M-axis prototype (this PR builds on its branch).
- Issue #368 — parent mega; Issue #375 — Sub-G native removal.
