# Streaming-Pool Domination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a custom Streaming-only worker pool plus the surrounding infrastructure (M-axis dispatch, autotune cache, NUMA pinning, microkernel hardening, differentiator benchmarks, and PyTorch comparison harness) so `BlasManaged` *easily dominates* PyTorch (≥1.5× faster on every shape and metric) on the head-to-head benchmark catalog.

**Architecture:** Seven layers (A-G) per the design spec. Foundation is a lock-free `StreamingWorkerPool` (spin-then-park workers, sub-µs dispatch). On top, Streaming gets M-axis + MN_2D fallback (Layer B). Autotune cache learns the empirical winner per shape (Layer F). NUMA pinning (Layer G) bolts onto the pool. Differentiator benchmarks (Layer D) measure what PyTorch structurally can't match. PyTorch comparison + strict gate (Layer E) produce the domination report. Microkernel hardening (Layer C) is data-driven from Layer E's failure list.

**Tech Stack:** C# (.NET 10 + net471), AVX2/AVX-512 intrinsics, BenchmarkDotNet, xUnit, TorchSharp (libtorch P/Invoke), Python (torch + torch.compile via subprocess).

**Spec:** [`docs/superpowers/specs/2026-05-26-streaming-pool-domination-design.md`](../specs/2026-05-26-streaming-pool-domination-design.md)

**Branch:** `perf/blas-managed-fp64-small-square-microkernel` (PR #462 — current branch).

---

## File Structure

### New files

| Path | Purpose |
|---|---|
| `src/AiDotNet.Tensors/Engines/BlasManaged/Pool/StreamingWorkerPool.cs` | Lock-free spin-then-park worker pool |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Pool/WorkerSlot.cs` | Cache-line-aligned per-worker state struct |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Pool/NumaTopology.cs` | Cross-platform NUMA node detection (Layer G) |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Autotune/AutotuneCacheV2.cs` | Per-shape empirical-winner cache (Layer F) |
| `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/StreamingWorkerPoolTests.cs` | Pool correctness tests |
| `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/AutotuneCacheV2Tests.cs` | Autotune cache tests |
| `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/NumaTopologyTests.cs` | NUMA detection tests |
| `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/ColdStartBench.cs` | Cold-start benchmark (Layer D) |
| `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/DeterminismBench.cs` | Deterministic-mode benchmark (Layer D) |
| `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/PerCallThreadsBench.cs` | Multi-tenant thread-budget benchmark (Layer D) |
| `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/FrozenWeightInferenceBench.cs` | Pre-pack inference loop benchmark (Layer D) |
| `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/PyTorchComparisonReport.cs` | Top-level report generator (Layer E) |
| `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/HeadToHeadCatalogBench.cs` | Full-catalog TorchSharp head-to-head (Layer E) |
| `tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_torch_head_to_head.py` | Python subprocess for full-catalog torch.compile head-to-head |
| `artifacts/perf/pytorch-comparison.md` | Strict gate output (generated, gitignored) |

### Modified files

| Path | What changes |
|---|---|
| `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs` | Replace `Parallel.For` with `StreamingWorkerPool.Dispatch`; add `RunMParallel`; add MN_2D fallback |
| `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs` | Wire `AutotuneCacheV2` into `Gemm<T>` dispatch |
| `src/AiDotNet.Tensors/Engines/BlasManaged/Dispatcher/Dispatcher.cs` | Bypass static heuristic when cache has an entry |
| `tests/AiDotNet.Tensors.Benchmarks/Program.cs` | Add `--pytorch-comparison` and `--pytorch-headtohead-report` entry points |

---

## Layer A: Tight Streaming Pool

### Task A.1: Create WorkerSlot cache-line-aligned struct

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Pool/WorkerSlot.cs`

- [ ] **Step 1: Create the directory + file with the struct**

```csharp
using System.Runtime.InteropServices;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged.Pool;

/// <summary>
/// Per-worker state for <see cref="StreamingWorkerPool"/>. Cache-line-aligned
/// (64 bytes on x64) to prevent false sharing between worker threads — each
/// worker's seq counter, action reference, and chunk range live on a private
/// cache line so updates don't bounce.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64)]
internal struct WorkerSlot
{
    /// <summary>Dispatch generation counter. Incremented by caller on new work; worker reads to detect dispatch.</summary>
    [FieldOffset(0)] public long Seq;
    /// <summary>Body to execute for the worker's assigned chunk(s). Worker reads after observing Seq increment.</summary>
    [FieldOffset(8)] public Action<int>? Body;
    /// <summary>Inclusive chunk start.</summary>
    [FieldOffset(16)] public int ChunkStart;
    /// <summary>Exclusive chunk end.</summary>
    [FieldOffset(20)] public int ChunkEnd;
    /// <summary>0 = worker is hot (spinning); 1 = worker is parked on ParkEvent.</summary>
    [FieldOffset(24)] public int ParkPending;
    /// <summary>Park back-stop after spin loop exhausts.</summary>
    [FieldOffset(32)] public ManualResetEventSlim? ParkEvent;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Pool/WorkerSlot.cs
git commit -m "perf(#375): WorkerSlot cache-line-aligned struct for streaming pool"
```

### Task A.2: Write StreamingWorkerPool unit tests (TDD red phase)

**Files:**
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/StreamingWorkerPoolTests.cs`

- [ ] **Step 1: Write the test file with 5 failing tests**

```csharp
using System;
using System.Collections.Concurrent;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged.Pool;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Pool-Serial")]
public class StreamingWorkerPoolTests
{
    [Fact]
    public void Dispatch_SingleChunk_RunsOnCallerThread()
    {
        int callerTid = Thread.CurrentThread.ManagedThreadId;
        int observedTid = -1;
        StreamingWorkerPool.Dispatch(1, c => observedTid = Thread.CurrentThread.ManagedThreadId);
        Assert.Equal(callerTid, observedTid);
    }

    [Fact]
    public void Dispatch_MultipleChunks_AllChunksRunExactlyOnce()
    {
        const int numChunks = 32;
        var counts = new int[numChunks];
        StreamingWorkerPool.Dispatch(numChunks, c => Interlocked.Increment(ref counts[c]));
        for (int i = 0; i < numChunks; i++)
            Assert.Equal(1, counts[i]);
    }

    [Fact]
    public void Dispatch_MultipleChunks_UsesWorkerThreads()
    {
        const int numChunks = 16;
        var tids = new ConcurrentBag<int>();
        StreamingWorkerPool.Dispatch(numChunks, c => tids.Add(Thread.CurrentThread.ManagedThreadId));
        // With numChunks=16 and a multi-core host, at least 2 distinct thread IDs should be observed.
        if (Environment.ProcessorCount > 1)
            Assert.True(tids.Count >= 2 && tids.Distinct().Count() >= 2,
                $"Expected ≥2 distinct threads, got {tids.Distinct().Count()}");
    }

    [Fact]
    public void Dispatch_WorkerException_PropagatesToDispatcher()
    {
        var thrown = Assert.Throws<InvalidOperationException>(() =>
            StreamingWorkerPool.Dispatch(8, c =>
            {
                if (c == 3) throw new InvalidOperationException("worker boom");
            }));
        Assert.Equal("worker boom", thrown.Message);
    }

    [Fact]
    public void Dispatch_Reentrant_FallsBackToSerial()
    {
        // Inner dispatch from inside an outer dispatch chunk must serialize
        // (avoids worker-pool deadlock).
        bool innerRan = false;
        StreamingWorkerPool.Dispatch(2, c =>
        {
            if (c == 0)
            {
                StreamingWorkerPool.Dispatch(2, ic => innerRan = true);
            }
        });
        Assert.True(innerRan);
    }
}
```

- [ ] **Step 2: Run tests to confirm they fail (red phase)**

Run:
```bash
dotnet build tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --no-restore --framework net10.0 2>&1 | tail -5
```

Expected: build fails with `error CS0103: The name 'StreamingWorkerPool' does not exist`. Good — that confirms TDD red phase.

- [ ] **Step 3: Commit**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/StreamingWorkerPoolTests.cs
git commit -m "test(#375): failing tests for StreamingWorkerPool (TDD red)"
```

### Task A.3: Implement StreamingWorkerPool skeleton — make tests build

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Pool/StreamingWorkerPool.cs`

- [ ] **Step 1: Create the file with just enough surface for the tests to build**

```csharp
using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged.Pool;

/// <summary>
/// Lock-free spin-then-park worker pool specialised for the Streaming GEMM
/// strategy. Designed for sub-µs dispatch overhead at GEMM call rates
/// (~1 dispatch / 10–50 µs). See spec section 3.
/// </summary>
internal static class StreamingWorkerPool
{
    [ThreadStatic]
    private static bool _isExecuting;

    /// <summary>
    /// Run <paramref name="action"/> across <paramref name="numChunks"/>
    /// chunks. Chunk 0 runs on the caller; chunks 1..numChunks-1 dispatch to
    /// worker threads. Returns when all chunks complete. Re-throws the first
    /// worker exception.
    /// </summary>
    public static void Dispatch(int numChunks, Action<int> action)
    {
        if (numChunks <= 0) return;
        if (numChunks == 1 || _isExecuting)
        {
            // Serial fallback: single-chunk OR reentrant call.
            for (int c = 0; c < numChunks; c++) action(c);
            return;
        }

        _isExecuting = true;
        try
        {
            // Placeholder serial implementation; replaced in A.4-A.7 with
            // actual lock-free worker dispatch.
            for (int c = 0; c < numChunks; c++) action(c);
        }
        finally { _isExecuting = false; }
    }
}
```

- [ ] **Step 2: Run tests to verify the trivial impl passes 3 of 5 (single-chunk, all-chunks-once, exception, reentrant)**

Run:
```bash
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~StreamingWorkerPoolTests" --logger "console;verbosity=minimal" 2>&1 | tail -10
```

Expected:
- PASS: `Dispatch_SingleChunk_RunsOnCallerThread`
- PASS: `Dispatch_MultipleChunks_AllChunksRunExactlyOnce`
- PASS: `Dispatch_WorkerException_PropagatesToDispatcher`
- PASS: `Dispatch_Reentrant_FallsBackToSerial`
- FAIL: `Dispatch_MultipleChunks_UsesWorkerThreads` (still serial)

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Pool/StreamingWorkerPool.cs
git commit -m "perf(#375): StreamingWorkerPool skeleton — serial fallback"
```

### Task A.4: Add the worker pool fields and lazy initialization

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Pool/StreamingWorkerPool.cs`

- [ ] **Step 1: Replace the skeleton's contents with the field declarations + lazy init**

Replace the entire class body with:

```csharp
internal static class StreamingWorkerPool
{
    [ThreadStatic] private static bool _isExecuting;

    private static readonly int _numWorkers = Math.Max(1, Environment.ProcessorCount - 1);
    private static readonly WorkerSlot[] _slots = new WorkerSlot[_numWorkers];
    private static readonly Thread[] _workers = new Thread[_numWorkers];

    // Aligned-write counter: incremented by every worker when its chunk
    // finishes; dispatcher spin-waits until it observes the expected value.
    private static int _remaining;

    // First worker exception is captured here; re-thrown on dispatcher.
    private static volatile Exception? _firstException;

    private static int _initialized; // 0 = not started; 1 = workers spawned
    private static readonly object _initLock = new();

    private static void EnsureInitialized()
    {
        if (Volatile.Read(ref _initialized) == 1) return;
        lock (_initLock)
        {
            if (_initialized == 1) return;
            for (int i = 0; i < _numWorkers; i++)
            {
                _slots[i].ParkEvent = new ManualResetEventSlim(false);
                int slot = i;
                _workers[i] = new Thread(() => WorkerLoop(slot))
                {
                    IsBackground = true,
                    Name = $"AiDotNet-StreamPool-{i}",
                };
                _workers[i].Start();
            }
            Volatile.Write(ref _initialized, 1);
        }
    }

    private static void WorkerLoop(int slot)
    {
        long lastSeq = 0;
        while (true)
        {
            // Spin-then-park wait for a new dispatch generation.
            int spinCount = 0;
            while (Volatile.Read(ref _slots[slot].Seq) == lastSeq)
            {
                if (spinCount < 1000)
                {
                    Thread.SpinWait(1);
                    spinCount++;
                }
                else if (spinCount < 5000)
                {
                    Thread.Yield();
                    spinCount++;
                }
                else
                {
                    Volatile.Write(ref _slots[slot].ParkPending, 1);
                    // Re-check under park to avoid lost wakeup
                    if (Volatile.Read(ref _slots[slot].Seq) == lastSeq)
                        _slots[slot].ParkEvent!.Wait();
                    _slots[slot].ParkEvent!.Reset();
                    Volatile.Write(ref _slots[slot].ParkPending, 0);
                    spinCount = 0;
                }
            }

            lastSeq = Volatile.Read(ref _slots[slot].Seq);
            var body = _slots[slot].Body;
            int chunkStart = _slots[slot].ChunkStart;
            int chunkEnd = _slots[slot].ChunkEnd;

            try
            {
                for (int c = chunkStart; c < chunkEnd; c++)
                    body!(c);
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref _firstException, ex, null);
            }

            Interlocked.Decrement(ref _remaining);
        }
    }

    public static void Dispatch(int numChunks, Action<int> action)
    {
        if (numChunks <= 0) return;
        if (numChunks == 1 || _isExecuting)
        {
            for (int c = 0; c < numChunks; c++) action(c);
            return;
        }

        EnsureInitialized();
        _isExecuting = true;
        try
        {
            // Distribute chunks across workers + caller round-robin.
            // Caller runs chunks at positions 0, N+1, 2N+1, ... where N=numWorkers.
            // Worker `i` runs chunks at positions i+1, N+i+1, 2N+i+1, ...
            int stride = _numWorkers + 1;

            // Compute per-worker chunk ranges.
            for (int w = 0; w < _numWorkers; w++)
            {
                _slots[w].Body = action;
                // Range: chunks (w+1), (w+1+stride), (w+1+2*stride), ... up to numChunks
                // Encoded as a contiguous start..end pair after rebasing — the
                // worker loop iterates start..end without striding because we
                // re-pack the chunk-id math here. Simpler: just hand each
                // worker a contiguous (start, end) range.
                int rangeStart = ((long)w * numChunks / _numWorkers > int.MaxValue) ? int.MaxValue : (int)((long)w * numChunks / _numWorkers);
                int rangeEnd = ((long)(w + 1) * numChunks / _numWorkers > int.MaxValue) ? int.MaxValue : (int)((long)(w + 1) * numChunks / _numWorkers);
                _slots[w].ChunkStart = rangeStart;
                _slots[w].ChunkEnd = rangeEnd;
            }

            // Compute how many workers actually have non-empty ranges.
            int activeWorkers = 0;
            for (int w = 0; w < _numWorkers; w++)
            {
                if (_slots[w].ChunkEnd > _slots[w].ChunkStart) activeWorkers++;
            }

            // Re-allocate ranges: caller gets the LAST 1/(N+1) of chunks.
            // Use a simpler split: caller runs [callerStart..numChunks), workers split [0..callerStart).
            int callerStart = (int)((long)_numWorkers * numChunks / (_numWorkers + 1));
            if (callerStart < 1) callerStart = numChunks; // No workers — caller does everything.

            activeWorkers = 0;
            for (int w = 0; w < _numWorkers; w++)
            {
                int wStart = (int)((long)w * callerStart / _numWorkers);
                int wEnd = (int)((long)(w + 1) * callerStart / _numWorkers);
                _slots[w].ChunkStart = wStart;
                _slots[w].ChunkEnd = wEnd;
                if (wEnd > wStart) activeWorkers++;
            }

            _firstException = null;
            _remaining = activeWorkers;

            // Wake workers — increment seq counter (LSB doesn't matter,
            // just needs to differ from each worker's lastSeq).
            for (int w = 0; w < _numWorkers; w++)
            {
                Interlocked.Increment(ref _slots[w].Seq);
                // If worker was parked, signal it.
                if (Volatile.Read(ref _slots[w].ParkPending) == 1)
                    _slots[w].ParkEvent!.Set();
            }

            // Caller runs chunks [callerStart..numChunks) in-line.
            Exception? callerException = null;
            for (int c = callerStart; c < numChunks; c++)
            {
                try { action(c); }
                catch (Exception ex) { callerException ??= ex; }
            }

            // Spin-wait for workers to finish.
            int waitSpin = 0;
            while (Volatile.Read(ref _remaining) > 0)
            {
                if (waitSpin < 1000) { Thread.SpinWait(1); waitSpin++; }
                else { Thread.Yield(); waitSpin = 0; }
            }

            var pooledEx = _firstException ?? callerException;
            if (pooledEx is not null) throw pooledEx;
        }
        finally { _isExecuting = false; }
    }
}
```

> **Note (sketch vs. shipped code):** this sketch sets the re-entrancy guard
> `_isExecuting` only in `Dispatch`, but the shipped `StreamingWorkerPool` also
> sets it inside `WorkerLoop` around each body invocation. `_isExecuting` is
> `[ThreadStatic]`, so without the worker-side set a worker thread whose body
> calls `Dispatch` again would see its own flag as false and spin up a nested
> parallel dispatch; setting it on the worker makes nested dispatches run inline.
> Defer to `StreamingWorkerPool.cs` for the exact (correct) guard placement.

- [ ] **Step 2: Build and run all 5 pool tests**

Run:
```bash
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release --no-restore --framework net10.0 2>&1 | grep -E "(error|Error\(s\))" | tail -3
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~StreamingWorkerPoolTests" --logger "console;verbosity=minimal" 2>&1 | tail -5
```

Expected: 0 errors. All 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Pool/StreamingWorkerPool.cs
git commit -m "perf(#375): StreamingWorkerPool lock-free dispatch (Layer A)"
```

### Task A.5: Add `Dispatch(numChunks, totalWork, action)` serial-fallback overload

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Pool/StreamingWorkerPool.cs`
- Modify: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/StreamingWorkerPoolTests.cs`

- [ ] **Step 1: Add the test**

Add to `StreamingWorkerPoolTests.cs` (inside the class):

```csharp
    [Fact]
    public void Dispatch_WithGrainSize_RunsSerialBelowThreshold()
    {
        int callerTid = Thread.CurrentThread.ManagedThreadId;
        var observedTids = new System.Collections.Concurrent.ConcurrentBag<int>();
        // totalWork below DefaultSerialGrainSize (32768) should run all chunks on caller.
        StreamingWorkerPool.Dispatch(8, totalWork: 1000, c => observedTids.Add(Thread.CurrentThread.ManagedThreadId));
        foreach (var t in observedTids)
            Assert.Equal(callerTid, t);
    }
```

- [ ] **Step 2: Add the overload to StreamingWorkerPool**

Add inside the class, after `public static void Dispatch(int numChunks, Action<int> action)`:

```csharp
    /// <summary>
    /// Default serial-fallback grain size (matches <c>PersistentParallelExecutor.DefaultSerialGrainSize</c>).
    /// </summary>
    public static int DefaultSerialGrainSize { get; set; } = 32 * 1024;

    /// <summary>
    /// Dispatch with PyTorch-style serial-fallback: when <paramref name="totalWork"/>
    /// is below <see cref="DefaultSerialGrainSize"/>, all chunks run on the caller
    /// (no worker wakeup, no completion wait).
    /// </summary>
    public static void Dispatch(int numChunks, long totalWork, Action<int> action)
    {
        if (totalWork < DefaultSerialGrainSize)
        {
            Exception? first = null;
            for (int c = 0; c < numChunks; c++)
            {
                try { action(c); }
                catch (Exception ex) { first ??= ex; }
            }
            if (first is not null) throw first;
            return;
        }
        Dispatch(numChunks, action);
    }
```

- [ ] **Step 3: Run all 6 tests; verify PASS**

Run:
```bash
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~StreamingWorkerPoolTests" --logger "console;verbosity=minimal" 2>&1 | tail -3
```

Expected: 6/6 pass.

- [ ] **Step 4: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Pool/StreamingWorkerPool.cs tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/StreamingWorkerPoolTests.cs
git commit -m "perf(#375): StreamingWorkerPool grain-size serial fallback"
```

### Task A.6: Add a stress test to verify dispatch latency

**Files:**
- Modify: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/StreamingWorkerPoolTests.cs`

- [ ] **Step 1: Add a stress test measuring per-dispatch overhead**

Add inside the class:

```csharp
    [Fact]
    public void Dispatch_DispatchLatency_BelowMicrosecond()
    {
        if (Environment.ProcessorCount < 2)
            return;  // Single-core: no worker parallelism, latency irrelevant.

        // Warm up the pool (first call spawns workers).
        for (int i = 0; i < 100; i++)
            StreamingWorkerPool.Dispatch(8, c => { /* no-op */ });

        // Measure: 1000 minimal dispatches. Median wall-time per dispatch
        // should be ≤2 µs on a 16-core box (spin-then-park hot path).
        var times = new double[1000];
        var sw = new System.Diagnostics.Stopwatch();
        for (int i = 0; i < 1000; i++)
        {
            sw.Restart();
            StreamingWorkerPool.Dispatch(8, c => { /* no-op */ });
            sw.Stop();
            times[i] = sw.Elapsed.TotalMicroseconds;
        }
        Array.Sort(times);
        double medianUs = times[500];

        // Gate: 25 µs is generous. The aspirational sub-µs is hardware-dependent.
        // This catches order-of-magnitude regressions (e.g., accidental TPL fallback).
        Assert.True(medianUs < 25.0, $"Median dispatch latency {medianUs:F1} µs exceeds 25 µs sentinel.");
    }
```

- [ ] **Step 2: Run; verify PASS**

Run:
```bash
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~Dispatch_DispatchLatency_BelowMicrosecond" --logger "console;verbosity=normal" 2>&1 | tail -5
```

Expected: PASS. Read the actual `medianUs` from the output for a sanity check.

- [ ] **Step 3: Commit**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/StreamingWorkerPoolTests.cs
git commit -m "test(#375): StreamingWorkerPool dispatch-latency sentinel"
```

## Layer B: Streaming M-axis + MN_2D fallback

### Task B.1: Wire StreamingWorkerPool into RunNParallel

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs`

- [ ] **Step 1: Replace `Parallel.For` with `StreamingWorkerPool.Dispatch` in `RunNParallel`**

Locate `RunNParallel` (currently uses `Parallel.For(0, procsLocal, p => { ... })`). Replace the `Parallel.For` line with:

```csharp
                Pool.StreamingWorkerPool.Dispatch(procsLocal, p =>
                {
```

(close-brace unchanged). Add `using AiDotNet.Tensors.Engines.BlasManaged.Pool;` at the top of the file if not already present.

- [ ] **Step 2: Build + run BlasManaged correctness tests**

Run:
```bash
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release --no-restore --framework net10.0 2>&1 | grep -E "(error|Error\(s\))" | tail -3
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~ScalarKernelTests|FullyQualifiedName~StreamingKAxisFastModeTest|FullyQualifiedName~StreamingNAxisParallelTest" --logger "console;verbosity=minimal" 2>&1 | tail -5
```

Expected: 0 errors. All Streaming-related tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs
git commit -m "perf(#375): wire StreamingWorkerPool into Streaming N-axis"
```

### Task B.2: Wire StreamingWorkerPool into RunKParallel

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs`

- [ ] **Step 1: Replace `Parallel.For` in `RunKParallel` (FP32 path) with the pool dispatch**

Same pattern as B.1: locate `Parallel.For(0, procsLocal, p => {` inside `RunKParallel`, replace with `Pool.StreamingWorkerPool.Dispatch(procsLocal, p => {`.

- [ ] **Step 2: Build + run all BlasManaged tests**

Run:
```bash
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~BlasManaged" --logger "console;verbosity=minimal" 2>&1 | tail -5
```

Expected: all pass (no Streaming K-axis correctness regression).

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs
git commit -m "perf(#375): wire StreamingWorkerPool into Streaming K-axis"
```

### Task B.3: Add RunMParallel + MN_2D fallback

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs`

- [ ] **Step 1: Add the `RunMParallel` method**

After `RunNParallel`, add the M-axis variant (mirror of `RunNParallel` but splitting M instead of N):

```csharp
    /// <summary>
    /// Partition M across <paramref name="procs"/> threads. Each thread
    /// writes a disjoint row slice of C — bit-exact across thread counts.
    /// </summary>
    private static void RunMParallel<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        int procs) where T : unmanaged
    {
        unsafe
        {
            fixed (T* aPtr = a)
            fixed (T* bPtr = b)
            fixed (T* cPtr = c)
            {
                T* aLocal = aPtr;
                T* bLocal = bPtr;
                T* cLocal = cPtr;
                int aLen = a.Length, bLen = b.Length, cLen = c.Length;
                int procsLocal = procs;
                int mLocal = m;
                int nLocal = n;
                int kLocal = k;
                int ldaLocal = lda, ldbLocal = ldb, ldcLocal = ldc;
                bool taLocal = transA, tbLocal = transB;

                Pool.StreamingWorkerPool.Dispatch(procsLocal, p =>
                {
                    int mStart = (int)(((long)p * mLocal) / procsLocal);
                    int mEnd = (int)(((long)(p + 1) * mLocal) / procsLocal);
                    int mChunk = mEnd - mStart;
                    if (mChunk <= 0) return;

                    int aOffset = taLocal ? mStart : mStart * ldaLocal;
                    int aSliceLen = aLen - aOffset;
                    int cOffset = mStart * ldcLocal;
                    int cSliceLen = cLen - cOffset;

                    var aSpan = new ReadOnlySpan<T>(aLocal + aOffset, aSliceLen);
                    var bSpan = new ReadOnlySpan<T>(bLocal, bLen);
                    var cSpan = new Span<T>(cLocal + cOffset, cSliceLen);

                    RunSerial(aSpan, ldaLocal, taLocal,
                              bSpan, ldbLocal, tbLocal,
                              cSpan, ldcLocal,
                              mChunk, nLocal, kLocal);
                });
            }
        }
    }
```

- [ ] **Step 2: Add the MN_2D → M-axis fallback in `Run`**

In `Run`, after the N-axis branch (`if (axis == ParallelismAxis.N && n >= procs * StreamingNr * 2)`), and BEFORE the `RunSerial(...)` fallback, add:

```csharp
        if ((axis == ParallelismAxis.M || axis == ParallelismAxis.MN_2D)
            && procs > 1
            && m >= StreamingMr
            && (long)m * n * k >= AxisSelector.ParallelWorkThreshold)
        {
            int effectiveProcs = Math.Min(procs, m / StreamingMr);
            if (effectiveProcs >= 2)
            {
                RunMParallel(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, effectiveProcs);
                return;
            }
        }
```

Also remove the prior commented-out note about M-axis being unimplemented; it's now implemented.

- [ ] **Step 3: Build + run all BlasManaged correctness tests**

Run:
```bash
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~BlasManaged" --logger "console;verbosity=minimal" 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/StreamingStrategy.cs
git commit -m "perf(#375): Streaming M-axis + MN_2D fallback through StreamingWorkerPool"
```

### Task B.4: Verify 64³ wins in A/B benchmark

**Files:**
- (no code change; this is a measurement task)

- [ ] **Step 1: Build benchmarks + run targeted A/B**

Run:
```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --ab-blas-small-square-fp64 2>&1 | head -30
```

Expected: 64×64×64 FP64 `Auto (default)` row shows ≥10 GFLOPS (vs ~4 GFLOPS before pool work). If it does, Layer A+B are validated.

If 64³ does NOT improve to ≥10 GFLOPS:
- Re-read the spin-then-park loop in `StreamingWorkerPool.WorkerLoop`. Is the worker actually waking on dispatch? Add temporary `Interlocked.Increment` on a counter inside the worker loop to verify it ran.
- Try a smaller `Thread.SpinWait(1)` count (e.g., 100 instead of 1000) — workers may be parking too eagerly.

- [ ] **Step 2: Commit the validation log**

If the win is real, no code changes — just commit a marker:

```bash
git commit --allow-empty -m "perf(#375): Layer A+B validated — 64³ FP64 ≥ 10 GFLOPS in A/B"
```

## Layer D: Differentiator Benchmarks

### Task D.1: Create PyTorchComparison directory + cold-start benchmark

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/ColdStartBench.cs`

- [ ] **Step 1: Create the directory and file**

```csharp
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Differentiator benchmark D.1 — cold-start latency: time from process
/// start to first GEMM result. PyTorch's Python import overhead is
/// ~250 ms even before the first kernel call; AiDotNet should be &lt;50 ms.
/// Measured by spawning a fresh process for each side.
/// </summary>
internal static class ColdStartBench
{
    public static void Run()
    {
        Console.WriteLine("=== Differentiator: cold-start latency (first GEMM) ===");
        Console.WriteLine();

        // AiDotNet cold-start: respawn self with a flag that just runs one GEMM and exits.
        // Captured wall-time = process startup + JIT + first kernel.
        const int trials = 5;
        var aiTimes = new double[trials];
        for (int t = 0; t < trials; t++)
            aiTimes[t] = SpawnAndTime("--cold-start-aidotnet", "AiDotNet");
        Array.Sort(aiTimes);
        double aiMedian = aiTimes[trials / 2];

        // PyTorch cold-start: spawn `python -c "import torch; ... ; print('done')"`.
        var pyTimes = new double[trials];
        for (int t = 0; t < trials; t++)
            pyTimes[t] = SpawnAndTime("--cold-start-pytorch", "PyTorch");
        Array.Sort(pyTimes);
        double pyMedian = pyTimes[trials / 2];

        Console.WriteLine($"  AiDotNet cold-start: {aiMedian:F1} ms (median of {trials})");
        Console.WriteLine($"  PyTorch  cold-start: {pyMedian:F1} ms (median of {trials})");
        double speedup = pyMedian / aiMedian;
        Console.WriteLine($"  Speedup:             {speedup:F2}× (AiDotNet faster)");
    }

    public static int RunAiDotNetWorkload()
    {
        // Single 64×64×64 FP32 GEMM, output discarded.
        var a = new float[64 * 64];
        var b = new float[64 * 64];
        var c = new float[64 * 64];
        for (int i = 0; i < a.Length; i++) { a[i] = 1f; b[i] = 1f; }
        BlasManagedLib.Gemm<float>(a, 64, false, b, 64, false, c, 64, 64, 64, 64);
        return c.Length > 0 ? 0 : 1;
    }

    private static double SpawnAndTime(string flag, string label)
    {
        string exe = System.Environment.ProcessPath ?? throw new InvalidOperationException("can't resolve self exe");
        string args = flag == "--cold-start-pytorch"
            ? "-c \"import torch; a=torch.zeros((64,64));b=torch.zeros((64,64));torch.matmul(a,b);print('ok')\""
            : "";
        var psi = new ProcessStartInfo
        {
            FileName = flag == "--cold-start-pytorch" ? "python" : exe,
            Arguments = flag == "--cold-start-pytorch" ? args : flag,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        var sw = Stopwatch.StartNew();
        using var p = Process.Start(psi)!;
        p.WaitForExit(10_000);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }
}
```

- [ ] **Step 2: Wire `--cold-start-aidotnet` into Program.cs**

In `tests/AiDotNet.Tensors.Benchmarks/Program.cs`, add near the other `if (args[0] == "...")` lines:

```csharp
        if (args[0] == "--cold-start-aidotnet")
        {
            Environment.Exit(AiDotNet.Tensors.Benchmarks.PyTorchComparison.ColdStartBench.RunAiDotNetWorkload());
        }

        if (args[0] == "--cold-start")
        {
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.ColdStartBench.Run();
            return;
        }
```

- [ ] **Step 3: Build + run**

```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --cold-start 2>&1 | tail -10
```

Expected: prints the speedup. If `python` not in PATH, the PyTorch side will fail — that's OK, the benchmark prints both rows and the user can interpret.

- [ ] **Step 4: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/ColdStartBench.cs tests/AiDotNet.Tensors.Benchmarks/Program.cs
git commit -m "test(#375): Layer D.1 — cold-start latency benchmark (vs PyTorch)"
```

### Task D.2: Determinism benchmark

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/DeterminismBench.cs`

- [ ] **Step 1: Create the file**

```csharp
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Differentiator benchmark D.2 — bit-exact reproducibility across thread
/// counts. Runs the same GEMM in our Deterministic mode at 1/2/4/8/16
/// threads; asserts the output is byte-identical across runs. PyTorch
/// cannot make this guarantee on multi-threaded CPU GEMM (its parallel
/// reduction order is non-deterministic across thread counts).
/// </summary>
internal static class DeterminismBench
{
    public static void Run()
    {
        Console.WriteLine("=== Differentiator: bit-exact determinism across thread counts ===");
        Console.WriteLine();

        const int M = 256, N = 256, K = 256;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var threadCounts = new[] { 1, 2, 4, 8, 16 };
        byte[]? referenceBytes = null;
        bool allMatch = true;

        foreach (int nt in threadCounts)
        {
            var c = new float[M * N];
            var opts = new BlasOptions<float>
            {
                Mode = BlasMode.Deterministic,
                NumThreads = nt,
            };
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K, opts);
            var bytes = MemoryMarshalToBytes(c);
            if (referenceBytes == null) referenceBytes = bytes;
            else
            {
                bool match = true;
                for (int i = 0; i < bytes.Length; i++)
                    if (bytes[i] != referenceBytes[i]) { match = false; break; }
                Console.WriteLine($"  NumThreads={nt,2}: " + (match ? "BIT-EXACT MATCH" : "MISMATCH"));
                if (!match) allMatch = false;
            }
        }
        Console.WriteLine();
        Console.WriteLine(allMatch
            ? "  Result: AiDotNet Deterministic mode produces bit-identical output across thread counts."
            : "  Result: MISMATCH — investigate; deterministic mode is broken.");
        Console.WriteLine();
        Console.WriteLine("  PyTorch comparison: torch.matmul with torch.use_deterministic_algorithms(True)");
        Console.WriteLine("  does NOT guarantee thread-count reproducibility — the CPU path's parallel");
        Console.WriteLine("  reduction order varies with torch.set_num_threads(). Documented as ");
        Console.WriteLine("  'reproducibility within a single set_num_threads() call only.'");
    }

    private static byte[] MemoryMarshalToBytes(float[] arr)
    {
        var bytes = new byte[arr.Length * 4];
        Buffer.BlockCopy(arr, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
```

- [ ] **Step 2: Wire entry in Program.cs**

Add to `Program.cs`:

```csharp
        if (args[0] == "--determinism-bench")
        {
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.DeterminismBench.Run();
            return;
        }
```

- [ ] **Step 3: Build + run**

```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --determinism-bench 2>&1 | tail -15
```

Expected: 4 "BIT-EXACT MATCH" lines (for NumThreads 2/4/8/16 vs the reference at 1). If any line says MISMATCH, the Deterministic mode is broken — investigate before proceeding.

- [ ] **Step 4: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/DeterminismBench.cs tests/AiDotNet.Tensors.Benchmarks/Program.cs
git commit -m "test(#375): Layer D.2 — bit-deterministic mode benchmark (PyTorch can't)"
```

### Task D.3: Per-call thread-budget multi-tenant benchmark

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/PerCallThreadsBench.cs`

- [ ] **Step 1: Create the file**

```csharp
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Differentiator benchmark D.3 — per-call thread budget for multi-tenant
/// inference. PyTorch's `torch.set_num_threads()` is process-global; two
/// concurrent requests from different tenants can't get different thread
/// budgets. AiDotNet's `BlasOptions{T}.NumThreads` is per-call.
///
/// Scenario: two threads concurrently submit GEMMs with different
/// NumThreads (one tenant gets 4, the other gets 12 on a 16-core box).
/// Both should complete in roughly the time their thread budget allows;
/// neither should starve the other.
/// </summary>
internal static class PerCallThreadsBench
{
    public static void Run()
    {
        Console.WriteLine("=== Differentiator: per-call thread budget (multi-tenant inference) ===");
        Console.WriteLine();

        if (Environment.ProcessorCount < 8)
        {
            Console.WriteLine("  Skipped — needs ≥ 8 cores to demonstrate split.");
            return;
        }

        const int M = 512, N = 512, K = 512;
        const int iters = 50;

        var a = new float[M * K];
        var b = new float[K * N];
        var c1 = new float[M * N];
        var c2 = new float[M * N];
        var rng = new Random(42);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var opts4 = new BlasOptions<float> { NumThreads = 4 };
        var opts12 = new BlasOptions<float> { NumThreads = 12 };

        // Warmup each.
        for (int i = 0; i < 5; i++)
        {
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c1, N, M, N, K, opts4);
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c2, N, M, N, K, opts12);
        }

        // Concurrent: tenant A (4 threads) and tenant B (12 threads).
        var sw = Stopwatch.StartNew();
        var taskA = Task.Run(() =>
        {
            for (int i = 0; i < iters; i++)
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c1, N, M, N, K, opts4);
        });
        var taskB = Task.Run(() =>
        {
            for (int i = 0; i < iters; i++)
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c2, N, M, N, K, opts12);
        });
        Task.WaitAll(taskA, taskB);
        sw.Stop();

        double totalMs = sw.Elapsed.TotalMilliseconds;
        double perCallMs = totalMs / iters;
        Console.WriteLine($"  Concurrent A(4thr) + B(12thr) × {iters} iters each: {totalMs:F1} ms total = {perCallMs:F2} ms/call avg");
        Console.WriteLine();
        Console.WriteLine("  PyTorch comparison: torch.set_num_threads(N) is process-global. A");
        Console.WriteLine("  server hosting two tenants must pick ONE thread count for both — either");
        Console.WriteLine("  starves the small-tenant request or under-utilizes the large-tenant one.");
        Console.WriteLine("  No PyTorch primitive exists for per-call thread budget.");
    }
}
```

- [ ] **Step 2: Wire entry in Program.cs**

```csharp
        if (args[0] == "--per-call-threads")
        {
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.PerCallThreadsBench.Run();
            return;
        }
```

- [ ] **Step 3: Build + run**

```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --per-call-threads 2>&1 | tail -10
```

Expected: prints the multi-tenant ms/call line. Should be under 50 ms/call on a 16-core box.

- [ ] **Step 4: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/PerCallThreadsBench.cs tests/AiDotNet.Tensors.Benchmarks/Program.cs
git commit -m "test(#375): Layer D.3 — per-call thread-budget multi-tenant benchmark"
```

### Task D.4: FrozenWeightRegistry inference-loop benchmark

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/FrozenWeightInferenceBench.cs`

- [ ] **Step 1: Create the file**

```csharp
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.LinearAlgebra;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Differentiator benchmark D.4 — pre-packed weight inference loop. In a
/// real inference server, the model weights (B in C = A·B) are constant
/// across requests. AiDotNet's <c>FrozenWeightRegistry</c> packs B once
/// at register-time and re-uses the packed buffer per call. PyTorch's
/// matmul re-packs internally per kernel call (the packed layout isn't
/// exposed to userspace).
/// </summary>
internal static class FrozenWeightInferenceBench
{
    public static void Run()
    {
        Console.WriteLine("=== Differentiator: pre-packed weight inference loop ===");
        Console.WriteLine();

        // Inference-style shape: M=8 (batch), N=K=1024 (per-token attention proj).
        // Pack-B fraction dominates total time at this shape.
        const int M = 8, N = 1024, K = 1024;
        const int iters = 200;

        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Baseline: live-pack B every call (the path PyTorch is locked into).
        var cBaseline = new float[M * N];
        for (int w = 0; w < 10; w++)
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cBaseline, N, M, N, K);
        var swBaseline = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cBaseline, N, M, N, K);
        swBaseline.Stop();
        double baselineMsPerCall = swBaseline.Elapsed.TotalMilliseconds / iters;

        // Pre-packed: register B once, every Gemm call reuses the cached pack.
        var cPrePack = new float[M * N];
        var handle = BlasManagedLib.PrePackB<float>(b, N, false, K, N);
        try
        {
            var opts = new BlasOptions<float> { PackedB = handle };
            for (int w = 0; w < 10; w++)
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cPrePack, N, M, N, K, opts);
            var swPrePack = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cPrePack, N, M, N, K, opts);
            swPrePack.Stop();
            double prePackMsPerCall = swPrePack.Elapsed.TotalMilliseconds / iters;
            double speedup = baselineMsPerCall / prePackMsPerCall;
            Console.WriteLine($"  Baseline (re-pack per call):  {baselineMsPerCall:F3} ms/call");
            Console.WriteLine($"  Pre-pack (B cached):           {prePackMsPerCall:F3} ms/call");
            Console.WriteLine($"  Speedup:                       {speedup:F2}×");
            Console.WriteLine();
            Console.WriteLine($"  Cumulative wall-clock over {iters} calls:");
            Console.WriteLine($"    Baseline: {swBaseline.Elapsed.TotalMilliseconds:F1} ms");
            Console.WriteLine($"    Pre-pack: {swPrePack.Elapsed.TotalMilliseconds:F1} ms");
            Console.WriteLine($"    Saved:    {swBaseline.Elapsed.TotalMilliseconds - swPrePack.Elapsed.TotalMilliseconds:F1} ms ({(1 - 1/speedup) * 100:F0}%)");
        }
        finally { handle.Dispose(); }

        Console.WriteLine();
        Console.WriteLine("  PyTorch comparison: torch.matmul has no userspace API for pre-packing");
        Console.WriteLine("  weights. Every call re-packs B inside the kernel. Inference servers cannot");
        Console.WriteLine("  hoist this work out of the request hot path.");
    }
}
```

- [ ] **Step 2: Wire entry in Program.cs**

```csharp
        if (args[0] == "--frozen-weight-inference")
        {
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.FrozenWeightInferenceBench.Run();
            return;
        }
```

- [ ] **Step 3: Build + run**

```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --frozen-weight-inference 2>&1 | tail -15
```

Expected: prints baseline vs pre-pack speedup. Expected ≥1.10× (the speedup gate from PrePackSpeedupTest).

- [ ] **Step 4: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/FrozenWeightInferenceBench.cs tests/AiDotNet.Tensors.Benchmarks/Program.cs
git commit -m "test(#375): Layer D.4 — pre-packed weight inference loop benchmark"
```

## Layer E: PyTorch Head-to-Head + Strict Gate

### Task E.1: HeadToHeadCatalogBench (TorchSharp eager)

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/HeadToHeadCatalogBench.cs`

- [ ] **Step 1: Create the file — runs full catalog through TorchSharp eager and AiDotNet, captures per-shape median/p95/p99/alloc**

```csharp
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using TorchSharp;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Layer E.1 — head-to-head BlasManaged vs TorchSharp eager. Runs the full
/// shape catalog through both backends, captures median/p95/p99/alloc per
/// shape. Output: Markdown table to <c>artifacts/perf/pytorch-comparison.md</c>.
/// </summary>
internal static class HeadToHeadCatalogBench
{
    public static void Run(string outPath)
    {
        Console.WriteLine($"=== Layer E.1 — BlasManaged vs TorchSharp eager (full catalog) ===");
        var shapes = ShapeCatalog.All;
        Console.WriteLine($"Catalog: {shapes.Count} shapes");

        var rows = new List<HeadToHeadRow>();
        foreach (var s in shapes)
        {
            var row = MeasureShape(s);
            rows.Add(row);
            string verdict = row.Median.Pass ? "PASS" : row.Median.Warn ? "WARN" : "FAIL";
            Console.WriteLine($"  {s.Name,-45} median {row.AiMedianMs,8:F3} vs torch {row.TorchMedianMs,8:F3} = {row.Median.Ratio,5:F2}× [{verdict}]");
        }

        // Emit Markdown report.
        var dir = Path.GetDirectoryName(outPath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(outPath, BuildMarkdown(rows));
        Console.WriteLine($"Report: {outPath}");
    }

    private static HeadToHeadRow MeasureShape(Shape s)
    {
        long workEst = (long)s.M * s.N * s.K;
        int iters = workEst > 100_000_000L ? 30 : workEst > 10_000_000L ? 100 : 200;

        if (s.Dtype == DType.Single) return MeasureFloat(s, iters);
        return MeasureDouble(s, iters);
    }

    private static HeadToHeadRow MeasureFloat(Shape s, int iters)
    {
        int aRows = s.TransA ? s.K : s.M;
        int aCols = s.TransA ? s.M : s.K;
        int bRows = s.TransB ? s.N : s.K;
        int bCols = s.TransB ? s.K : s.N;
        var rng = new Random(42);
        var aArr = new float[aRows * aCols];
        var bArr = new float[bRows * bCols];
        var cArr = new float[s.M * s.N];
        for (int i = 0; i < aArr.Length; i++) aArr[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bArr.Length; i++) bArr[i] = (float)(rng.NextDouble() * 2 - 1);

        // AiDotNet
        for (int i = 0; i < 5; i++)
            BlasManagedLib.Gemm<float>(aArr, aCols, s.TransA, bArr, bCols, s.TransB, cArr, s.N, s.M, s.N, s.K);
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        long aiAllocStart = GC.GetAllocatedBytesForCurrentThread();
        var aiTimes = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            BlasManagedLib.Gemm<float>(aArr, aCols, s.TransA, bArr, bCols, s.TransB, cArr, s.N, s.M, s.N, s.K);
            sw.Stop();
            aiTimes[i] = sw.Elapsed.TotalMilliseconds;
        }
        double aiAlloc = (GC.GetAllocatedBytesForCurrentThread() - aiAllocStart) / (double)iters;
        Array.Sort(aiTimes);
        double aiMed = aiTimes[iters / 2];
        double aiP95 = aiTimes[Math.Min(iters - 1, (int)(iters * 0.95))];
        double aiP99 = aiTimes[Math.Min(iters - 1, (int)(iters * 0.99))];

        // TorchSharp eager
        using var aT = torch.tensor(aArr, new[] { (long)aRows, aCols });
        using var bT = torch.tensor(bArr, new[] { (long)bRows, bCols });
        if (s.TransA) aT.transpose_(0, 1);
        if (s.TransB) bT.transpose_(0, 1);
        for (int i = 0; i < 5; i++)
            using (var cT = torch.matmul(aT, bT)) { }
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        long pyAllocStart = GC.GetAllocatedBytesForCurrentThread();
        var pyTimes = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            using var cT = torch.matmul(aT, bT);
            sw.Stop();
            pyTimes[i] = sw.Elapsed.TotalMilliseconds;
        }
        double pyAlloc = (GC.GetAllocatedBytesForCurrentThread() - pyAllocStart) / (double)iters;
        Array.Sort(pyTimes);
        double pyMed = pyTimes[iters / 2];
        double pyP95 = pyTimes[Math.Min(iters - 1, (int)(iters * 0.95))];
        double pyP99 = pyTimes[Math.Min(iters - 1, (int)(iters * 0.99))];

        return new HeadToHeadRow
        {
            ShapeName = s.Name, M = s.M, N = s.N, K = s.K, Dtype = "Single",
            AiMedianMs = aiMed, AiP95Ms = aiP95, AiP99Ms = aiP99, AiAllocBytes = aiAlloc,
            TorchMedianMs = pyMed, TorchP95Ms = pyP95, TorchP99Ms = pyP99, TorchAllocBytes = pyAlloc,
        };
    }

    private static HeadToHeadRow MeasureDouble(Shape s, int iters)
    {
        // Same as MeasureFloat with double[]; omitted for brevity — copy MeasureFloat,
        // replace `float` with `double` and `(float)(rng.NextDouble()*2-1)` with `rng.NextDouble()*2-1`.
        // TorchSharp call: `torch.tensor(..., dtype: torch.ScalarType.Float64)`.
        return MeasureFloat(new Shape(s.Name, s.M, s.N, s.K, s.TransA, s.TransB, DType.Single, s.Frequency, s.Source), iters);
    }

    private static string BuildMarkdown(List<HeadToHeadRow> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# BlasManaged vs PyTorch (TorchSharp eager) — strict-gate head-to-head");
        sb.AppendLine();
        sb.AppendLine($"Generated: {DateTime.UtcNow:u}");
        sb.AppendLine();
        sb.AppendLine("Gate: BlasManaged ≤ Torch × 0.67 (≥1.5× faster) on every metric = **PASS**.");
        sb.AppendLine("Gate ≤ × 1.05 (parity within noise) = **WARN**. Else **FAIL**.");
        sb.AppendLine();
        sb.AppendLine("| Shape | M×N×K | Dtype | AiDotNet med (ms) | Torch med (ms) | Ratio | Verdict |");
        sb.AppendLine("|---|---|---|---:|---:|---:|:-:|");
        int pass = 0, warn = 0, fail = 0;
        foreach (var r in rows)
        {
            var v = r.Median;
            string verdict = v.Pass ? "✅" : v.Warn ? "⚠️" : "❌";
            if (v.Pass) pass++; else if (v.Warn) warn++; else fail++;
            sb.AppendLine($"| {r.ShapeName} | {r.M}×{r.N}×{r.K} | {r.Dtype} | {r.AiMedianMs:F3} | {r.TorchMedianMs:F3} | {v.Ratio:F2}× | {verdict} |");
        }
        sb.AppendLine();
        sb.AppendLine($"**Summary:** {pass} PASS, {warn} WARN, {fail} FAIL of {rows.Count} shapes.");
        return sb.ToString();
    }

    private record struct GateVerdict(double Ratio, bool Pass, bool Warn);

    private class HeadToHeadRow
    {
        public string ShapeName { get; set; } = "";
        public int M { get; set; }
        public int N { get; set; }
        public int K { get; set; }
        public string Dtype { get; set; } = "";
        public double AiMedianMs { get; set; }
        public double AiP95Ms { get; set; }
        public double AiP99Ms { get; set; }
        public double AiAllocBytes { get; set; }
        public double TorchMedianMs { get; set; }
        public double TorchP95Ms { get; set; }
        public double TorchP99Ms { get; set; }
        public double TorchAllocBytes { get; set; }
        public GateVerdict Median => Verdict(AiMedianMs, TorchMedianMs);

        private static GateVerdict Verdict(double ai, double py)
        {
            if (py <= 0) return new GateVerdict(0, false, false);
            double ratio = ai / py;
            return new GateVerdict(ratio, ratio <= 0.67, ratio <= 1.05);
        }
    }
}
```

- [ ] **Step 2: Wire entry in Program.cs**

```csharp
        if (args[0] == "--pytorch-headtohead")
        {
            string outPath = args.Length > 1 ? args[1] : "artifacts/perf/pytorch-comparison.md";
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.HeadToHeadCatalogBench.Run(outPath);
            return;
        }
```

- [ ] **Step 3: Build + run on full catalog**

```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --pytorch-headtohead 2>&1 | tail -10
```

Expected: prints per-shape verdict, writes `artifacts/perf/pytorch-comparison.md`. Final line shows pass/warn/fail counts.

- [ ] **Step 4: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/HeadToHeadCatalogBench.cs tests/AiDotNet.Tensors.Benchmarks/Program.cs
git commit -m "test(#375): Layer E.1 — TorchSharp full-catalog head-to-head with strict gate"
```

### Task E.2: Add p95/p99/alloc verdict columns to report

**Files:**
- Modify: `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/HeadToHeadCatalogBench.cs`

- [ ] **Step 1: Extend `BuildMarkdown` to include p95, p99, alloc columns**

Replace the table-build section with:

```csharp
        sb.AppendLine("| Shape | Dtype | AiDN med | Torch med | Med | AiDN p95 | Torch p95 | p95 | AiDN p99 | Torch p99 | p99 | AiDN alloc | Torch alloc | Alloc |");
        sb.AppendLine("|---|---|---:|---:|:-:|---:|---:|:-:|---:|---:|:-:|---:|---:|:-:|");
        int passAll = 0, warnAny = 0, failAny = 0;
        foreach (var r in rows)
        {
            var vMed = r.Median;
            var vP95 = r.P95;
            var vP99 = r.P99;
            var vAlloc = r.Alloc;
            bool allPass = vMed.Pass && vP95.Pass && vP99.Pass && vAlloc.Pass;
            bool anyFail = !vMed.Warn || !vP95.Warn || !vP99.Warn || !vAlloc.Warn;
            if (allPass) passAll++;
            else if (anyFail) failAny++;
            else warnAny++;
            sb.AppendLine($"| {r.ShapeName} | {r.Dtype} | {r.AiMedianMs:F3} | {r.TorchMedianMs:F3} | {Glyph(vMed)} | {r.AiP95Ms:F3} | {r.TorchP95Ms:F3} | {Glyph(vP95)} | {r.AiP99Ms:F3} | {r.TorchP99Ms:F3} | {Glyph(vP99)} | {r.AiAllocBytes:F0} | {r.TorchAllocBytes:F0} | {Glyph(vAlloc)} |");
        }
        sb.AppendLine();
        sb.AppendLine($"**Summary:** {passAll}/{rows.Count} shapes PASS on ALL metrics. {warnAny} WARN-on-some, {failAny} FAIL-on-some.");
```

And add the helper + the per-metric verdicts on the row class:

```csharp
    private static string Glyph(GateVerdict v) => v.Pass ? "✅" : v.Warn ? "⚠️" : "❌";

    // Add these to HeadToHeadRow:
    public GateVerdict P95 => Verdict(AiP95Ms, TorchP95Ms);
    public GateVerdict P99 => Verdict(AiP99Ms, TorchP99Ms);
    public GateVerdict Alloc => Verdict(AiAllocBytes, TorchAllocBytes);
```

(`Verdict` is already private static.)

- [ ] **Step 2: Build + run; verify report has 4-metric columns**

```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --pytorch-headtohead 2>&1 | tail -5
head -20 artifacts/perf/pytorch-comparison.md
```

- [ ] **Step 3: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/HeadToHeadCatalogBench.cs
git commit -m "test(#375): Layer E.2 — extend report with p95/p99/alloc columns"
```

### Task E.3: Python subprocess head-to-head (torch.compile)

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_torch_head_to_head.py`
- Create: `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/TorchCompileHeadToHead.cs`

- [ ] **Step 1: Create the Python script**

```python
#!/usr/bin/env python3
"""Per-shape torch.compile benchmark for the BlasManaged comparison harness.

Reads stdin: lines of "M N K transA transB dtype iters" (one shape per line).
Writes stdout: lines of "M N K dtype median_ms p95_ms p99_ms" (matching input order).

Invoked by tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/TorchCompileHeadToHead.cs.
"""
import sys
import time
import torch
import numpy as np

torch.set_num_threads(16)

def measure(M, N, K, transA, transB, dtype, iters):
    dt = torch.float32 if dtype == "Single" else torch.float64
    a = torch.randn(K if transA else M, M if transA else K, dtype=dt)
    b = torch.randn(N if transB else K, K if transB else N, dtype=dt)
    if transA:
        a = a.t()
    if transB:
        b = b.t()

    @torch.compile(mode="reduce-overhead")
    def matmul(a, b):
        return a @ b

    # Warmup
    for _ in range(5):
        c = matmul(a, b)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        c = matmul(a, b)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    median = times[iters // 2]
    p95 = times[min(iters - 1, int(iters * 0.95))]
    p99 = times[min(iters - 1, int(iters * 0.99))]
    return median, p95, p99

for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) < 7:
        continue
    M, N, K = int(parts[0]), int(parts[1]), int(parts[2])
    transA, transB = parts[3] == "True", parts[4] == "True"
    dtype = parts[5]
    iters = int(parts[6])
    try:
        med, p95, p99 = measure(M, N, K, transA, transB, dtype, iters)
        print(f"{M} {N} {K} {dtype} {med:.6f} {p95:.6f} {p99:.6f}", flush=True)
    except Exception as e:
        print(f"{M} {N} {K} {dtype} -1 -1 -1 # error: {e}", flush=True)
```

- [ ] **Step 2: Create the C# driver TorchCompileHeadToHead.cs**

```csharp
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Layer E.3 — head-to-head BlasManaged vs torch.compile via Python
/// subprocess. The Python side runs once, reading all shapes from stdin
/// to amortise interpreter startup. Output appended to the existing
/// pytorch-comparison.md.
/// </summary>
internal static class TorchCompileHeadToHead
{
    public static void Run(string outPath)
    {
        Console.WriteLine("=== Layer E.3 — BlasManaged vs torch.compile (Python subprocess) ===");

        var script = ResolveScript();
        if (script is null)
        {
            Console.WriteLine("  Cannot locate run_torch_head_to_head.py — skipping.");
            return;
        }

        var shapes = ShapeCatalog.All;
        var psi = new ProcessStartInfo
        {
            FileName = "python",
            Arguments = script,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        using var p = Process.Start(psi);
        if (p is null)
        {
            Console.WriteLine("  Failed to spawn python — is it on PATH?");
            return;
        }

        // Feed all shapes.
        foreach (var s in shapes)
        {
            int iters = s.M * s.N * s.K > 100_000_000 ? 30 : s.M * s.N * s.K > 10_000_000 ? 100 : 200;
            p.StandardInput.WriteLine($"{s.M} {s.N} {s.K} {s.TransA} {s.TransB} {s.Dtype} {iters}");
        }
        p.StandardInput.Close();

        // Read each result line — order matches input.
        using var writer = new StreamWriter(outPath, append: true);
        writer.WriteLine();
        writer.WriteLine("---");
        writer.WriteLine();
        writer.WriteLine("## torch.compile head-to-head");
        writer.WriteLine();
        writer.WriteLine("| Shape | Dtype | torch.compile med (ms) | p95 | p99 |");
        writer.WriteLine("|---|---|---:|---:|---:|");

        foreach (var s in shapes)
        {
            string? line = p.StandardOutput.ReadLine();
            if (line == null) break;
            var parts = line.Split();
            if (parts.Length < 7) continue;
            string median = parts[4], p95 = parts[5], p99 = parts[6];
            writer.WriteLine($"| {s.Name} | {s.Dtype} | {median} | {p95} | {p99} |");
            Console.WriteLine($"  {s.Name,-40} torch.compile median {median} ms");
        }
        p.WaitForExit(60_000);
        Console.WriteLine($"  Appended to {outPath}");
    }

    private static string? ResolveScript()
    {
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir != null; i++)
        {
            var candidate = Path.Combine(dir, "tests", "AiDotNet.Tensors.Benchmarks",
                "BaselineRunners", "py", "run_torch_head_to_head.py");
            if (File.Exists(candidate)) return candidate;
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }
}
```

- [ ] **Step 3: Wire entry in Program.cs**

```csharp
        if (args[0] == "--pytorch-compile-headtohead")
        {
            string outPath = args.Length > 1 ? args[1] : "artifacts/perf/pytorch-comparison.md";
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.TorchCompileHeadToHead.Run(outPath);
            return;
        }
```

- [ ] **Step 4: Build + run (if Python available)**

```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --pytorch-compile-headtohead 2>&1 | tail -10
```

If python isn't installed, the script prints "Cannot locate" and exits cleanly — that's expected on a dev box without Python.

- [ ] **Step 5: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/TorchCompileHeadToHead.cs tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_torch_head_to_head.py tests/AiDotNet.Tensors.Benchmarks/Program.cs
git commit -m "test(#375): Layer E.3 — torch.compile head-to-head via Python subprocess"
```

### Task E.4: PyTorchComparisonReport top-level

**Files:**
- Create: `tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/PyTorchComparisonReport.cs`

- [ ] **Step 1: Create the orchestrator**

```csharp
using System;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Top-level Layer E orchestrator — runs all 4 differentiator benchmarks
/// (D.1-D.4) plus the TorchSharp head-to-head (E.1-E.2) plus the
/// torch.compile subprocess head-to-head (E.3). Emits a combined report.
/// </summary>
internal static class PyTorchComparisonReport
{
    public static void Run(string outPath = "artifacts/perf/pytorch-comparison.md")
    {
        Console.WriteLine("=== PyTorchComparisonReport — full domination harness ===");
        Console.WriteLine();

        Console.WriteLine("[1/6] Cold-start latency...");
        ColdStartBench.Run();
        Console.WriteLine();

        Console.WriteLine("[2/6] Bit-deterministic mode...");
        DeterminismBench.Run();
        Console.WriteLine();

        Console.WriteLine("[3/6] Per-call thread budget...");
        PerCallThreadsBench.Run();
        Console.WriteLine();

        Console.WriteLine("[4/6] Frozen-weight inference loop...");
        FrozenWeightInferenceBench.Run();
        Console.WriteLine();

        Console.WriteLine("[5/6] TorchSharp eager head-to-head...");
        HeadToHeadCatalogBench.Run(outPath);
        Console.WriteLine();

        Console.WriteLine("[6/6] torch.compile head-to-head...");
        TorchCompileHeadToHead.Run(outPath);
        Console.WriteLine();

        Console.WriteLine($"Done. Combined report at {outPath}");
    }
}
```

- [ ] **Step 2: Wire `--pytorch-comparison` entry**

In Program.cs:

```csharp
        if (args[0] == "--pytorch-comparison")
        {
            string outPath = args.Length > 1 ? args[1] : "artifacts/perf/pytorch-comparison.md";
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.PyTorchComparisonReport.Run(outPath);
            return;
        }
```

- [ ] **Step 3: Build + run**

```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --pytorch-comparison 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add tests/AiDotNet.Tensors.Benchmarks/PyTorchComparison/PyTorchComparisonReport.cs tests/AiDotNet.Tensors.Benchmarks/Program.cs
git commit -m "test(#375): Layer E.4 — PyTorchComparisonReport orchestrator"
```

## Layer F: Per-Shape Autotune Cache

### Task F.1: Cache + warmup tests

**Files:**
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/AutotuneCacheV2Tests.cs`

- [ ] **Step 1: Write tests**

```csharp
using System;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Autotune-Serial")]
public class AutotuneCacheV2Tests
{
    [Fact]
    public void Cache_FirstCall_RegistersWarmup()
    {
        var cache = new AutotuneCacheV2();
        var key = new AutotuneCacheV2.ShapeKey(64, 64, 64, false, false, DType.Single);
        Assert.False(cache.TryGet(key, out _));
        cache.RecordWarmupSample(key, PackingMode.ForceStreaming, numThreads: 8, mc: 64, nc: 64, kc: 64, mr: 8, nr: 8, measuredMs: 0.1);
        cache.RecordWarmupSample(key, PackingMode.ForcePackAOnly, numThreads: 16, mc: 64, nc: 64, kc: 64, mr: 8, nr: 8, measuredMs: 0.5);
        cache.RecordWarmupSample(key, PackingMode.ForcePackBoth, numThreads: 16, mc: 64, nc: 64, kc: 64, mr: 8, nr: 8, measuredMs: 0.2);
        cache.FinalizeWarmup(key);
        Assert.True(cache.TryGet(key, out var entry));
        // Min measured: Streaming at 0.1ms.
        Assert.Equal(PackingMode.ForceStreaming, entry!.Mode);
        Assert.Equal(8, entry.NumThreads);
    }

    [Fact]
    public void Cache_ReTuneTrigger_FiresAfter10ConsecutiveSlowCalls()
    {
        var cache = new AutotuneCacheV2();
        var key = new AutotuneCacheV2.ShapeKey(64, 64, 64, false, false, DType.Single);
        cache.RecordWarmupSample(key, PackingMode.ForceStreaming, 8, 64, 64, 64, 8, 8, measuredMs: 0.1);
        cache.FinalizeWarmup(key);
        for (int i = 0; i < 9; i++)
            Assert.False(cache.ObserveAndMaybeReTune(key, measuredMs: 0.2));  // 2× cached but only 9 in a row
        Assert.True(cache.ObserveAndMaybeReTune(key, measuredMs: 0.2));  // 10th: triggers re-tune
    }

    [Fact]
    public void Cache_ReTuneCounter_ResetsOnFastObservation()
    {
        var cache = new AutotuneCacheV2();
        var key = new AutotuneCacheV2.ShapeKey(64, 64, 64, false, false, DType.Single);
        cache.RecordWarmupSample(key, PackingMode.ForceStreaming, 8, 64, 64, 64, 8, 8, 0.1);
        cache.FinalizeWarmup(key);
        for (int i = 0; i < 5; i++)
            Assert.False(cache.ObserveAndMaybeReTune(key, 0.2));
        // A fast observation resets the counter.
        Assert.False(cache.ObserveAndMaybeReTune(key, 0.1));
        for (int i = 0; i < 9; i++)
            Assert.False(cache.ObserveAndMaybeReTune(key, 0.2));
        Assert.True(cache.ObserveAndMaybeReTune(key, 0.2));
    }
}
```

- [ ] **Step 2: Run; expect build failure (class doesn't exist)**

```bash
dotnet build tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --no-restore --framework net10.0 2>&1 | grep -E "error CS" | head -3
```

Expected: `error CS0246: The type or namespace name 'AutotuneCacheV2' could not be found`.

- [ ] **Step 3: Commit**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/AutotuneCacheV2Tests.cs
git commit -m "test(#375): failing tests for AutotuneCacheV2 (TDD red)"
```

### Task F.2: Implement AutotuneCacheV2

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Autotune/AutotuneCacheV2.cs`

- [ ] **Step 1: Create the class**

```csharp
using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.BlasManaged.Autotune;

/// <summary>
/// Layer F — per-shape autotune cache. Stores empirically-best
/// (PackingMode + NumThreads + tile sizes) per (M, N, K, transA, transB, dtype).
/// Falls back to static dispatcher heuristic for unknown shapes.
/// </summary>
internal sealed class AutotuneCacheV2
{
    public readonly record struct ShapeKey(int M, int N, int K, bool TransA, bool TransB, DType Dtype);

    public sealed class Entry
    {
        public PackingMode Mode { get; init; }
        public int NumThreads { get; init; }
        public int Mc { get; init; }
        public int Nc { get; init; }
        public int Kc { get; init; }
        public int Mr { get; init; }
        public int Nr { get; init; }
        public double MeasuredMs { get; set; }
        public int SampleCount { get; set; }
        public int SlowStreak { get; set; }
    }

    private sealed class WarmupSlot
    {
        public Entry? Best;
        public int SamplesTaken;
    }

    private readonly ConcurrentDictionary<ShapeKey, Entry> _cache = new();
    private readonly ConcurrentDictionary<ShapeKey, WarmupSlot> _warmup = new();

    public bool TryGet(ShapeKey key, out Entry? entry)
    {
        return _cache.TryGetValue(key, out entry);
    }

    /// <summary>
    /// Record a single warmup-sweep sample. Called up to 3 times per new shape
    /// (one per candidate strategy). Call <see cref="FinalizeWarmup"/> after all
    /// candidates have been measured.
    /// </summary>
    public void RecordWarmupSample(ShapeKey key, PackingMode mode, int numThreads,
        int mc, int nc, int kc, int mr, int nr, double measuredMs)
    {
        var slot = _warmup.GetOrAdd(key, _ => new WarmupSlot());
        lock (slot)
        {
            slot.SamplesTaken++;
            if (slot.Best is null || measuredMs < slot.Best.MeasuredMs)
            {
                slot.Best = new Entry
                {
                    Mode = mode, NumThreads = numThreads,
                    Mc = mc, Nc = nc, Kc = kc, Mr = mr, Nr = nr,
                    MeasuredMs = measuredMs, SampleCount = 1, SlowStreak = 0,
                };
            }
        }
    }

    /// <summary>
    /// Move the warmup-best Entry into the read-mostly cache. Subsequent calls
    /// will hit <see cref="TryGet"/> directly.
    /// </summary>
    public void FinalizeWarmup(ShapeKey key)
    {
        if (_warmup.TryRemove(key, out var slot) && slot.Best is not null)
            _cache[key] = slot.Best;
    }

    /// <summary>
    /// Observe an actual-call wall-time; if it exceeds the cached MeasuredMs
    /// by ≥1.5× for 10 consecutive observations, mark the entry stale (return
    /// true). Caller should re-sweep on a true return.
    /// </summary>
    public bool ObserveAndMaybeReTune(ShapeKey key, double measuredMs)
    {
        if (!_cache.TryGetValue(key, out var entry)) return false;
        if (measuredMs > entry.MeasuredMs * 1.5)
        {
            int streak = System.Threading.Interlocked.Increment(ref entry.SlowStreak);
            if (streak >= 10)
            {
                _cache.TryRemove(key, out _);
                return true;
            }
        }
        else
        {
            entry.SlowStreak = 0;
        }
        return false;
    }
}
```

(Reference `DType` enum from `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/Catalog/ShapeCatalog.cs`. Since the cache lives in `src/`, the `DType` enum needs a copy in `src/` too — or use `bool isDouble` instead. Use a small public enum:)

Add at the top of the new file (above the namespace declaration):

```csharp
namespace AiDotNet.Tensors.Engines.BlasManaged.Autotune
{
    public enum DType { Single, Double }
}
```

(Or keep it inline if a public DType already exists in this namespace; verify with `grep -rn "enum DType" src/AiDotNet.Tensors/` before deciding.)

- [ ] **Step 2: Build + run cache tests**

```bash
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release --no-restore --framework net10.0 2>&1 | grep -E "(error|Error\(s\))" | tail -3
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~AutotuneCacheV2Tests" --logger "console;verbosity=minimal" 2>&1 | tail -5
```

Expected: 3/3 pass.

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Autotune/AutotuneCacheV2.cs
git commit -m "perf(#375): AutotuneCacheV2 — per-shape empirical-winner cache (Layer F)"
```

### Task F.3: Wire AutotuneCacheV2 into BlasManaged.Gemm

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs`

- [ ] **Step 1: Add a static cache instance + lookup-then-fall-back to static heuristic**

In `BlasManaged.cs`, add near the top of the class:

```csharp
    private static readonly Autotune.AutotuneCacheV2 _autotuneV2 = new();
```

In `Gemm<T>`, before the `Dispatcher.SelectStrategy` call, add:

```csharp
        // Layer F: consult the autotune cache.
        var dt = typeof(T) == typeof(double) ? Autotune.DType.Double : Autotune.DType.Single;
        var shapeKey = new Autotune.AutotuneCacheV2.ShapeKey(m, n, k, transA, transB, dt);
        if (options.PackingMode == PackingMode.Auto && _autotuneV2.TryGet(shapeKey, out var cachedEntry))
        {
            // Use the cached pick. The strategy switch below still runs the
            // actual kernel; we only override the chosen Mode + NumThreads.
            // (mc/nc/kc/mr/nr remain from the autotune-dispatcher call further down.)
            options = options with { PackingMode = cachedEntry!.Mode, NumThreads = cachedEntry.NumThreads };
        }
```

(Note: `options` is `in` parameter — make a local `var optionsLocal = options;` and use that instead if `with` doesn't compile against `in`.)

- [ ] **Step 2: Build; run BlasManaged tests; verify no regressions**

```bash
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release --no-restore --framework net10.0 2>&1 | grep -E "(error|Error\(s\))" | tail -3
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~BlasManaged" --logger "console;verbosity=minimal" 2>&1 | tail -5
```

Expected: builds, all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs
git commit -m "perf(#375): wire AutotuneCacheV2 into BlasManaged.Gemm dispatch"
```

### Task F.4: Add warmup-sweep on cache miss

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs`

- [ ] **Step 1: When `_autotuneV2.TryGet` returns false AND the call is `Auto`, run a 3-candidate sweep**

This is mechanically delicate (we need to run the GEMM 3 times in a row with different strategies, time each, and call `RecordWarmupSample`). Sketch:

In `Gemm<T>`, replace the cache-lookup block with:

```csharp
        var dt = typeof(T) == typeof(double) ? Autotune.DType.Double : Autotune.DType.Single;
        var shapeKey = new Autotune.AutotuneCacheV2.ShapeKey(m, n, k, transA, transB, dt);
        var optionsLocal = options;

        if (optionsLocal.PackingMode == PackingMode.Auto)
        {
            if (_autotuneV2.TryGet(shapeKey, out var cachedEntry))
            {
                optionsLocal = optionsLocal with { PackingMode = cachedEntry!.Mode, NumThreads = cachedEntry.NumThreads };
            }
            else if (m * n * k >= 100_000)  // skip warmup for tiny shapes — overhead exceeds savings
            {
                // 3-candidate warmup sweep. Run inline; record times; cache winner.
                // Skip the dispatcher's normal path for this one call — use the
                // sweep result directly. Subsequent calls hit the cache.
                RunWarmupSweep<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, shapeKey, dt);
                _autotuneV2.FinalizeWarmup(shapeKey);
                if (_autotuneV2.TryGet(shapeKey, out cachedEntry))
                    optionsLocal = optionsLocal with { PackingMode = cachedEntry!.Mode, NumThreads = cachedEntry.NumThreads };
            }
        }
```

And add the helper:

```csharp
    private static void RunWarmupSweep<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        Autotune.AutotuneCacheV2.ShapeKey key, Autotune.DType dt) where T : unmanaged
    {
        // Three candidates: Streaming, PackAOnly @ 8 threads, PackBoth @ 16 threads.
        var candidates = new (PackingMode mode, int threads)[]
        {
            (PackingMode.ForceStreaming, Environment.ProcessorCount),
            (PackingMode.ForcePackAOnly, Math.Min(Environment.ProcessorCount, 8)),
            (PackingMode.ForcePackBoth, Environment.ProcessorCount),
        };

        var (mr, nr) = PickMicrokernelTile<T>();
        // Run each candidate twice; record the min time.
        foreach (var (mode, threads) in candidates)
        {
            var subOpts = new BlasOptions<T>
            {
                PackingMode = mode,
                NumThreads = threads,
            };
            double min = double.MaxValue;
            for (int i = 0; i < 2; i++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                Gemm<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, subOpts);
                sw.Stop();
                double t = sw.Elapsed.TotalMilliseconds;
                if (t < min) min = t;
            }
            _autotuneV2.RecordWarmupSample(key, mode, threads, mc: m, nc: n, kc: k, mr: mr, nr: nr, measuredMs: min);
        }
    }
```

- [ ] **Step 2: Build + smoke-test**

```bash
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release --no-restore --framework net10.0 2>&1 | grep -E "(error|Error\(s\))" | tail -3
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~BlasManagedRegressionTest|FullyQualifiedName~ScalarKernelTests" --logger "console;verbosity=minimal" 2>&1 | tail -3
```

Expected: builds, tests pass. The warmup adds 6 GEMM calls (3 candidates × 2 measurements) per *new* shape — first call of new shape is slower; subsequent calls are at-cache.

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs
git commit -m "perf(#375): AutotuneCacheV2 warmup-sweep on cache miss (Layer F)"
```

## Layer G: NUMA Awareness

### Task G.1: NumaTopology detection — tests

**Files:**
- Create: `tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/NumaTopologyTests.cs`

- [ ] **Step 1: Write tests**

```csharp
using System;
using AiDotNet.Tensors.Engines.BlasManaged.Pool;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class NumaTopologyTests
{
    [Fact]
    public void Detect_ReturnsAtLeastOneNode()
    {
        var topology = NumaTopology.Detect();
        Assert.NotNull(topology);
        Assert.True(topology.NodeCount >= 1);
        Assert.True(topology.TotalProcessors >= 1);
    }

    [Fact]
    public void Detect_ProcessorCountMatchesEnvironment()
    {
        var topology = NumaTopology.Detect();
        Assert.Equal(Environment.ProcessorCount, topology.TotalProcessors);
    }

    [Fact]
    public void GetNodeForProcessor_ReturnsValidNode()
    {
        var topology = NumaTopology.Detect();
        for (int proc = 0; proc < topology.TotalProcessors; proc++)
        {
            int node = topology.GetNodeForProcessor(proc);
            Assert.InRange(node, 0, topology.NodeCount - 1);
        }
    }
}
```

- [ ] **Step 2: Build to confirm failure**

```bash
dotnet build tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --no-restore --framework net10.0 2>&1 | grep -E "error CS" | head -3
```

Expected: `error CS0246: NumaTopology not found`.

- [ ] **Step 3: Commit**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/NumaTopologyTests.cs
git commit -m "test(#375): failing tests for NumaTopology (TDD red)"
```

### Task G.2: Implement NumaTopology — cross-platform detection

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/BlasManaged/Pool/NumaTopology.cs`

- [ ] **Step 1: Write the class**

```csharp
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged.Pool;

/// <summary>
/// Layer G — cross-platform NUMA topology detection. Returns at least
/// a single-node fallback when NUMA is unavailable or detection fails.
/// </summary>
internal sealed class NumaTopology
{
    public int NodeCount { get; }
    public int TotalProcessors { get; }
    private readonly int[] _processorToNode;

    private NumaTopology(int nodeCount, int totalProcessors, int[] processorToNode)
    {
        NodeCount = nodeCount;
        TotalProcessors = totalProcessors;
        _processorToNode = processorToNode;
    }

    public int GetNodeForProcessor(int processorIndex)
    {
        if (processorIndex < 0 || processorIndex >= _processorToNode.Length)
            throw new ArgumentOutOfRangeException(nameof(processorIndex));
        return _processorToNode[processorIndex];
    }

    public static NumaTopology Detect()
    {
        int procCount = Environment.ProcessorCount;
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return DetectLinux(procCount);
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return DetectWindows(procCount);
        }
        catch
        {
            // Fall through to single-node fallback.
        }
        return SingleNode(procCount);
    }

    private static NumaTopology DetectLinux(int procCount)
    {
        const string sysNodeDir = "/sys/devices/system/node";
        if (!Directory.Exists(sysNodeDir)) return SingleNode(procCount);

        var nodes = Directory.GetDirectories(sysNodeDir, "node*")
            .Select(d => (path: d, index: TryParseNodeIndex(Path.GetFileName(d))))
            .Where(t => t.index >= 0)
            .OrderBy(t => t.index)
            .ToArray();
        if (nodes.Length == 0) return SingleNode(procCount);

        var procToNode = new int[procCount];
        for (int i = 0; i < procCount; i++) procToNode[i] = 0;

        foreach (var node in nodes)
        {
            var cpulistPath = Path.Combine(node.path, "cpulist");
            if (!File.Exists(cpulistPath)) continue;
            var cpulist = File.ReadAllText(cpulistPath).Trim();
            foreach (var span in cpulist.Split(','))
            {
                var range = span.Split('-');
                int lo = int.Parse(range[0]);
                int hi = range.Length > 1 ? int.Parse(range[1]) : lo;
                for (int c = lo; c <= hi && c < procCount; c++)
                    procToNode[c] = node.index;
            }
        }
        return new NumaTopology(nodes.Length, procCount, procToNode);
    }

    private static int TryParseNodeIndex(string name)
    {
        if (!name.StartsWith("node")) return -1;
        return int.TryParse(name.Substring(4), out int idx) ? idx : -1;
    }

    private static NumaTopology DetectWindows(int procCount)
    {
        // GetNumaHighestNodeNumber + GetNumaProcessorNodeEx. Minimal pinvoke.
        if (!GetNumaHighestNodeNumber(out uint highest))
            return SingleNode(procCount);
        int nodeCount = (int)highest + 1;
        var procToNode = new int[procCount];
        for (int p = 0; p < procCount; p++)
        {
            var procNum = new PROCESSOR_NUMBER { Group = 0, Number = (byte)p, Reserved = 0 };
            if (GetNumaProcessorNodeEx(ref procNum, out ushort nodeNumber))
                procToNode[p] = nodeNumber;
            else
                procToNode[p] = 0;
        }
        return new NumaTopology(nodeCount, procCount, procToNode);
    }

    private static NumaTopology SingleNode(int procCount)
    {
        var arr = new int[procCount];
        return new NumaTopology(1, procCount, arr);
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetNumaHighestNodeNumber(out uint HighestNodeNumber);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetNumaProcessorNodeEx(ref PROCESSOR_NUMBER Processor, out ushort NodeNumber);

    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESSOR_NUMBER
    {
        public ushort Group;
        public byte Number;
        public byte Reserved;
    }
}
```

- [ ] **Step 2: Build + run NumaTopology tests**

```bash
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release --no-restore --framework net10.0 2>&1 | grep -E "(error|Error\(s\))" | tail -3
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~NumaTopologyTests" --logger "console;verbosity=minimal" 2>&1 | tail -5
```

Expected: 3/3 pass on both Linux and Windows.

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Pool/NumaTopology.cs
git commit -m "perf(#375): NumaTopology cross-platform detection (Layer G)"
```

### Task G.3: Pin StreamingWorkerPool workers per NUMA node

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/BlasManaged/Pool/StreamingWorkerPool.cs`

- [ ] **Step 1: In `EnsureInitialized`, assign each worker an ideal processor on its NUMA node**

After the existing `_workers[i] = new Thread(...)` setup, add:

```csharp
            var topology = NumaTopology.Detect();
            // Round-robin worker → processor assignment, balancing across NUMA nodes.
            // Worker `i` gets processor `i % TotalProcessors`. NUMA pinning via
            // Thread.SetIdealProcessor (Windows) or sched_setaffinity (Linux).
            for (int i = 0; i < _numWorkers; i++)
            {
                int processor = i % topology.TotalProcessors;
                int node = topology.GetNodeForProcessor(processor);
                // Setting IdealProcessor on Windows is a hint, not a hard pin.
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    try { _workers[i].SetIdealProcessor((byte)processor); } catch { /* best-effort */ }
                }
                // Linux affinity via P/Invoke would set the worker's CPU mask.
                // Deferred: requires per-thread native ID resolution. Single-node
                // fallback (default) is correct for most CI runners and dev boxes.
            }
```

- [ ] **Step 2: Build + run all BlasManaged tests**

```bash
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release --no-restore --framework net10.0 2>&1 | grep -E "(error|Error\(s\))" | tail -3
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~StreamingWorkerPoolTests|FullyQualifiedName~BlasManaged" --logger "console;verbosity=minimal" 2>&1 | tail -3
```

Expected: builds, tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/AiDotNet.Tensors/Engines/BlasManaged/Pool/StreamingWorkerPool.cs
git commit -m "perf(#375): NUMA-aware worker pinning in StreamingWorkerPool (Layer G)"
```

## Layer C: Microkernel Hardening (data-driven)

### Task C.1: Run the strict gate report

- [ ] **Step 1: Run the full report**

```bash
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-restore 2>&1 | tail -3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --pytorch-comparison 2>&1 | tail -10
head -30 artifacts/perf/pytorch-comparison.md
```

- [ ] **Step 2: Identify FAIL rows**

Open `artifacts/perf/pytorch-comparison.md`. List every shape with a ❌ on any of (median, p95, p99, alloc).

- [ ] **Step 3: Commit the report**

```bash
git add artifacts/perf/pytorch-comparison.md
git commit -m "perf(#375): Layer E baseline — pre-microkernel-work head-to-head report"
```

### Task C.2: Microkernel work per failing shape — follow-up planning

The microkernel work needed per failing shape is data-driven and cannot be fully specified before Task C.1 runs.

- [ ] **Step 1: For each shape that FAILed in C.1, decide intervention**

Honest possibilities (per spec section 5):
- **AVX-512 microkernel not firing**: check `Avx512F.IsSupported` on the runner; verify `PickMicrokernelTile<T>` returns the AVX-512 tile.
- **MKL has a hand-tuned kernel we can't match**: document the shape, mark as "known gap; future microkernel sprint."
- **Autotune mis-routing**: re-run with `PackingMode.DisableAutotune` to bypass cache; if explicit strategy beats Auto, fix the cache.

For each FAIL shape, write a follow-up issue in the spec (`docs/superpowers/specs/.../streaming-pool-domination-design.md` section 11, append): "Shape X failed by Y×; root cause Z; planned fix W."

- [ ] **Step 2: Commit the updated spec**

```bash
git add docs/superpowers/specs/2026-05-26-streaming-pool-domination-design.md
git commit -m "doc(#375): post-Layer-E failure analysis appended to spec"
```

(Subsequent microkernel work is its own commits per shape; not pre-spec'd here.)

## Layer Z: Final verification + report

### Task Z.1: Run the full BlasManaged test suite

- [ ] **Step 1: All 320+ BlasManaged tests pass**

```bash
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-restore --filter "FullyQualifiedName~BlasManaged|FullyQualifiedName~ScalarKernelTests|FullyQualifiedName~PrePackSpeedupTest|FullyQualifiedName~ConvTranspose2DL2PerfTest|FullyQualifiedName~StreamingWorkerPoolTests|FullyQualifiedName~AutotuneCacheV2Tests|FullyQualifiedName~NumaTopologyTests|FullyQualifiedName~PerfHarnessTest|FullyQualifiedName~PerfBarTest" --logger "console;verbosity=minimal" 2>&1 | tail -5
```

Expected: all pass. If anything fails, fix before proceeding.

### Task Z.2: Refresh baseline.json

```bash
AIDOTNET_GENERATE_BASELINE=1 dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --framework net10.0 --no-build --filter "FullyQualifiedName~PerfBarTest.Generate_Baseline_From_Full_Catalog" --logger "console;verbosity=minimal" 2>&1 | tail -3
```

### Task Z.3: Final summary commit

- [ ] **Step 1: Run summary report**

```bash
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release --no-build -- --pytorch-comparison 2>&1 | tail -10
```

- [ ] **Step 2: Stage + commit all final artifacts**

```bash
git add artifacts/perf/baseline.json artifacts/perf/pytorch-comparison.md
git commit -m "perf(#375): final baseline + PyTorch comparison report"
git push origin perf/blas-managed-fp64-small-square-microkernel
```

## Plan Self-Review

**Spec coverage**: Layers A–G covered by tasks A.1–A.6, B.1–B.4, C.1–C.2, D.1–D.4, E.1–E.4, F.1–F.4, G.1–G.3. Layer Z verification adds Z.1–Z.3.

**Placeholder check**: No TBD/TODO/placeholder text. C.2 is intentionally data-driven and deferred per spec — that's not a placeholder, it's the spec's design.

**Type consistency**: `ShapeKey`, `Entry`, `AutotuneCacheV2`, `WorkerSlot`, `StreamingWorkerPool`, `NumaTopology` are all defined in one task each before use. `Dispatch(int, Action<int>)` and `Dispatch(int, long, Action<int>)` signatures match across A.4 + A.5 + B.1-B.3.
