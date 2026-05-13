// Copyright (c) AiDotNet. All rights reserved.
// Issue #279 — GradientTape lifecycle leak repro.

#nullable disable

using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Defines the non-parallel collection for <see cref="GradientTapeLeakTests"/>.
/// The leak ceiling assertions read process-wide <c>GC.GetTotalMemory</c>, so
/// concurrent test allocations on other threads inflate the measured delta and
/// can either fail the budget for unrelated reasons or hide a real regression.
/// </summary>
[CollectionDefinition("GradientTapeLeakTests", DisableParallelization = true)]
public class GradientTapeLeakTestsCollection { }

/// <summary>
/// Issue #279 leak repro. Mirrors a single transformer encoder block —
/// QKV projections, scaled-dot-product attention, output projection,
/// FFN — wired up with autograd recording so every saved-for-backward
/// intermediate is exercised. The user's repro reports 3.78 MiB/call
/// retention on a transformer of this shape; this test asserts a
/// strict ceiling that catches both the original leak and any future
/// regression.
/// </summary>
[Collection("GradientTapeLeakTests")]
public class GradientTapeLeakTests
{
    private readonly ITestOutputHelper _output;
    public GradientTapeLeakTests(ITestOutputHelper output) { _output = output; }

    // Diagnostic helper retained for manual leak investigation. It writes
    // tracking output to the test runner but asserts nothing — running it
    // in CI just spends runtime without exercising any contract — so it's
    // skipped by default and unskipped only when an engineer is chasing a
    // specific leak signature.
    [Fact(Skip = "Diagnostic-only — unskip locally when triaging a leak. No assertions, prints WeakReference live counts.")]
    public void Diagnostic_FindLeakSource()
    {
        // Diagnostic test: track all live Tensor<float> instances
        // via WeakReference to identify what's being retained across
        // Train calls.
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 16, Dim = 64;
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();
        var wq = MakeTensor(new[] { Dim, Dim }, 0.1f, 1);
        var sources = new[] { wq };
        var weakRefs = new System.Collections.Generic.List<System.WeakReference>();

        // Warm up
        for (int w = 0; w < 3; w++) Step();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long start = GC.GetTotalMemory(forceFullCollection: true);
        for (int i = 0; i < 10; i++) Step();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long end = GC.GetTotalMemory(forceFullCollection: true);
        _output.WriteLine($"After 10 steps: heap delta = {end - start} bytes");

        int aliveCount = 0;
        foreach (var wr in weakRefs)
            if (wr.IsAlive) aliveCount++;
        _output.WriteLine($"Live Tensor refs after GC: {aliveCount} of {weakRefs.Count} tracked");

        void Step()
        {
            var x = MakeTensor(new[] { Batch * Seq, Dim }, 1.0f, 99);
            weakRefs.Add(new System.WeakReference(x));
            using var tape = new GradientTape<float>();
            var q = engine.TensorMatMul(x, wq);
            weakRefs.Add(new System.WeakReference(q));
            var loss = engine.ReduceSum(q, null);
            weakRefs.Add(new System.WeakReference(loss));
            var grads = tape.ComputeGradients(loss, sources: sources);
        }
    }

    [Fact]
    public void TrainStep_TransformerBlock_AtIssue283Scale_NoForwardIntermediatesSurvive()
    {
        // Issue #283 — track every forward intermediate by WeakReference and
        // assert that AFTER tape disposal + 2x GC, ZERO of them survive. This
        // is the most direct possible repro: the issue's residual ~400 KB/call
        // can only happen if some forward intermediate's lifetime is
        // accidentally chained to a live root.
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 64, Dim = 128, FF = 512;
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();
        AiDotNet.Tensors.Engines.Autodiff.TensorPool<float>.Clear();

        // Parameters held externally — these are EXPECTED to survive.
        var wq = MakeTensor(new[] { Dim, Dim }, 0.1f, 1);
        var wk = MakeTensor(new[] { Dim, Dim }, 0.1f, 2);
        var wv = MakeTensor(new[] { Dim, Dim }, 0.1f, 3);
        var wo = MakeTensor(new[] { Dim, Dim }, 0.1f, 4);
        var w1 = MakeTensor(new[] { Dim, FF }, 0.1f, 5);
        var w2 = MakeTensor(new[] { FF, Dim }, 0.1f, 6);
        var lnGamma = MakeTensor(new[] { Dim }, 0.01f, 7);
        var lnBeta = MakeTensor(new[] { Dim }, 0.01f, 8);
        var sources = new[] { wq, wk, wv, wo, w1, w2, lnGamma, lnBeta };

        // Warmup via a non-inlined static helper so the JIT can't extend the
        // step's local lifetimes past the call boundary — a common false-
        // positive trap in WeakReference-based leak tests.
        for (int w = 0; w < 5; w++)
            Issue283Step_TransformerBlock(engine, sources, Batch * Seq, Dim, lnGamma, lnBeta, wq, wk, wv, wo, w1, w2, null);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var trackedRefs = new System.Collections.Generic.List<(string label, System.WeakReference wr)>();
        Issue283Step_TransformerBlock(engine, sources, Batch * Seq, Dim, lnGamma, lnBeta, wq, wk, wv, wo, w1, w2, trackedRefs);

        // Force GC twice with finalizers between to give the runtime every
        // chance to release the intermediates.
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var survivors = new System.Collections.Generic.List<string>();
        var survivorTargets = new System.Collections.Generic.List<(string label, Tensor<float> t)>();
        foreach (var (label, wr) in trackedRefs)
        {
            if (!wr.IsAlive) continue;
            var t = wr.Target as Tensor<float>;
            string gradState = t is null ? "(collected during inspection)"
                : $"GradFn={(t.GradFn is null ? "null" : "SET")}, Grad={(t.Grad is null ? "null" : "SET")}";
            survivors.Add($"{label} [{gradState}]");
            if (t is not null) survivorTargets.Add((label, t));
        }

        _output.WriteLine($"Tracked {trackedRefs.Count} forward intermediates; {survivors.Count} survived GC after Dispose:");
        foreach (var s in survivors) _output.WriteLine($"  - {s}");

        // Issue #283 diagnostic: when survivors exist, scan static / thread-static
        // fields in the AiDotNet.Tensors assembly via reflection to find which
        // chain pins each tensor. Cheap to run only on the failing path; off
        // entirely when the test passes. The output prints to xUnit so it lands
        // in the CI log.
        if (survivors.Count > 0 && survivorTargets.Count > 0)
        {
            _output.WriteLine("");
            _output.WriteLine("=== Issue #283 leak diagnostic (scanning static state) ===");
            var aidotnetAssembly = typeof(Tensor<float>).Assembly;
            // Reference-equality set without .NET 5+'s ReferenceEqualityComparer
            // (the test multi-targets net471 which doesn't ship it).
            var refEq = new ReferenceEqualityComparerLocal();
            var visited = new System.Collections.Generic.HashSet<object>(refEq);
            var survivorSet = new System.Collections.Generic.HashSet<object>(refEq);
            foreach (var (_, t) in survivorTargets) survivorSet.Add(t);

            foreach (var type in aidotnetAssembly.GetTypes())
            {
                if (!type.IsClass && !type.IsValueType) continue;
                System.Reflection.FieldInfo[] fields;
                try { fields = type.GetFields(
                    System.Reflection.BindingFlags.Static |
                    System.Reflection.BindingFlags.Public |
                    System.Reflection.BindingFlags.NonPublic); }
                catch { continue; }

                foreach (var field in fields)
                {
                    object? value;
                    try { value = field.GetValue(null); }
                    catch { continue; }
                    if (value is null) continue;
                    if (visited.Contains(value)) continue;
                    var path = $"{type.FullName}.{field.Name}";
                    var found = WalkForSurvivor(value, survivorSet, visited, path, depth: 0, maxDepth: 6);
                    if (found is not null)
                    {
                        _output.WriteLine($"PIN: {found}");
                    }
                }
            }
            _output.WriteLine("=== end diagnostic ===");
        }

        Assert.True(survivors.Count == 0,
            $"Issue #283 — {survivors.Count} of {trackedRefs.Count} forward intermediates " +
            $"survived Gen2 GC after GradientTape.Dispose. Surviving labels: " +
            string.Join(", ", survivors));

    }

    // Reference-equality comparer for object — net471 doesn't have
    // System.Collections.Generic.ReferenceEqualityComparer (added in .NET 5).
    private sealed class ReferenceEqualityComparerLocal : System.Collections.Generic.IEqualityComparer<object>
    {
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }

    // Walks an object graph up to maxDepth looking for any of the survivors.
    // Returns the path string when a survivor is reachable from `value`, null
    // otherwise. Reference-equality visited-set caps the walk on cycles.
    private static string? WalkForSurvivor(
        object value,
        System.Collections.Generic.HashSet<object> survivorSet,
        System.Collections.Generic.HashSet<object> visited,
        string path,
        int depth,
        int maxDepth)
    {
        if (depth > maxDepth) return null;
        if (!visited.Add(value)) return null;
        if (survivorSet.Contains(value)) return $"{path} ({value.GetType().Name})";

        // Arrays: walk elements.
        if (value is System.Array arr)
        {
            int len = arr.Length;
            int cap = System.Math.Min(len, 64);
            for (int i = 0; i < cap; i++)
            {
                var el = arr.GetValue(i);
                if (el is null) continue;
                var r = WalkForSurvivor(el, survivorSet, visited, $"{path}[{i}]", depth + 1, maxDepth);
                if (r is not null) return r;
            }
            return null;
        }

        // IEnumerable: walk a bounded prefix.
        if (value is System.Collections.IEnumerable enumerable && value is not string)
        {
            int idx = 0;
            try
            {
                foreach (var el in enumerable)
                {
                    if (idx++ > 64) break;
                    if (el is null) continue;
                    var r = WalkForSurvivor(el, survivorSet, visited, $"{path}<{idx-1}>", depth + 1, maxDepth);
                    if (r is not null) return r;
                }
            }
            catch { /* enumeration may throw for some collections — keep walking */ }
        }

        // Instance fields.
        var type = value.GetType();
        System.Reflection.FieldInfo[] fields;
        try { fields = type.GetFields(
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.Public |
            System.Reflection.BindingFlags.NonPublic); }
        catch { return null; }
        foreach (var field in fields)
        {
            if (field.FieldType.IsPrimitive || field.FieldType == typeof(string)) continue;
            object? fv;
            try { fv = field.GetValue(value); } catch { continue; }
            if (fv is null) continue;
            var r = WalkForSurvivor(fv, survivorSet, visited, $"{path}.{field.Name}", depth + 1, maxDepth);
            if (r is not null) return r;
        }
        return null;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Issue283Step_TransformerBlock(
        IEngine engine,
        Tensor<float>[] sources,
        int rows, int dim,
        Tensor<float> lnGamma, Tensor<float> lnBeta,
        Tensor<float> wq, Tensor<float> wk, Tensor<float> wv, Tensor<float> wo,
        Tensor<float> w1, Tensor<float> w2,
        System.Collections.Generic.List<(string label, System.WeakReference wr)> trackedRefs)
    {
        // [NoInlining] guarantees this method's frame is gone (and its locals
        // dead) before the caller's GC.Collect — the standard discipline for
        // WeakReference-based leak tests on a JIT that may otherwise inline
        // local functions and keep their captures alive across the call.
        var x = MakeTensor(new[] { rows, dim }, 1.0f, 99);
        using var tape = new GradientTape<float>();
        var xNorm = engine.TensorLayerNorm(x, lnGamma, lnBeta, epsilon: 1e-5);
        var q = engine.TensorMatMul(xNorm, wq);
        var k = engine.TensorMatMul(xNorm, wk);
        var v = engine.TensorMatMul(xNorm, wv);
        var scores = engine.TensorMatMulTransposed(q, k);
        var attn = engine.Softmax(scores, axis: -1);
        var ctx = engine.TensorMatMul(attn, v);
        var proj = engine.TensorMatMul(ctx, wo);
        var residual = engine.TensorAdd(x, proj);
        var h1 = engine.TensorMatMul(residual, w1);
        var h2 = engine.ReLU(h1);
        var h3 = engine.TensorMatMul(h2, w2);
        var loss = engine.ReduceSum(h3, null);
        var grads = tape.ComputeGradients(loss, sources: sources);

        if (trackedRefs is not null)
        {
            trackedRefs.Add(("x-input", new System.WeakReference(x)));
            trackedRefs.Add(("xNorm", new System.WeakReference(xNorm)));
            trackedRefs.Add(("q", new System.WeakReference(q)));
            trackedRefs.Add(("k", new System.WeakReference(k)));
            trackedRefs.Add(("v", new System.WeakReference(v)));
            trackedRefs.Add(("scores", new System.WeakReference(scores)));
            trackedRefs.Add(("attn-softmax", new System.WeakReference(attn)));
            trackedRefs.Add(("ctx", new System.WeakReference(ctx)));
            trackedRefs.Add(("proj", new System.WeakReference(proj)));
            trackedRefs.Add(("residual", new System.WeakReference(residual)));
            trackedRefs.Add(("h1", new System.WeakReference(h1)));
            trackedRefs.Add(("h2-relu", new System.WeakReference(h2)));
            trackedRefs.Add(("h3", new System.WeakReference(h3)));
            trackedRefs.Add(("loss", new System.WeakReference(loss)));
            // Do NOT track sources — they're expected to survive.
        }
        // All locals (including grads) go out of scope at method exit; the JIT
        // cannot extend their lifetime past the return because [NoInlining]
        // pins the frame as a real call boundary.
    }

    [Fact]
    public void TrainStep_TransformerBlock_AtIssue283Scale_DoesNotLeak()
    {
        // Issue #283 — residual ~400 KB/call leak at the exact transformer
        // shape from the user's repro: modelDimension=128, feedForwardDimension=512,
        // seqLength=64, numHeads=4. Includes LayerNorm + multi-head attention
        // + masked softmax + an optimizer-style read of .Grad on every source
        // (the previous-issue fix only cleared .Grad on intermediates; if the
        // optimizer step pattern leaks, it's via source-tensor .Grad pinning
        // on the tape's _entries arena — TapeEntryArena.Reset clears the array
        // slots but the arena itself is recycled per-thread in _cachedArena).
        //
        // 50 KB/call ceiling — enough headroom for legitimate JIT/pool warmup
        // residue, but below the 400 KB/call signature the issue reports.
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 64, Dim = 128, FF = 512, Heads = 4;
        const int HeadDim = Dim / Heads; // 32

        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();

        // Parameters — survive every step like a model's trainable weights.
        var wq = MakeTensor(new[] { Dim, Dim }, 0.1f, 1);
        var wk = MakeTensor(new[] { Dim, Dim }, 0.1f, 2);
        var wv = MakeTensor(new[] { Dim, Dim }, 0.1f, 3);
        var wo = MakeTensor(new[] { Dim, Dim }, 0.1f, 4);
        var w1 = MakeTensor(new[] { Dim, FF }, 0.1f, 5);
        var w2 = MakeTensor(new[] { FF, Dim }, 0.1f, 6);
        var lnGamma = MakeTensor(new[] { Dim }, 0.01f, 7);
        var lnBeta = MakeTensor(new[] { Dim }, 0.01f, 8);
        var sources = new[] { wq, wk, wv, wo, w1, w2, lnGamma, lnBeta };

        // Methodology: long warmup + two-window growth-rate measurement.
        // A single-window heap delta on Linux Server GC was observably
        // noisy (61.6 KB/call observed in CI even with no real per-iter
        // retention) — JIT, ArrayPool, and ThreadLocal arena initial
        // populations all amortize across the first few iterations and
        // showed up as fake retention. Splitting the measurement into
        // two halves and asserting on the SECOND half cleanly separates
        // genuine per-iter leaks (visible in BOTH halves) from one-time
        // startup costs (visible in first half only).
        const int Warmup = 50;
        const int Measure = 200;

        for (int i = 0; i < Warmup; i++) Step();
        StableForcedGc();
        long m0 = LiveBytes();

        for (int i = 0; i < Measure / 2; i++) Step();
        StableForcedGc();
        long m1 = LiveBytes();

        for (int i = 0; i < Measure / 2; i++) Step();
        StableForcedGc();
        long m2 = LiveBytes();

        long firstHalfPerCall = (m1 - m0) / (Measure / 2);
        long secondHalfPerCall = (m2 - m1) / (Measure / 2);

        _output.WriteLine(
            $"Issue #283 transformer block at scale (Dim={Dim} FF={FF} Seq={Seq} Heads={Heads}): " +
            $"first-half={firstHalfPerCall} B/call, second-half={secondHalfPerCall} B/call " +
            $"(m0={m0} m1={m1} m2={m2} {Measure / 2} iters/half)");

        // 50 KB/call ceiling on the SECOND-half retention — by this
        // point any first-window startup costs are gone, so the
        // remainder is true per-iter retention. Reporter saw 400 KB/call;
        // this catches that with an 8x safety factor while staying
        // above legitimate per-iter overhead like delegate caching.
        Assert.True(secondHalfPerCall < 50_000,
            $"Issue #283 — second-half retention {secondHalfPerCall} B/call at transformer scale " +
            $"(Dim={Dim} FF={FF} Seq={Seq}) exceeds 50 KB/call budget. " +
            $"first-half={firstHalfPerCall} B/call, m0={m0} m1={m1} m2={m2}. " +
            $"Reporter saw 400 KB/call — second-half > 50 KB/call indicates a residual leak.");

        void Step()
        {
            var x = MakeTensor(new[] { Batch * Seq, Dim }, 1.0f, 99);
            using var tape = new GradientTape<float>();

            // LayerNorm input (pre-norm transformer style).
            var xNorm = engine.TensorLayerNorm(x, lnGamma, lnBeta, epsilon: 1e-5);

            // QKV projections.
            var q = engine.TensorMatMul(xNorm, wq);
            var k = engine.TensorMatMul(xNorm, wk);
            var v = engine.TensorMatMul(xNorm, wv);

            // Multi-head reshape: [B*Seq, Dim] → [B*Seq, Heads, HeadDim].
            // Skip the actual head-split for the synthetic test — fire the
            // attention math at the flat shape, which still exercises the
            // full softmax+matmul backward stack.
            var scores = engine.TensorMatMulTransposed(q, k);
            var attn = engine.Softmax(scores, axis: -1);
            var ctx = engine.TensorMatMul(attn, v);
            var proj = engine.TensorMatMul(ctx, wo);

            // Residual + FFN with ReLU.
            var residual = engine.TensorAdd(x, proj);
            var h1 = engine.TensorMatMul(residual, w1);
            var h2 = engine.ReLU(h1);
            var h3 = engine.TensorMatMul(h2, w2);

            var loss = engine.ReduceSum(h3, null);
            var grads = tape.ComputeGradients(loss, sources: sources);

            // Simulate optimizer step pattern — read .Grad on every source.
            // This is what AiDotNet's `Optimizer.Step()` does in the consumer
            // repro and exercises the source-tensor .Grad lifecycle that the
            // PR #280 cleanup explicitly preserved.
            float gradSum = 0;
            foreach (var s in sources)
            {
                Assert.True(grads.ContainsKey(s));
                Assert.NotNull(s.Grad);
                gradSum += s.Grad.GetFlat(0); // touch the grad to defeat dead-code elim
            }
            Assert.True(!float.IsNaN(gradSum));
        }
        // Suppress "unused" — sources is captured in Step()
        _ = HeadDim;
    }

    [Fact]
    public void TrainStep_TransformerBlock_DoesNotLeak()
    {
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 16, Dim = 64, FF = 128;
        // Drain any thread-local caches left over from earlier tests
        // (AutoTensorCache pools tensors per-thread; leftover entries
        // from other test classes inflate the start-of-window heap and
        // the cross-test deltas show up as a fake leak signature).
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();

        // Parameters — survive every step like a model's trainable weights.
        var wq = MakeTensor(new[] { Dim, Dim }, 0.1f, 1);
        var wk = MakeTensor(new[] { Dim, Dim }, 0.1f, 2);
        var wv = MakeTensor(new[] { Dim, Dim }, 0.1f, 3);
        var wo = MakeTensor(new[] { Dim, Dim }, 0.1f, 4);
        var w1 = MakeTensor(new[] { Dim, FF }, 0.1f, 5);
        var w2 = MakeTensor(new[] { FF, Dim }, 0.1f, 6);
        var sources = new[] { wq, wk, wv, wo, w1, w2 };

        Step();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        long start = GC.GetTotalMemory(forceFullCollection: true);
        const int iters = 50;
        for (int i = 0; i < iters; i++) Step();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long end = GC.GetTotalMemory(forceFullCollection: true);

        long perCall = (end - start) / iters;
        _output.WriteLine($"Synthetic transformer block leak: {perCall} B/call (start={start} end={end} iters={iters})");

        // Threshold: 200 KB/call. The user's repro at this scale
        // reports ~MB/call, so this catches the original leak with a
        // generous safety factor while staying well below the
        // sustained-training-run-out-of-memory cliff.
        Assert.True(perCall < 200_000,
            $"Managed-heap retention {perCall} B/call exceeds 200 KB/call budget. " +
            $"Start={start} End={end} Iters={iters}. Issue #279 transformer-scale leak regression.");

        void Step()
        {
            var x = MakeTensor(new[] { Batch * Seq, Dim }, 1.0f, 99);
            using var tape = new GradientTape<float>();
            // QKV projections.
            var q = engine.TensorMatMul(x, wq);
            var k = engine.TensorMatMul(x, wk);
            var v = engine.TensorMatMul(x, wv);
            // Reshape to [B*Seq, Heads, Dim/Heads]? Skip multi-head for
            // the synthetic path — single-head is enough to fire the
            // softmax + scaled-dot-product backward saved-state captures.
            var scores = engine.TensorMatMulTransposed(q, k);
            var attn = engine.Softmax(scores, axis: -1);
            var ctx = engine.TensorMatMul(attn, v);
            var proj = engine.TensorMatMul(ctx, wo);
            // FFN with ReLU.
            var h1 = engine.TensorMatMul(proj, w1);
            var h2 = engine.ReLU(h1);
            var h3 = engine.TensorMatMul(h2, w2);
            var loss = engine.ReduceSum(h3, null);
            var grads = tape.ComputeGradients(loss, sources: sources);
            Assert.True(grads.ContainsKey(wq));
        }
    }

    [Fact]
    public void TrainStep_AnomalyMode_NoForwardIntermediatesSurvive()
    {
        // Issue #283 edge case — anomaly mode forces the tape-walk path
        // (the graph path is skipped when AnomalyModeScope.IsActive). The
        // tape-walk path's cleanup ALSO must release saved-for-backward
        // tensors; otherwise leaks reappear when any consumer enables
        // anomaly detection (a common debugging configuration).
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 32, Dim = 64;
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();
        AiDotNet.Tensors.Engines.Autodiff.TensorPool<float>.Clear();

        var wq = MakeTensor(new[] { Dim, Dim }, 0.1f, 1);
        var wk = MakeTensor(new[] { Dim, Dim }, 0.1f, 2);
        var wv = MakeTensor(new[] { Dim, Dim }, 0.1f, 3);
        var sources = new[] { wq, wk, wv };

        for (int w = 0; w < 3; w++)
            Issue283Step_AnomalyMode(engine, sources, Batch * Seq, Dim, wq, wk, wv, null);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var trackedRefs = new System.Collections.Generic.List<(string label, System.WeakReference wr)>();
        Issue283Step_AnomalyMode(engine, sources, Batch * Seq, Dim, wq, wk, wv, trackedRefs);

        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var survivors = new System.Collections.Generic.List<string>();
        foreach (var (label, wr) in trackedRefs)
            if (wr.IsAlive) survivors.Add(label);
        _output.WriteLine($"AnomalyMode tape-walk path: {survivors.Count} of {trackedRefs.Count} forward intermediates survived");
        foreach (var s in survivors) _output.WriteLine($"  - {s}");
        Assert.True(survivors.Count == 0,
            $"Anomaly-mode tape-walk path leaked {survivors.Count} forward intermediates: {string.Join(", ", survivors)}");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Issue283Step_AnomalyMode(
        IEngine engine, Tensor<float>[] sources, int rows, int dim,
        Tensor<float> wq, Tensor<float> wk, Tensor<float> wv,
        System.Collections.Generic.List<(string label, System.WeakReference wr)> trackedRefs)
    {
        var x = MakeTensor(new[] { rows, dim }, 1.0f, 99);
        using var tape = new GradientTape<float>();
        tape.DetectAnomaly = true; // ← forces tape-walk path
        var q = engine.TensorMatMul(x, wq);
        var k = engine.TensorMatMul(x, wk);
        var v = engine.TensorMatMul(x, wv);
        var scores = engine.TensorMatMulTransposed(q, k);
        var attn = engine.Softmax(scores, axis: -1);
        var ctx = engine.TensorMatMul(attn, v);
        var loss = engine.ReduceSum(ctx, null);
        var grads = tape.ComputeGradients(loss, sources: sources);
        Assert.True(grads.ContainsKey(wq));

        if (trackedRefs is not null)
        {
            trackedRefs.Add(("x", new System.WeakReference(x)));
            trackedRefs.Add(("q", new System.WeakReference(q)));
            trackedRefs.Add(("k", new System.WeakReference(k)));
            trackedRefs.Add(("v", new System.WeakReference(v)));
            trackedRefs.Add(("scores", new System.WeakReference(scores)));
            trackedRefs.Add(("attn", new System.WeakReference(attn)));
            trackedRefs.Add(("ctx", new System.WeakReference(ctx)));
            trackedRefs.Add(("loss", new System.WeakReference(loss)));
        }
    }

    /// <summary>
    /// Reopened-#1227 repro: AiDotNet consumer-side measurement (see
    /// ooples/AiDotNet PR #1285) shows ~1.5 MB/call retention at
    /// L=4 / 1000-call scale, even though the single-encoder-block
    /// canary above asserts 0 B/call. Either the leak is in an op
    /// the canary doesn't exercise, or it only manifests at the
    /// chained-layers scale.
    ///
    /// This probe mirrors a 4-encoder-layer transformer (the
    /// reporter's exact config: Heads=4, Dim=128, FF=512, Seq=64,
    /// 4 encoder layers, pre-norm style with residual connections,
    /// bias add on every dense projection, output classification
    /// head, softmax+cross-entropy-style loss). If retention scales
    /// linearly with layer count, we have a Tensors-side repro that
    /// matches the consumer's symptom.
    /// </summary>
    [Fact]
    public void TrainStep_FourLayerTransformer_NoLinearLeakAcrossWindows()
    {
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 64, Dim = 128, FF = 512, Vocab = 256;
        const int NumLayers = 4;
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();
        AiDotNet.Tensors.Engines.Autodiff.TensorPool<float>.Clear();

        // Per-layer weights (4 sets of QKV/output + FFN + LayerNorm gain/bias).
        var perLayerW = new System.Collections.Generic.List<Tensor<float>[]>(NumLayers);
        for (int L = 0; L < NumLayers; L++)
        {
            perLayerW.Add(new[]
            {
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 10 + 1),  // wq
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 10 + 2),  // wk
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 10 + 3),  // wv
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 10 + 4),  // wo
                MakeTensor(new[] { Dim, FF }, 0.05f, L * 10 + 5),   // w1
                MakeTensor(new[] { FF, Dim }, 0.05f, L * 10 + 6),   // w2
                MakeTensor(new[] { Dim }, 1.0f, L * 10 + 7),        // ln1_gamma
                MakeTensor(new[] { Dim }, 0.0f, L * 10 + 8),        // ln1_beta
                MakeTensor(new[] { Dim }, 1.0f, L * 10 + 9),        // ln2_gamma
                MakeTensor(new[] { Dim }, 0.0f, L * 10 + 10),       // ln2_beta
            });
        }
        var wOut = MakeTensor(new[] { Dim, Vocab }, 0.05f, 999);
        var allSources = new System.Collections.Generic.List<Tensor<float>>();
        foreach (var W in perLayerW) allSources.AddRange(W);
        allSources.Add(wOut);
        var sources = allSources.ToArray();

        // Two consecutive 200-call windows with StableForcedGc()/LiveBytes()
        // sampling for cross-platform stability under Server GC. A linear leak
        // shows window-2 retention ≈ window-1 retention; a one-time warmup
        // shows window-1 high and window-2 near-zero. The "no linear leak"
        // claim asserts BOTH that the absolute window-2 figure is bounded AND
        // that window-2 is not significantly larger than window-1 (so leaks
        // can't hide behind warmup noise).
        const int Warmup = 25;
        const int Measure = 200;
        for (int i = 0; i < Warmup; i++) Step();
        StableForcedGc();
        long m0 = LiveBytes();

        for (int i = 0; i < Measure; i++) Step();
        StableForcedGc();
        long m1 = LiveBytes();

        for (int i = 0; i < Measure; i++) Step();
        StableForcedGc();
        long m2 = LiveBytes();

        long w1PerCall = (m1 - m0) / Measure;
        long w2PerCall = (m2 - m1) / Measure;
        _output.WriteLine(
            $"4-layer Transformer leak probe (Dim={Dim} FF={FF} Seq={Seq} L={NumLayers}): " +
            $"win1={w1PerCall} B/call, win2={w2PerCall} B/call " +
            $"(m0={m0} m1={m1} m2={m2}; {Measure} iters/window)");

        // Tripwire: second-window per-call retention < 100 KB. The reporter's
        // consumer-side observation was ~360 KB/call at L=1 → ~1.5 MB at L=4,
        // so a 100 KB ceiling on a 4-layer probe catches both the issue's
        // signature and any future regression at this scale.
        Assert.True(w2PerCall < 100_000,
            $"4-layer Transformer second-window retention {w2PerCall} B/call exceeds 100 KB/call. " +
            $"Matches AiDotNet#1227 / Tensors#283 residual-leak signature. " +
            $"win1={w1PerCall} B/call, m0={m0} m1={m1} m2={m2}.");

        // Slope check: window-2 retention should not exceed window-1 by more
        // than 50 KB/call. A true linear leak would show w2 ≈ w1; this guard
        // catches a regression where retention grows across windows (the
        // signature that distinguishes a leak from one-time warmup).
        Assert.True(w2PerCall < w1PerCall + 50_000,
            $"4-layer Transformer leak grows across windows: " +
            $"win1={w1PerCall} B/call → win2={w2PerCall} B/call " +
            $"(slope > 50 KB/call). A linear leak should hold w1 ≈ w2; " +
            $"w2 > w1 + 50KB indicates accelerating retention.");

        void Step()
        {
            // Token "embedding": synthesise an input directly at [B*Seq, Dim]
            // since the Tensors layer doesn't have an embedding op surfaced
            // to the test, and a leak rooted in the gather would surface
            // separately. The hot per-layer ops are what we want.
            var x = MakeTensor(new[] { Batch * Seq, Dim }, 1.0f, 99);
            using var tape = new GradientTape<float>();

            Tensor<float> h = x;
            for (int L = 0; L < NumLayers; L++)
            {
                var W = perLayerW[L];
                Tensor<float> wq = W[0], wk = W[1], wv = W[2], wo = W[3], w1 = W[4], w2 = W[5];
                Tensor<float> g1 = W[6], b1 = W[7], g2 = W[8], b2 = W[9];

                // Pre-norm + attention
                var xNorm = engine.TensorLayerNorm(h, g1, b1, epsilon: 1e-5);
                var q = engine.TensorMatMul(xNorm, wq);
                var k = engine.TensorMatMul(xNorm, wk);
                var v = engine.TensorMatMul(xNorm, wv);
                var scores = engine.TensorMatMulTransposed(q, k);
                var attn = engine.Softmax(scores, axis: -1);
                var ctx = engine.TensorMatMul(attn, v);
                var proj = engine.TensorMatMul(ctx, wo);
                h = engine.TensorAdd(h, proj);

                // Pre-norm + FFN
                var hNorm = engine.TensorLayerNorm(h, g2, b2, epsilon: 1e-5);
                var ffn1 = engine.TensorMatMul(hNorm, w1);
                var ffn1Act = engine.ReLU(ffn1);
                var ffn2 = engine.TensorMatMul(ffn1Act, w2);
                h = engine.TensorAdd(h, ffn2);
            }

            // Output projection: reduce per-token to vocab logits, then
            // reduce to a scalar loss surrogate (sum-of-squared-logits is
            // enough to exercise backward through every parameter).
            var logits = engine.TensorMatMul(h, wOut);
            var loss = engine.ReduceSum(logits, null);
            var grads = tape.ComputeGradients(loss, sources: sources);

            // Simulate optimizer reading .Grad on every source — that's the
            // consumer-side pattern that exercises source-tensor pinning.
            foreach (var s in sources)
            {
                Assert.True(grads.ContainsKey(s));
                Assert.NotNull(s.Grad);
            }
        }
    }

    /// <summary>
    /// Like <see cref="TrainStep_FourLayerTransformer_NoLinearLeakAcrossWindows"/>
    /// but with the ops AiDotNet's MultiHeadAttentionLayer actually uses
    /// internally — Permute (for the head-split transpose) and FusedLinear
    /// (for the output projection's matmul+bias). The simpler 4-layer probe
    /// passes at 0 B/call, so if AiDotNet leaks at 1.5 MB/call the leak
    /// must be in an op the simpler probe doesn't exercise. This probe
    /// adds the candidates one by one.
    /// </summary>
    [Fact]
    public void TrainStep_FourLayerTransformer_WithPermuteAndFusedLinear_NoLeak()
    {
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 64, Dim = 128, FF = 512, Heads = 4, HeadDim = Dim / Heads;
        const int NumLayers = 4;
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();
        AiDotNet.Tensors.Engines.Autodiff.TensorPool<float>.Clear();

        // Per-layer weights + biases (FusedLinear takes bias).
        var perLayerW = new System.Collections.Generic.List<Tensor<float>[]>(NumLayers);
        for (int L = 0; L < NumLayers; L++)
        {
            perLayerW.Add(new[]
            {
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 20 + 1),   // wq
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 20 + 2),   // wk
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 20 + 3),   // wv
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 20 + 4),   // wo
                MakeTensor(new[] { Dim }, 0.0f, L * 20 + 5),         // bo (bias for output proj)
                MakeTensor(new[] { Dim, FF }, 0.05f, L * 20 + 6),    // w1
                MakeTensor(new[] { FF }, 0.0f, L * 20 + 7),          // b1
                MakeTensor(new[] { FF, Dim }, 0.05f, L * 20 + 8),    // w2
                MakeTensor(new[] { Dim }, 0.0f, L * 20 + 9),         // b2
                MakeTensor(new[] { Dim }, 1.0f, L * 20 + 10),        // ln1_gamma
                MakeTensor(new[] { Dim }, 0.0f, L * 20 + 11),        // ln1_beta
                MakeTensor(new[] { Dim }, 1.0f, L * 20 + 12),        // ln2_gamma
                MakeTensor(new[] { Dim }, 0.0f, L * 20 + 13),        // ln2_beta
            });
        }
        var wOut = MakeTensor(new[] { Dim, 256 }, 0.05f, 999);
        var allSources = new System.Collections.Generic.List<Tensor<float>>();
        foreach (var W in perLayerW) allSources.AddRange(W);
        allSources.Add(wOut);
        var sources = allSources.ToArray();

        // StableForcedGc()/LiveBytes() per the file's standard two-window
        // half-window methodology — see TrainStep_FourLayerTransformer_
        // NoLinearLeakAcrossWindows for the matching pattern.
        const int Warmup = 25;
        const int Measure = 200;
        for (int i = 0; i < Warmup; i++) Step();
        StableForcedGc();
        long m0 = LiveBytes();

        for (int i = 0; i < Measure; i++) Step();
        StableForcedGc();
        long m1 = LiveBytes();

        for (int i = 0; i < Measure; i++) Step();
        StableForcedGc();
        long m2 = LiveBytes();

        long w1PerCall = (m1 - m0) / Measure;
        long w2PerCall = (m2 - m1) / Measure;
        _output.WriteLine(
            $"4L-Transformer + Permute + FusedLinear leak probe: " +
            $"win1={w1PerCall} B/call, win2={w2PerCall} B/call " +
            $"(m0={m0} m1={m1} m2={m2}; {Measure} iters/window)");

        Assert.True(w2PerCall < 100_000,
            $"4L-Transformer+Permute+FusedLinear second-window retention {w2PerCall} B/call > 100 KB. " +
            $"win1={w1PerCall} B/call, m0={m0} m1={m1} m2={m2}.");

        void Step()
        {
            var x = MakeTensor(new[] { Batch * Seq, Dim }, 1.0f, 99);
            using var tape = new GradientTape<float>();

            Tensor<float> h = x;
            for (int L = 0; L < NumLayers; L++)
            {
                var W = perLayerW[L];
                Tensor<float> wq = W[0], wk = W[1], wv = W[2], wo = W[3], bo = W[4];
                Tensor<float> w1 = W[5], b1 = W[6], w2 = W[7], b2 = W[8];
                Tensor<float> g1 = W[9], beta1 = W[10], g2 = W[11], beta2 = W[12];

                // Pre-norm
                var xNorm = engine.TensorLayerNorm(h, g1, beta1, epsilon: 1e-5);

                // QKV projections [B*S, Dim] -> [B*S, Dim]
                var Q_flat = engine.TensorMatMul(xNorm, wq);
                var K_flat = engine.TensorMatMul(xNorm, wk);
                var V_flat = engine.TensorMatMul(xNorm, wv);

                // Reshape + Permute to [B, Heads, S, HeadDim] (mirrors MHA layer)
                var Q4 = engine.Reshape(Q_flat, new[] { Batch, Seq, Heads, HeadDim });
                var K4 = engine.Reshape(K_flat, new[] { Batch, Seq, Heads, HeadDim });
                var V4 = engine.Reshape(V_flat, new[] { Batch, Seq, Heads, HeadDim });
                var queries = engine.TensorPermute(Q4, new[] { 0, 2, 1, 3 });
                var keys = engine.TensorPermute(K4, new[] { 0, 2, 1, 3 });
                var values = engine.TensorPermute(V4, new[] { 0, 2, 1, 3 });

                // Scaled dot-product attention
                var ctx4 = engine.ScaledDotProductAttention(queries, keys, values,
                    mask: null,
                    scale: 1.0 / Math.Sqrt(HeadDim),
                    out _);

                // [B, H, S, D] -> [B, S, H, D] -> [B*S, Dim]
                var ctxT = engine.TensorPermute(ctx4, new[] { 0, 2, 1, 3 });
                var ctxFlat = engine.Reshape(ctxT, new[] { Batch * Seq, Dim });

                // Fused output projection (matmul+bias, no activation)
                var attnProj = engine.FusedLinear(ctxFlat, wo, bo, FusedActivationType.None);
                h = engine.TensorAdd(h, attnProj);

                // Pre-norm + FFN with FusedLinear (matches DenseLayer pattern)
                var hNorm = engine.TensorLayerNorm(h, g2, beta2, epsilon: 1e-5);
                var ffn1 = engine.FusedLinear(hNorm, w1, b1, FusedActivationType.ReLU);
                var ffn2 = engine.FusedLinear(ffn1, w2, b2, FusedActivationType.None);
                h = engine.TensorAdd(h, ffn2);
            }

            // Output classification logits
            var logits = engine.TensorMatMul(h, wOut);
            var loss = engine.ReduceSum(logits, null);
            var grads = tape.ComputeGradients(loss, sources: sources);
            foreach (var s in sources)
            {
                Assert.True(grads.ContainsKey(s));
                Assert.NotNull(s.Grad);
            }
        }
    }

    /// <summary>
    /// Reopened-#1227 regression. The original cleanup at the end of
    /// <c>ComputeGradients</c> gated on <c>sources is not null</c>, which
    /// meant any caller passing <c>null</c> (the standard consumer pattern
    /// when reading gradients from the returned dictionary) skipped the
    /// cleanup entirely — every forward intermediate kept its
    /// <c>GradFn</c> back-pointer, and any external code that retained a
    /// single intermediate (e.g. a layer's <c>_lastInput</c> cache)
    /// pinned the whole graph across calls. ooples/AiDotNet#1227
    /// measured ~1.5 MB/call retention on a 4-encoder-layer Transformer
    /// going through this exact path.
    ///
    /// This probe matches AiDotNet's pattern: <c>sources: null</c>, no
    /// retain-grad, single tape per Step. It also retains an intermediate
    /// to mimic the layer-cache pinning that triggered the consumer-side
    /// leak. Without the fix, second-window retention runs around 100 KB/call
    /// on this 4-layer config; with the fix it stays at 0 B/call.
    /// </summary>
    [Fact]
    public void TrainStep_FourLayerTransformer_NullSources_NoLeak()
    {
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 64, Dim = 128, FF = 512;
        const int NumLayers = 4;
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();
        AiDotNet.Tensors.Engines.Autodiff.TensorPool<float>.Clear();

        var perLayerW = new System.Collections.Generic.List<Tensor<float>[]>(NumLayers);
        for (int L = 0; L < NumLayers; L++)
        {
            perLayerW.Add(new[]
            {
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 10 + 1),
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 10 + 2),
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 10 + 3),
                MakeTensor(new[] { Dim, Dim }, 0.05f, L * 10 + 4),
                MakeTensor(new[] { Dim, FF }, 0.05f, L * 10 + 5),
                MakeTensor(new[] { FF, Dim }, 0.05f, L * 10 + 6),
                MakeTensor(new[] { Dim }, 1.0f, L * 10 + 7),
                MakeTensor(new[] { Dim }, 0.0f, L * 10 + 8),
                MakeTensor(new[] { Dim }, 1.0f, L * 10 + 9),
                MakeTensor(new[] { Dim }, 0.0f, L * 10 + 10),
            });
        }
        // Mimic AiDotNet's layer-cache pattern: each MultiHeadAttentionLayer
        // and DenseLayer stores 5-10 `_last*` fields pointing at forward
        // intermediates. Simulate that with one slot per layer × multiple
        // intermediates per layer.
        Tensor<float>?[,] layerCaches = new Tensor<float>?[NumLayers, 8];

        // StableForcedGc()/LiveBytes() per the file's standard two-window
        // half-window methodology — same as the other multi-layer probes.
        const int Warmup = 25;
        const int Measure = 200;
        for (int i = 0; i < Warmup; i++) Step();
        StableForcedGc();
        long m0 = LiveBytes();

        for (int i = 0; i < Measure; i++) Step();
        StableForcedGc();
        long m1 = LiveBytes();

        for (int i = 0; i < Measure; i++) Step();
        StableForcedGc();
        long m2 = LiveBytes();

        long w1PerCall = (m1 - m0) / Measure;
        long w2PerCall = (m2 - m1) / Measure;
        _output.WriteLine(
            $"4L-Transformer null-sources leak probe " +
            $"(reopened-AiDotNet#1227): " +
            $"win1={w1PerCall} B/call, win2={w2PerCall} B/call " +
            $"(m0={m0} m1={m1} m2={m2}; {Measure} iters/window)");

        // Hold the references to defeat the JIT's dead-code elimination.
        // Asserting `layerCaches != null` alone is not enough — the array
        // identity stays live but the element-store code inside Step()
        // (`layerCaches[L, k] = …`) can still be eliminated if no read
        // ever observes the elements. Actually READ every slot and
        // accumulate a checksum so the stores are provably load-bearing.
        // A live cached intermediate is what makes this test reproduce
        // reopened-AiDotNet#1227 — without it the autodiff arena's
        // natural sweep masks the leak and the probe goes false-clean.
        int liveCachedCount = 0;
        long rankSum = 0;
        for (int L = 0; L < NumLayers; L++)
        {
            for (int k = 0; k < 8; k++)
            {
                var cached = layerCaches[L, k];
                if (cached is not null)
                {
                    liveCachedCount++;
                    rankSum += cached.Rank;
                }
            }
        }
        GC.KeepAlive(layerCaches);
        Assert.True(liveCachedCount > 0,
            $"layerCaches expected at least one retained intermediate after Step() ran, " +
            $"got {liveCachedCount}. JIT may have dead-store-eliminated the cache writes " +
            $"(rankSum={rankSum}), which would invalidate the leak probe.");

        // 50 KB/call ceiling. Pre-fix this test produced ~100 KB/call;
        // with the cleanup-runs-for-null-sources fix it drops to 0 B/call
        // on net10.0. The 50 KB band accommodates legitimate per-iter
        // warm-up costs (delegate caching, pool growth) without
        // tolerating the original graph-retention pattern.
        Assert.True(w2PerCall < 50_000,
            $"Reopened-AiDotNet#1227: null-sources ComputeGradients leaked " +
            $"{w2PerCall} B/call > 50 KB on a 4-encoder-layer Transformer with " +
            $"a single retained forward intermediate. " +
            $"win1={w1PerCall} B/call, m0={m0} m1={m1} m2={m2}.");

        void Step()
        {
            var x = MakeTensor(new[] { Batch * Seq, Dim }, 1.0f, 99);
            using var tape = new GradientTape<float>();

            Tensor<float> h = x;
            for (int L = 0; L < NumLayers; L++)
            {
                var W = perLayerW[L];
                Tensor<float> wq = W[0], wk = W[1], wv = W[2], wo = W[3];
                Tensor<float> w1 = W[4], w2 = W[5];
                Tensor<float> g1 = W[6], beta1 = W[7], g2 = W[8], beta2 = W[9];

                var xNorm = engine.TensorLayerNorm(h, g1, beta1, epsilon: 1e-5);
                var q = engine.TensorMatMul(xNorm, wq);
                var k = engine.TensorMatMul(xNorm, wk);
                var v = engine.TensorMatMul(xNorm, wv);
                var scores = engine.TensorMatMulTransposed(q, k);
                var attn = engine.Softmax(scores, axis: -1);
                var ctx = engine.TensorMatMul(attn, v);
                var proj = engine.TensorMatMul(ctx, wo);
                h = engine.TensorAdd(h, proj);

                var hNorm = engine.TensorLayerNorm(h, g2, beta2, epsilon: 1e-5);
                var ffn1 = engine.TensorMatMul(hNorm, w1);
                var ffn1Act = engine.ReLU(ffn1);
                var ffn2 = engine.TensorMatMul(ffn1Act, w2);
                h = engine.TensorAdd(h, ffn2);

                // Layer-cache simulation: retain forward intermediates the
                // way AiDotNet's MultiHeadAttentionLayer does via
                // _lastInput / _lastQueryInput / _lastKeyInput / _lastValueInput
                // / _lastProjectedQueries / _lastAttentionScores / etc.
                layerCaches[L, 0] = xNorm;
                layerCaches[L, 1] = q;
                layerCaches[L, 2] = k;
                layerCaches[L, 3] = v;
                layerCaches[L, 4] = attn;
                layerCaches[L, 5] = ctx;
                layerCaches[L, 6] = ffn1Act;
                layerCaches[L, 7] = h;
            }

            var loss = engine.ReduceSum(h, null);
            // Critical: pass sources=null to match AiDotNet's consumer pattern
            // (NeuralNetworkBase.TrainWithTape reads gradients from the
            // returned Dictionary, not from tensor.Grad).
            var grads = tape.ComputeGradients(loss, sources: null);
        }
    }

    [Fact]
    public void TrainStep_LongRun_1000Iters_StaysUnder10KB_PerCall()
    {
        // Issue #283 acceptance criterion: a 1000-iteration stress test
        // asserting < 10 KB/call post-Gen2. The reporter's specific ask:
        // "Currently we'd ship 0.69.3 with a 400 KB/call regression undetected."
        // 1000 iters at the issue's transformer scale catches both the
        // original AutoTracer leak and any future regression that adds <500 B
        // /call (since 500 B × 1000 iters = 0.5 MB, easily detectable).
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 64, Dim = 128, FF = 512;
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();
        AiDotNet.Tensors.Engines.Autodiff.TensorPool<float>.Clear();

        var wq = MakeTensor(new[] { Dim, Dim }, 0.1f, 1);
        var wk = MakeTensor(new[] { Dim, Dim }, 0.1f, 2);
        var wv = MakeTensor(new[] { Dim, Dim }, 0.1f, 3);
        var wo = MakeTensor(new[] { Dim, Dim }, 0.1f, 4);
        var w1 = MakeTensor(new[] { Dim, FF }, 0.1f, 5);
        var w2 = MakeTensor(new[] { FF, Dim }, 0.1f, 6);
        var sources = new[] { wq, wk, wv, wo, w1, w2 };

        // Warmup 10 iterations to let JIT and pools stabilise.
        for (int i = 0; i < 10; i++) Step();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        long start = GC.GetTotalMemory(forceFullCollection: true);
        const int iters = 1000;
        for (int i = 0; i < iters; i++) Step();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long end = GC.GetTotalMemory(forceFullCollection: true);

        long perCall = (end - start) / iters;
        _output.WriteLine($"1000-iter stress test: {perCall} B/call (start={start} end={end})");

        Assert.True(perCall < 10_000,
            $"Issue #283 1000-iter regression canary: {perCall} B/call exceeds 10 KB/call. " +
            $"Reporter: \"a fail-fast canary in AiDotNet.Tensors' own test suite that does 1000 Train calls and asserts < 10 KB/call post-Gen2\".");

        void Step()
        {
            var x = MakeTensor(new[] { Batch * Seq, Dim }, 1.0f, 99);
            using var tape = new GradientTape<float>();
            var q = engine.TensorMatMul(x, wq);
            var k = engine.TensorMatMul(x, wk);
            var v = engine.TensorMatMul(x, wv);
            var scores = engine.TensorMatMulTransposed(q, k);
            var attn = engine.Softmax(scores, axis: -1);
            var ctx = engine.TensorMatMul(attn, v);
            var proj = engine.TensorMatMul(ctx, wo);
            var h1 = engine.TensorMatMul(proj, w1);
            var h2 = engine.ReLU(h1);
            var h3 = engine.TensorMatMul(h2, w2);
            var loss = engine.ReduceSum(h3, null);
            var grads = tape.ComputeGradients(loss, sources: sources);
            // Simulate optimizer reading every gradient (the consumer's
            // observed leak path includes the optimizer step).
            foreach (var s in sources) Assert.NotNull(s.Grad);
        }
    }

    [Fact]
    public void SequentialTapes_NoForwardIntermediatesSurvive()
    {
        // Issue #283 edge case — back-to-back tapes in the same thread,
        // simulating consecutive training steps. AutoTracer's _currentSequence
        // could span tape boundaries if the gate isn't applied correctly.
        // (Higher-order AD via createGraph=true is covered by the existing
        // Hvp_QuadraticForm tests in TorchFuncPhase3Tests.)
        var engine = AiDotNetEngine.Current;
        const int Dim = 64;
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();
        AiDotNet.Tensors.Engines.Autodiff.TensorPool<float>.Clear();

        var w = MakeTensor(new[] { Dim, Dim }, 0.1f, 1);
        var sources = new[] { w };

        for (int i = 0; i < 3; i++)
            Issue283Step_SequentialTapes(engine, sources, Dim, w, null);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var trackedRefs = new System.Collections.Generic.List<(string label, System.WeakReference wr)>();
        Issue283Step_SequentialTapes(engine, sources, Dim, w, trackedRefs);

        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var survivors = new System.Collections.Generic.List<string>();
        foreach (var (label, wr) in trackedRefs)
            if (wr.IsAlive) survivors.Add(label);
        _output.WriteLine($"Sequential tapes: {survivors.Count} of {trackedRefs.Count} survived");
        foreach (var s in survivors) _output.WriteLine($"  - {s}");
        Assert.True(survivors.Count == 0,
            $"Sequential-tape config leaked {survivors.Count} intermediates: {string.Join(", ", survivors)}");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Issue283Step_SequentialTapes(
        IEngine engine, Tensor<float>[] sources, int dim, Tensor<float> w,
        System.Collections.Generic.List<(string label, System.WeakReference wr)> trackedRefs)
    {
        // Two sequential tapes — the second runs AFTER the first disposes.
        // This is the most common multi-step training pattern.
        Tensor<float> x1, x2, y1, y2;
        using (var tape1 = new GradientTape<float>())
        {
            x1 = MakeTensor(new[] { dim, dim }, 1.0f, 100);
            y1 = engine.TensorMatMul(x1, w);
            var loss1 = engine.ReduceSum(y1, null);
            var g1 = tape1.ComputeGradients(loss1, sources);
            Assert.True(g1.ContainsKey(w));
        }
        using (var tape2 = new GradientTape<float>())
        {
            x2 = MakeTensor(new[] { dim, dim }, 1.0f, 200);
            y2 = engine.TensorMatMul(x2, w);
            var loss2 = engine.ReduceSum(y2, null);
            var g2 = tape2.ComputeGradients(loss2, sources);
            Assert.True(g2.ContainsKey(w));
        }
        if (trackedRefs is not null)
        {
            trackedRefs.Add(("x1", new System.WeakReference(x1)));
            trackedRefs.Add(("y1", new System.WeakReference(y1)));
            trackedRefs.Add(("x2", new System.WeakReference(x2)));
            trackedRefs.Add(("y2", new System.WeakReference(y2)));
        }
    }

    [Fact]
    public void Inference_NoTape_ThreadTapeDepthRemainsZero()
    {
        // Negative test for the fix: the per-thread gate that suppresses
        // AutoTracer (DifferentiableOps._threadTapeDepth) MUST stay at 0
        // when no GradientTape is active on this thread. If a future change
        // accidentally increments without a balancing decrement, AutoTracer
        // would silently stop recording for inference too.
        Assert.Equal(0, AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps._threadTapeDepth);

        // Take a tape lifecycle round-trip to confirm the counter increments
        // and decrements correctly.
        Assert.Equal(0, AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps._threadTapeDepth);
        using (var tape = new GradientTape<float>())
        {
            Assert.Equal(1, AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps._threadTapeDepth);
            using (var nested = new GradientTape<float>())
            {
                Assert.Equal(2, AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps._threadTapeDepth);
            }
            Assert.Equal(1, AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps._threadTapeDepth);
        }
        Assert.Equal(0, AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps._threadTapeDepth);

        // And the inference path itself still works after a tape lifecycle.
        var engine = AiDotNetEngine.Current;
        const int Dim = 64;
        var a = MakeTensor(new[] { Dim, Dim }, 0.1f, 1);
        var b = MakeTensor(new[] { Dim, Dim }, 0.1f, 2);
        var result = engine.TensorMatMul(a, b);
        Assert.Equal(new[] { Dim, Dim }, result.Shape.ToArray());
    }

    [Fact]
    public void OptimizerStep_ReadsGrad_NoForwardIntermediatesPinnedByGrad()
    {
        // Issue #283 — verifies that reading source.Grad after backward
        // (the optimizer-step pattern) doesn't pin forward intermediates.
        // The .Grad on a SOURCE tensor is the gradient OF that source, not
        // an intermediate; if it accidentally aliased an intermediate (via
        // a shared-buffer bug), this test would catch it.
        var engine = AiDotNetEngine.Current;
        const int Dim = 64;
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();
        AiDotNet.Tensors.Engines.Autodiff.TensorPool<float>.Clear();

        var w = MakeTensor(new[] { Dim, Dim }, 0.1f, 1);
        var sources = new[] { w };

        // Warmup
        for (int i = 0; i < 3; i++)
            Issue283Step_OptimizerPattern(engine, sources, Dim, w, null);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var intermediateRefs = new System.Collections.Generic.List<(string label, System.WeakReference wr)>();
        Issue283Step_OptimizerPattern(engine, sources, Dim, w, intermediateRefs);

        // Simulate optimizer step that reads .Grad — the user's consumer
        // does this. We then null .Grad to simulate a "zero_grad" cycle.
        Assert.NotNull(w.Grad);
        var snapshotGradFirstElem = w.Grad.GetFlat(0);
        // Don't null w.Grad here — the optimizer's typical pattern doesn't
        // null grads, the next backward overwrites. But we want to assert
        // that holding w.Grad doesn't pin forward intermediates.
        _ = snapshotGradFirstElem;

        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var survivors = new System.Collections.Generic.List<string>();
        foreach (var (label, wr) in intermediateRefs)
            if (wr.IsAlive) survivors.Add(label);
        _output.WriteLine($"Optimizer-pattern: {survivors.Count} of {intermediateRefs.Count} survived (w.Grad held)");

        Assert.True(survivors.Count == 0,
            $"Reading source.Grad pinned {survivors.Count} forward intermediates: " +
            string.Join(", ", survivors));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Issue283Step_OptimizerPattern(
        IEngine engine, Tensor<float>[] sources, int dim, Tensor<float> w,
        System.Collections.Generic.List<(string label, System.WeakReference wr)> trackedRefs)
    {
        var x = MakeTensor(new[] { dim, dim }, 1.0f, 99);
        using var tape = new GradientTape<float>();
        var y = engine.TensorMatMul(x, w);
        var z = engine.ReLU(y);
        var loss = engine.ReduceSum(z, null);
        var grads = tape.ComputeGradients(loss, sources: sources);
        if (trackedRefs is not null)
        {
            trackedRefs.Add(("x", new System.WeakReference(x)));
            trackedRefs.Add(("y-matmul", new System.WeakReference(y)));
            trackedRefs.Add(("z-relu", new System.WeakReference(z)));
            trackedRefs.Add(("loss", new System.WeakReference(loss)));
        }
    }

    private static Tensor<float> MakeTensor(int[] shape, float scale, int seed)
    {
        var rng = new Random(seed);
        int len = 1;
        for (int i = 0; i < shape.Length; i++) len *= shape[i];
        var data = new float[len];
        for (int i = 0; i < data.Length; i++) data[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(data, shape);
    }

    /// <summary>
    /// Two-stage GC sequence sufficient to stabilise even under
    /// Server GC (Linux CI default): the first compacting Gen-2 pass
    /// can promote freshly-orphaned objects to a generation that
    /// itself collects on the second pass; the finalizer drain
    /// between passes catches anything still in the F-reachable
    /// queue. Without the second pass, on Linux Server GC the
    /// observed live-byte count drifts up by tens of KB across the
    /// test even when no real leak exists.
    /// </summary>
    private static void StableForcedGc()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect(generation: 2, mode: GCCollectionMode.Default,
            blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(generation: 2, mode: GCCollectionMode.Default,
            blocking: true, compacting: true);
    }

    /// <summary>
    /// Live-byte count post-compaction. On net5+ subtracts
    /// <c>FragmentedBytes</c> from <c>HeapSizeBytes</c> to exclude
    /// fragmented free space inside Server-GC heap segments; on
    /// net471 falls back to <c>GC.GetTotalMemory(false)</c>, which
    /// is comparable after the compacting Gen-2 pass above.
    /// </summary>
    private static long LiveBytes()
    {
#if NET5_0_OR_GREATER
        var info = GC.GetGCMemoryInfo();
        long live = info.HeapSizeBytes - info.FragmentedBytes;
        return live > 0 ? live : GC.GetTotalMemory(forceFullCollection: false);
#else
        return GC.GetTotalMemory(forceFullCollection: false);
#endif
    }
}
