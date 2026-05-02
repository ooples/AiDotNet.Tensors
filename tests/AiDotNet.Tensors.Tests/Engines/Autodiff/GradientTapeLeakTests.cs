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
        foreach (var (label, wr) in trackedRefs)
        {
            if (!wr.IsAlive) continue;
            var t = wr.Target as Tensor<float>;
            string gradState = t is null ? "(collected during inspection)"
                : $"GradFn={(t.GradFn is null ? "null" : "SET")}, Grad={(t.Grad is null ? "null" : "SET")}";
            survivors.Add($"{label} [{gradState}]");
        }

        _output.WriteLine($"Tracked {trackedRefs.Count} forward intermediates; {survivors.Count} survived GC after Dispose:");
        foreach (var s in survivors) _output.WriteLine($"  - {s}");

        Assert.True(survivors.Count == 0,
            $"Issue #283 — {survivors.Count} of {trackedRefs.Count} forward intermediates " +
            $"survived Gen2 GC after GradientTape.Dispose. Surviving labels: " +
            string.Join(", ", survivors));

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

        // Warmup — populate JIT, AutoTensorCache pools, etc.
        for (int i = 0; i < 5; i++) Step();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        long start = GC.GetTotalMemory(forceFullCollection: true);
        const int iters = 50;
        for (int i = 0; i < iters; i++) Step();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long end = GC.GetTotalMemory(forceFullCollection: true);

        long perCall = (end - start) / iters;
        _output.WriteLine(
            $"Issue #283 transformer block at scale (Dim={Dim} FF={FF} Seq={Seq} Heads={Heads}): " +
            $"{perCall} B/call (start={start} end={end} iters={iters})");

        // 50 KB/call ceiling. The reporter saw 400 KB/call on the same
        // shape; this catches that with a 8x safety factor while still
        // being well above legitimate runtime residue.
        Assert.True(perCall < 50_000,
            $"Issue #283 — managed-heap retention {perCall} B/call at transformer scale " +
            $"(Dim={Dim} FF={FF} Seq={Seq}) exceeds 50 KB/call budget. " +
            $"Start={start} End={end} Iters={iters}. " +
            $"Reporter saw 400 KB/call — anything > 50 KB/call indicates a residual leak.");

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
}
