// Copyright (c) AiDotNet. All rights reserved.
// Issue #279 — GradientTape lifecycle leak repro.

#nullable disable

using System;
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

    [Fact]
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
    public void TrainStep_TransformerBlock_DoesNotLeak()
    {
        var engine = AiDotNetEngine.Current;
        const int Batch = 1, Seq = 16, Dim = 64, Heads = 4, FF = 128;
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
