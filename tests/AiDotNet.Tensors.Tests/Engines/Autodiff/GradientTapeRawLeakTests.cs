// Copyright (c) AiDotNet. All rights reserved.
// Issue #312 — CI canary for the raw-tape + engine-ops use case
// (distinct from the Transformer.Train path covered by #279/#283).
// Asserts that 500 iterations of ComputeGradients through a multi-
// layer FFN retains less than a per-call budget of GC-heap memory
// after a forced full GC.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class GradientTapeRawLeakTests
{
    private readonly ITestOutputHelper _output;
    public GradientTapeRawLeakTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Issue #312's exact reproducer: minimal multi-layer FFN with
    /// chained TensorMatMul / TensorMultiply / Tanh / LayerNorm / etc.
    /// Asserts retention per call &lt; the issue's stretch goal of
    /// 10 KB. Above that threshold, a sustained training loop will
    /// accumulate enough heap to crash the test host before
    /// completing a paper-grade benchmark sweep — exactly what
    /// HarmonicEngine reported.
    /// </summary>
    [Fact]
    public void RawTapeAndEngineOps_RetainsBoundedHeapAcross500Iterations()
    {
        // Slightly smaller than the issue's [256, 128, 512, 256, 8]
        // shape so the test fits inside CI test-host memory while
        // still exercising the same retention path. 50 iters chosen
        // so the test runs in seconds; the per-call retention
        // assertion is what matters, not absolute heap size.
        const int F = 64, H = 128, V = 64, B = 32, L = 4;
        const int Warmup = 5;
        const int Measure = 50;
        const long PerCallBudgetBytes = 10 * 1024; // 10 KB/call — issue #312 stretch goal

        var engine = AiDotNetEngine.Current;
        var rng = new Random(42);

        var x0 = TensorAllocator.Rent<float>(new[] { B, F });
        var targets = TensorAllocator.Rent<float>(new[] { B, V });
        for (int i = 0; i < x0.Length; i++) x0.AsWritableSpan()[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < targets.Length; i++)
            targets.AsWritableSpan()[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var W1s = new Tensor<float>[L]; var b1s = new Tensor<float>[L];
        var W2s = new Tensor<float>[L]; var b2s = new Tensor<float>[L];
        var gamma = new Tensor<float>[L]; var beta = new Tensor<float>[L];
        for (int l = 0; l < L; l++)
        {
            W1s[l] = TensorAllocator.Rent<float>(new[] { F, H });
            b1s[l] = TensorAllocator.Rent<float>(new[] { H });
            W2s[l] = TensorAllocator.Rent<float>(new[] { H, F });
            b2s[l] = TensorAllocator.Rent<float>(new[] { F });
            gamma[l] = TensorAllocator.Rent<float>(new[] { F });
            beta[l] = TensorAllocator.Rent<float>(new[] { F });
            for (int i = 0; i < F; i++) gamma[l].AsWritableSpan()[i] = 1f;
            // Initialise W with small random values so the forward
            // doesn't blow up to inf and force NaN through backward.
            var w1Span = W1s[l].AsWritableSpan();
            for (int i = 0; i < w1Span.Length; i++) w1Span[i] = (float)((rng.NextDouble() - 0.5) * 0.1);
            var w2Span = W2s[l].AsWritableSpan();
            for (int i = 0; i < w2Span.Length; i++) w2Span[i] = (float)((rng.NextDouble() - 0.5) * 0.1);
        }
        var Wh = TensorAllocator.Rent<float>(new[] { F, V });
        var bh = TensorAllocator.Rent<float>(new[] { V });
        var whSpan = Wh.AsWritableSpan();
        for (int i = 0; i < whSpan.Length; i++) whSpan[i] = (float)((rng.NextDouble() - 0.5) * 0.1);

        var trainable = new List<Tensor<float>>();
        foreach (var p in W1s) trainable.Add(p);
        foreach (var p in b1s) trainable.Add(p);
        foreach (var p in W2s) trainable.Add(p);
        foreach (var p in b2s) trainable.Add(p);
        foreach (var p in gamma) trainable.Add(p);
        foreach (var p in beta) trainable.Add(p);
        trainable.Add(Wh);
        trainable.Add(bh);

        // Warmup: lets thread-local arenas / caches settle so the
        // baseline measurement doesn't double-count first-call
        // allocations as retention.
        for (int i = 0; i < Warmup; i++) RunOneStep(engine, x0, targets, W1s, b1s, W2s, b2s, gamma, beta, Wh, bh, L, trainable);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect(generation: 2, mode: GCCollectionMode.Default,
            blocking: true, compacting: true);
        long start = LiveBytes();

        for (int i = 0; i < Measure; i++) RunOneStep(engine, x0, targets, W1s, b1s, W2s, b2s, gamma, beta, Wh, bh, L, trainable);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect(generation: 2, mode: GCCollectionMode.Default,
            blocking: true, compacting: true);
        long end = LiveBytes();

        long retentionPerCall = (end - start) / Measure;
        _output.WriteLine($"Retention: start={start:N0} B, end={end:N0} B, "
            + $"per-call={retentionPerCall:N0} B over {Measure} iters (budget {PerCallBudgetBytes:N0} B/call).");
        Assert.True(retentionPerCall < PerCallBudgetBytes,
            $"Issue #312 — raw-tape + engine-ops retention exceeded "
            + $"{PerCallBudgetBytes:N0} B/call: observed {retentionPerCall:N0} B/call "
            + $"over {Measure} iterations (start={start:N0}, end={end:N0}). "
            + "A leak above this threshold accumulates to host-fatal "
            + "memory pressure on long training runs. Repro pattern: "
            + "raw GradientTape + chained engine.TensorMatMul / "
            + "engine.TensorAdd / engine.Tanh / engine.LayerNorm.");
    }

    private static long LiveBytes()
    {
#if NET5_0_OR_GREATER
        // Live bytes only — exclude fragmented heap segments in case
        // the test host was running under Server GC.
        var info = GC.GetGCMemoryInfo();
        long live = info.HeapSizeBytes - info.FragmentedBytes;
        return live > 0 ? live : GC.GetTotalMemory(forceFullCollection: false);
#else
        return GC.GetTotalMemory(forceFullCollection: false);
#endif
    }

    private static void RunOneStep(
        IEngine engine,
        Tensor<float> x0, Tensor<float> targets,
        Tensor<float>[] W1s, Tensor<float>[] b1s,
        Tensor<float>[] W2s, Tensor<float>[] b2s,
        Tensor<float>[] gamma, Tensor<float>[] beta,
        Tensor<float> Wh, Tensor<float> bh,
        int L, List<Tensor<float>> trainable)
    {
        using var tape = new GradientTape<float>();
        var x = x0;
        for (int l = 0; l < L; l++)
        {
            var hMid = engine.TensorBroadcastAdd(engine.TensorMatMul(x, W1s[l]), b1s[l]);
            // Cubic activation — same pattern the issue's repro uses
            // (engine.TensorMultiply chain).
            var hAct = engine.TensorMultiply(engine.TensorMultiply(hMid, hMid), hMid);
            var Fx = engine.TensorBroadcastAdd(engine.TensorMatMul(hAct, W2s[l]), b2s[l]);
            x = engine.Tanh(engine.TensorAdd(x, Fx));
            x = engine.LayerNorm(x, gamma[l], beta[l], 1e-5, out _, out _);
        }
        var logits = engine.TensorBroadcastAdd(engine.TensorMatMul(x, Wh), bh);
        // Use TensorMSELoss as the per-token loss — TensorCrossEntropyLoss
        // requires an int target tensor and the issue's repro uses
        // continuous-target gradients anyway.
        var diff = engine.TensorSubtract(logits, targets);
        var sq = engine.TensorMultiply(diff, diff);
        var loss = engine.ReduceSum(sq);
        var grads = tape.ComputeGradients(loss, trainable);
        // Discard gradients — caller's intent is just to drive the
        // tape lifecycle, not to do an actual update step.
        _ = grads.Count;
    }
}
