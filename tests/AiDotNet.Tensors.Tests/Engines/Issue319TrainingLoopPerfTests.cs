// Copyright (c) AiDotNet. All rights reserved.
// Issue #319 — diagnostic harness for the *training* hot path
// (forward + backward + optimizer), not the forward-only block
// covered by Issue319TransformerBlockPerfTests.
//
// Why this exists:
//   The consumer-side ClipPerfHarness reports ViT-Base train at
//   ~3242 ms/iter (after the BLAS-default-on / grain-size /
//   specialized-matmul wins in PR #321). The forward-only Tensors
//   integration test sits at ~51 ms/iter — i.e. the remaining ~3000
//   ms/iter is in:
//     1. Tape recording overhead (RecordUnary/Binary + GradNode alloc)
//     2. Backward op dispatch (ComputeGradients walks tape entries)
//     3. Optimizer step (parameter update + Adam moments)
//     4. Per-iter allocator pressure (each Record alloc → Gen0 churn)
//
// This test exercises a representative subset of the training hot
// path on shapes that match ViT-Base patch [197, 768], times each
// phase separately, and reports GC allocation bytes per iter so the
// overhead breakdown is visible.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue319TrainingLoopPerfTests
{
    private readonly ITestOutputHelper _output;
    public Issue319TrainingLoopPerfTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Times one full training iteration on a 4-layer MLP at ViT-Base
    /// hidden dimensions: each layer is matmul + bias-add + GELU +
    /// LayerNorm. Reports forward, backward, and optimizer time
    /// separately so the dominant phase is visible.
    /// </summary>
    /// <remarks>
    /// Uses 4 layers (not 12) to keep test wall-clock low while still
    /// exercising the same op chain. The per-op cost dominates
    /// regardless of layer count; multiply by 3 for ViT-Base estimate.
    /// </remarks>
    [Fact]
    public void TrainingStep_PhaseBreakdown_ShowsWhereTimeGoes()
    {
        const int Hidden = 768;
        const int Seq = 197;
        const int Layers = 4;
        const int WarmupIters = 3;
        const int MeasureIters = 10;

        var engine = new CpuEngine();
        var rng = new Random(1);

        var x = MakeTensor(new[] { Seq, Hidden }, rng, 1.0);
        var weights = new Tensor<float>[Layers];
        var biases = new Tensor<float>[Layers];
        var gammas = new Tensor<float>[Layers];
        var betas = new Tensor<float>[Layers];
        for (int l = 0; l < Layers; l++)
        {
            weights[l] = MakeTensor(new[] { Hidden, Hidden }, rng, 0.02);
            biases[l] = MakeTensor(new[] { Hidden }, rng, 0.01);
            gammas[l] = MakeOnes(new[] { Hidden });
            betas[l] = MakeTensor(new[] { Hidden }, rng, 0.0);
        }

        // Warmup — JIT, allocator, arena caches settle.
        for (int w = 0; w < WarmupIters; w++)
        {
            RunTrainStep(engine, x, weights, biases, gammas, betas);
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Per-phase timing. Forward + backward + optimizer reported
        // separately by re-running with sub-phase Stopwatches.
        double totalMs = 0, fwdMs = 0, bwdMs = 0, optMs = 0;
#if NET5_0_OR_GREATER
        long allocStart = GC.GetTotalAllocatedBytes(precise: true);
#else
        long allocStart = GC.GetTotalMemory(forceFullCollection: false);
#endif
        int gen0Start = GC.CollectionCount(0);
        int gen1Start = GC.CollectionCount(1);
        int gen2Start = GC.CollectionCount(2);

        for (int it = 0; it < MeasureIters; it++)
        {
            var phase = RunTrainStepTimed(engine, x, weights, biases, gammas, betas);
            totalMs += phase.totalMs;
            fwdMs += phase.fwdMs;
            bwdMs += phase.bwdMs;
            optMs += phase.optMs;
        }

#if NET5_0_OR_GREATER
        long allocEnd = GC.GetTotalAllocatedBytes(precise: true);
#else
        long allocEnd = GC.GetTotalMemory(forceFullCollection: false);
#endif
        int gen0End = GC.CollectionCount(0);
        int gen1End = GC.CollectionCount(1);
        int gen2End = GC.CollectionCount(2);

        double avgTotal = totalMs / MeasureIters;
        double avgFwd = fwdMs / MeasureIters;
        double avgBwd = bwdMs / MeasureIters;
        double avgOpt = optMs / MeasureIters;
        double allocPerIterKb = (allocEnd - allocStart) / 1024.0 / MeasureIters;

        _output.WriteLine($"Training-step phase breakdown @ ViT-Base shape [{Seq}, {Hidden}], {Layers} layers, {MeasureIters} iters:");
        _output.WriteLine($"  Total          : {avgTotal,8:F2} ms/iter");
        _output.WriteLine($"  Forward+tape   : {avgFwd,8:F2} ms/iter ({100.0 * avgFwd / avgTotal:F1}%)");
        _output.WriteLine($"  Backward       : {avgBwd,8:F2} ms/iter ({100.0 * avgBwd / avgTotal:F1}%)");
        _output.WriteLine($"  Optimizer      : {avgOpt,8:F2} ms/iter ({100.0 * avgOpt / avgTotal:F1}%)");
        _output.WriteLine($"  Allocs         : {allocPerIterKb,8:F1} KB/iter");
        _output.WriteLine($"  GC gen0/1/2    : {gen0End - gen0Start} / {gen1End - gen1Start} / {gen2End - gen2Start} (total over {MeasureIters} iters)");

        // Extrapolate to ViT-Base 12-layer estimate
        double vitBaseEst = avgTotal * (12.0 / Layers);
        _output.WriteLine($"  Extrapolated to 12-layer ViT-Base: {vitBaseEst:F1} ms/iter (linear scaling assumption)");

        // No assertion — this is a diagnostic. Future regression-gating
        // can compare current numbers to a recorded baseline.
        Assert.True(avgTotal > 0, "Training step measurement broken.");
    }

    /// <summary>
    /// Side-by-side comparison: same training step run with the naive
    /// optimizer pattern (TensorMultiplyScalar + TensorSubtractInPlace)
    /// vs <see cref="OptimizerKernels.SgdInPlace{T}"/>. Reports allocation
    /// delta so the win from the fused kernel is measurable in isolation.
    /// </summary>
    /// <remarks>
    /// Both paths are numerically equivalent (covered by
    /// OptimizerKernelsTests). This test isolates the GC-pressure win
    /// of the fused kernel on the actual training-loop hot path.
    /// </remarks>
    [Fact]
    public void TrainingStep_FusedSgdInPlace_ReducesAllocsVsNaiveOptimizer()
    {
        const int Hidden = 768;
        const int Seq = 197;
        const int Layers = 4;
        const int WarmupIters = 3;
        const int MeasureIters = 10;

        var engine = new CpuEngine();
        var rng = new Random(1);

        var x = MakeTensor(new[] { Seq, Hidden }, rng, 1.0);
        var weights = new Tensor<float>[Layers];
        var biases = new Tensor<float>[Layers];
        var gammas = new Tensor<float>[Layers];
        var betas = new Tensor<float>[Layers];
        for (int l = 0; l < Layers; l++)
        {
            weights[l] = MakeTensor(new[] { Hidden, Hidden }, rng, 0.02);
            biases[l] = MakeTensor(new[] { Hidden }, rng, 0.01);
            gammas[l] = MakeOnes(new[] { Hidden });
            betas[l] = MakeTensor(new[] { Hidden }, rng, 0.0);
        }

        // Warmup
        for (int w = 0; w < WarmupIters; w++) RunTrainStep(engine, x, weights, biases, gammas, betas, useFusedSgd: false);
        for (int w = 0; w < WarmupIters; w++) RunTrainStep(engine, x, weights, biases, gammas, betas, useFusedSgd: true);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Naive optimizer path.
        var swNaive = Stopwatch.StartNew();
#if NET5_0_OR_GREATER
        long naiveAllocStart = GC.GetTotalAllocatedBytes(precise: true);
#else
        long naiveAllocStart = GC.GetTotalMemory(forceFullCollection: false);
#endif
        for (int it = 0; it < MeasureIters; it++)
            RunTrainStep(engine, x, weights, biases, gammas, betas, useFusedSgd: false);
        swNaive.Stop();
#if NET5_0_OR_GREATER
        long naiveAllocEnd = GC.GetTotalAllocatedBytes(precise: true);
#else
        long naiveAllocEnd = GC.GetTotalMemory(forceFullCollection: false);
#endif

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Fused optimizer path.
        var swFused = Stopwatch.StartNew();
#if NET5_0_OR_GREATER
        long fusedAllocStart = GC.GetTotalAllocatedBytes(precise: true);
#else
        long fusedAllocStart = GC.GetTotalMemory(forceFullCollection: false);
#endif
        for (int it = 0; it < MeasureIters; it++)
            RunTrainStep(engine, x, weights, biases, gammas, betas, useFusedSgd: true);
        swFused.Stop();
#if NET5_0_OR_GREATER
        long fusedAllocEnd = GC.GetTotalAllocatedBytes(precise: true);
#else
        long fusedAllocEnd = GC.GetTotalMemory(forceFullCollection: false);
#endif

        double naiveMs = swNaive.Elapsed.TotalMilliseconds / MeasureIters;
        double fusedMs = swFused.Elapsed.TotalMilliseconds / MeasureIters;
        double naiveAllocKb = (naiveAllocEnd - naiveAllocStart) / 1024.0 / MeasureIters;
        double fusedAllocKb = (fusedAllocEnd - fusedAllocStart) / 1024.0 / MeasureIters;

        _output.WriteLine($"Naive vs FusedSgd training step ({MeasureIters} iters, {Layers} layers, [{Seq}, {Hidden}]):");
        _output.WriteLine($"  Naive (MultiplyScalar + SubtractInPlace) : {naiveMs,8:F2} ms/iter, {naiveAllocKb,9:F1} KB/iter alloc");
        _output.WriteLine($"  Fused OptimizerKernels.SgdInPlace        : {fusedMs,8:F2} ms/iter, {fusedAllocKb,9:F1} KB/iter alloc");
        _output.WriteLine($"  Alloc reduction                          : {naiveAllocKb - fusedAllocKb,9:F1} KB/iter saved");

        // Hardware variance is enormous on these workloads; the alloc
        // delta is the stable signal. The fused path should allocate
        // strictly less than the naive path — anything else means the
        // kernel got mis-routed.
        Assert.True(fusedAllocKb <= naiveAllocKb,
            $"Fused SGD allocated more than naive: fused={fusedAllocKb} KB/iter, naive={naiveAllocKb} KB/iter.");
    }

    private static (double totalMs, double fwdMs, double bwdMs, double optMs)
        RunTrainStepTimed(CpuEngine engine, Tensor<float> x,
            Tensor<float>[] weights, Tensor<float>[] biases,
            Tensor<float>[] gammas, Tensor<float>[] betas)
    {
        var sw = Stopwatch.StartNew();
        var swTotal = Stopwatch.StartNew();
        double fwd, bwd, opt;

        using (var tape = new GradientTape<float>())
        {
            var sources = new System.Collections.Generic.List<Tensor<float>>();
            for (int l = 0; l < weights.Length; l++)
            {
                sources.Add(weights[l]);
                sources.Add(biases[l]);
                sources.Add(gammas[l]);
                sources.Add(betas[l]);
            }

            sw.Restart();
            var current = x;
            for (int l = 0; l < weights.Length; l++)
            {
                var z = engine.TensorMatMul(current, weights[l]);
                var z2 = engine.TensorBroadcastAdd(z, biases[l]);
                var a = engine.GELU(z2);
                current = engine.LayerNorm(a, gammas[l], betas[l], 1e-5, out _, out _);
            }
            var loss = engine.ReduceSum(current, axes: null, keepDims: false);
            sw.Stop();
            fwd = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            var grads = tape.ComputeGradients(loss, sources);
            sw.Stop();
            bwd = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            // Simulate SGD update (the simplest optimizer step). Real
            // optimizers do more (Adam moments, weight decay) but SGD
            // captures the parameter-traversal + scalar-multiply cost
            // that's identical across optimizers.
            const float lr = 1e-4f;
            foreach (var src in sources)
            {
                if (!grads.TryGetValue(src, out var g)) continue;
                var update = engine.TensorMultiplyScalar(g, lr);
                engine.TensorSubtractInPlace(src, update);
            }
            sw.Stop();
            opt = sw.Elapsed.TotalMilliseconds;
        }

        swTotal.Stop();
        return (swTotal.Elapsed.TotalMilliseconds, fwd, bwd, opt);
    }

    private static void RunTrainStep(CpuEngine engine, Tensor<float> x,
        Tensor<float>[] weights, Tensor<float>[] biases,
        Tensor<float>[] gammas, Tensor<float>[] betas,
        bool useFusedSgd = false)
    {
        using var tape = new GradientTape<float>();
        var sources = new System.Collections.Generic.List<Tensor<float>>();
        for (int l = 0; l < weights.Length; l++)
        {
            sources.Add(weights[l]);
            sources.Add(biases[l]);
            sources.Add(gammas[l]);
            sources.Add(betas[l]);
        }

        var current = x;
        for (int l = 0; l < weights.Length; l++)
        {
            var z = engine.TensorMatMul(current, weights[l]);
            var z2 = engine.TensorBroadcastAdd(z, biases[l]);
            var a = engine.GELU(z2);
            current = engine.LayerNorm(a, gammas[l], betas[l], 1e-5, out _, out _);
        }
        var loss = engine.ReduceSum(current, axes: null, keepDims: false);
        var grads = tape.ComputeGradients(loss, sources);
        const float lr = 1e-4f;
        foreach (var src in sources)
        {
            if (!grads.TryGetValue(src, out var g)) continue;
            if (useFusedSgd)
            {
                OptimizerKernels.SgdInPlace(src, g, lr);
            }
            else
            {
                var update = engine.TensorMultiplyScalar(g, lr);
                engine.TensorSubtractInPlace(src, update);
            }
        }
    }

    private static Tensor<float> MakeTensor(int[] shape, Random rng, double scale)
    {
        int total = 1;
        foreach (var d in shape) total *= d;
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < total; i++) s[i] = (float)((rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static Tensor<float> MakeOnes(int[] shape)
    {
        int total = 1;
        foreach (var d in shape) total *= d;
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < total; i++) s[i] = 1f;
        return t;
    }
}
