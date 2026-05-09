// Copyright (c) AiDotNet. All rights reserved.
// Issue #319 — integration test that times a representative ViT
// transformer-block forward pass and reports per-op breakdown.
//
// This test is the deliverable the consumer asked for: a
// reproducible measurement of CPU wall-clock on the actual hot
// path, so any future migration of Parallel.For / PersistentParallel-
// Executor call sites can be validated against a concrete number
// rather than promises.
//
// What it measures:
//   * Per-op µs/call for TensorMatMul, TensorBroadcastAdd, GELU,
//     LayerNorm, Sigmoid at canonical ViT-Base patch shape
//     [197, 768] — the shapes that dominate every layer's wall-clock
//   * End-to-end ms/iter for a 12-layer block performing matmul
//     + bias-add + GELU + LayerNorm chains
//
// What it does NOT promise:
//   * That every Parallel.For call site has been migrated. There
//     are 240+ Parallel.For sites in CpuEngine.cs alone; PR #316
//     migrated 1 PersistentParallelExecutor site, this PR migrates
//     a few more. A comprehensive sweep is beyond a single PR.
//   * That ViT-Base CPU train ms/iter matches PyTorch. The original
//     issue cites 30-80 ms/iter for PyTorch fp32 vs 4183 ms/iter
//     for AiDotNet — that's a 50-100× gap. Closing it requires
//     work across many subsystems (tape recording overhead, ParallelFor
//     migration, op fusion, MKL parity), not just grain-size threshold
//     gating.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue319TransformerBlockPerfTests
{
    private readonly ITestOutputHelper _output;
    public Issue319TransformerBlockPerfTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Compares raw SIMD-kernel time vs engine-wrapper time per op.
    /// This tells us whether wall-clock is dominated by inner-loop
    /// SIMD compute or by wrapper overhead (allocation, recording,
    /// contiguity checks). Without this split, optimisation work is
    /// just guessing about where the time goes.
    /// </summary>
    [Fact]
    public void RawSimdVsEngineWrapper_FindsWhereTimeGoes()
    {
        const int Hidden = 768;
        const int Seq = 197;
        const int Calls = 200;
        var engine = new CpuEngine();
        var rng = new Random(7);

        var x = new Tensor<float>(new[] { Seq, Hidden });
        var xs = x.AsWritableSpan();
        for (int i = 0; i < xs.Length; i++) xs[i] = (float)((rng.NextDouble() - 0.5));

        // Pre-allocate output once so we measure the kernel only.
        var output = new Tensor<float>(new[] { Seq, Hidden });
        var xArr = (x.GetDataArray() as float[])!;
        var oArr = (output.GetDataArray() as float[])!;

        // Warmup — JIT both paths.
        for (int i = 0; i < 3; i++) engine.GELU(x);

        // Engine-wrapped GELU.
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Calls; i++) engine.GELU(x);
        sw.Stop();
        double engineGeluUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Calls;

        // Raw SIMD GELU — bypass everything.
        sw.Restart();
        unsafe
        {
            fixed (float* p = xArr)
            fixed (float* o = oArr)
            {
                for (int i = 0; i < Calls; i++)
                    AiDotNet.Tensors.Engines.Simd.SimdKernels.GELUUnsafe(p, o, x.Length);
            }
        }
        sw.Stop();
        double rawGeluUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Calls;

        _output.WriteLine($"GELU on [{Seq}, {Hidden}] = {x.Length:N0} elements ({Calls} calls):");
        _output.WriteLine($"  engine.GELU(x)        : {engineGeluUs,8:F1} µs/call");
        _output.WriteLine($"  raw SimdKernels.GELU  : {rawGeluUs,8:F1} µs/call");
        _output.WriteLine($"  wrapper overhead      : {engineGeluUs - rawGeluUs,8:F1} µs/call "
            + $"({(engineGeluUs - rawGeluUs) / engineGeluUs * 100,6:F1}% of total)");
    }

    /// <summary>
    /// Per-op timing breakdown for the canonical ViT-Base patch
    /// shape <c>[197, 768]</c>. Reports µs/call for each op so a
    /// future migration round can be measured against current
    /// numbers. The aggregate ms/iter assertion only catches gross
    /// regressions; the per-op output is the diagnostic.
    /// </summary>
    [Fact]
    public void TransformerBlockHotPath_PerOpTimingsAndBudget()
    {
        const int Hidden = 768;
        const int Seq = 197;
        const int Layers = 12;
        const int Iters = 30;

        var engine = new CpuEngine();
        var rng = new Random(1);

        var x = MakeTensor(new[] { Seq, Hidden }, rng, 1.0);
        var weight = MakeTensor(new[] { Hidden, Hidden }, rng, 0.01);
        var bias = MakeTensor(new[] { Hidden }, rng, 0.01);
        var gamma = new Tensor<float>(new[] { Hidden });
        var beta = new Tensor<float>(new[] { Hidden });
        var gs = gamma.AsWritableSpan();
        var bts = beta.AsWritableSpan();
        for (int i = 0; i < Hidden; i++) { gs[i] = 1f; bts[i] = 0f; }

        // Warmup — JIT, allocator, ThreadLocal arena settle.
        for (int w = 0; w < 5; w++) RunFullBlock(engine, x, weight, bias, gamma, beta, Layers);

        // Per-op breakdown — same call count as one full benchmark
        // run so total-time numbers are comparable.
        int totalCalls = Iters * Layers;
        long matmulMs = TimeOp(totalCalls, () => engine.TensorMatMul(x, weight));
        long biasAddMs = TimeOp(totalCalls, () => engine.TensorBroadcastAdd(x, bias));
        long geluMs = TimeOp(totalCalls, () => engine.GELU(x));
        long layerNormMs = TimeOp(totalCalls, () => engine.LayerNorm(x, gamma, beta, 1e-5, out _, out _));

        var sw = Stopwatch.StartNew();
        for (int it = 0; it < Iters; it++) RunFullBlock(engine, x, weight, bias, gamma, beta, Layers);
        sw.Stop();
        double msPerIter = sw.Elapsed.TotalMilliseconds / Iters;

        double matmulUs = (double)matmulMs * 1000.0 / totalCalls;
        double biasAddUs = (double)biasAddMs * 1000.0 / totalCalls;
        double geluUs = (double)geluMs * 1000.0 / totalCalls;
        double layerNormUs = (double)layerNormMs * 1000.0 / totalCalls;

        _output.WriteLine($"Per-op µs/call at ViT-Base shape [{Seq}, {Hidden}] "
            + $"(over {totalCalls} calls each):");
        _output.WriteLine($"  TensorMatMul       : {matmulUs,8:F1} µs/call (total {matmulMs} ms)");
        _output.WriteLine($"  TensorBroadcastAdd : {biasAddUs,8:F1} µs/call (total {biasAddMs} ms)");
        _output.WriteLine($"  GELU               : {geluUs,8:F1} µs/call (total {geluMs} ms)");
        _output.WriteLine($"  LayerNorm          : {layerNormUs,8:F1} µs/call (total {layerNormMs} ms)");
        _output.WriteLine($"Full block chain     : {msPerIter:F2} ms/iter "
            + $"({Layers} layers × (matmul + bias + GELU + LayerNorm) over {Iters} iters)");

        // Budget: 1500 ms/iter on the full 12-layer block. The
        // 0.75.3 baseline (issue #319) was 4183 ms/iter for the
        // FULL ViT-Base train (forward + backward + optimizer);
        // forward-only is roughly 1/6th of that. Anything > 1500
        // ms/iter on this benchmark indicates a major regression
        // in the matmul or per-op overhead path. The number is
        // documented in the failure message so the next migration
        // round can tighten it.
        Assert.True(msPerIter < 1500.0,
            $"#319 — full transformer-block forward exceeded 1500 ms/iter budget: "
            + $"observed {msPerIter:F2} ms/iter. Per-op timings: matmul={matmulUs:F1} µs, "
            + $"bias={biasAddUs:F1} µs, GELU={geluUs:F1} µs, LayerNorm={layerNormUs:F1} µs. "
            + "A regression here indicates either a) a kernel re-introduced parallel "
            + "dispatch on small ops, or b) the matmul kernel itself regressed.");
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

    private static long TimeOp(int n, Action body)
    {
        // Warmup specifically for this op.
        for (int i = 0; i < 3; i++) body();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < n; i++) body();
        sw.Stop();
        return sw.ElapsedMilliseconds;
    }

    private static void RunFullBlock(CpuEngine engine,
        Tensor<float> x, Tensor<float> weight, Tensor<float> bias,
        Tensor<float> gamma, Tensor<float> beta, int layers)
    {
        var current = x;
        for (int l = 0; l < layers; l++)
        {
            // Per-layer: matmul + bias-add + GELU + LayerNorm. The
            // matmul dominates compute; the others test that small-
            // op dispatch overhead doesn't accumulate.
            var z = engine.TensorMatMul(current, weight);
            var z2 = engine.TensorBroadcastAdd(z, bias);
            var a = engine.GELU(z2);
            current = engine.LayerNorm(a, gamma, beta, 1e-5, out _, out _);
        }
    }
}
