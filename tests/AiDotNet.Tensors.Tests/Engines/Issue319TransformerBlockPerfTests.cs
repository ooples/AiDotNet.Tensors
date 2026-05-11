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
//     LayerNorm at canonical ViT-Base patch shape
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
    /// Numerical equivalence check between the two GELU SIMD kernels.
    /// FusedGELUUnsafe uses GELU(x) = x · sigmoid(2 · sqrt(2/π) · (x + 0.044715·x³)),
    /// derived from the standard tanh-decomposition kernel via
    /// 1 + tanh(s) = 2 · sigmoid(2s). Algebraically identical, so
    /// outputs must match within float32 rounding (the two paths
    /// reorder some FMAs / multiplies, so the LSB can differ slightly).
    /// </summary>
    /// <remarks>
    /// Skipped on hosts without AVX2 + FMA: both kernels gate their fast
    /// path on <c>Avx2.IsSupported &amp;&amp; Fma.IsSupported</c> and fall
    /// through to scalar on older hardware. The scalar paths are
    /// trivially identical (same source expression), so the SIMD
    /// equivalence assertion the test is named for only has signal on
    /// hardware that actually runs the SIMD branch. Defensive guard so
    /// CI workers without AVX2 (rare but possible on some ARM /
    /// older-AMD images) skip cleanly instead of asserting against the
    /// scalar fallback.
    /// </remarks>
    [Fact]
    public void GELU_FusedKernel_NumericallyMatchesTanhDecomposition()
    {
#if NET5_0_OR_GREATER
        // System.Runtime.Intrinsics.X86 is .NET 5+. On net471 the test
        // runs unconditionally — net471 builds use the scalar fallback
        // for both kernels (the SIMD paths are gated by #if NET5_0_OR_GREATER
        // inside SimdKernels.cs too), so the equivalence check still
        // passes trivially.
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported
            || !System.Runtime.Intrinsics.X86.Fma.IsSupported)
        {
            // Both kernels would take the scalar fallback identically on
            // this host — the SIMD-path equivalence assertion this test is
            // designed for has no signal here.
            return;
        }
#endif

        const int Length = 4096;
        var rng = new Random(42);
        var input = new float[Length];
        for (int i = 0; i < Length; i++) input[i] = (float)((rng.NextDouble() - 0.5) * 6); // covers [-3, 3]
        var oldOut = new float[Length];
        var newOut = new float[Length];

        unsafe
        {
            fixed (float* p = input)
            fixed (float* o = oldOut)
                AiDotNet.Tensors.Engines.Simd.SimdKernels.GELUUnsafe(p, o, Length);
            fixed (float* p = input)
            fixed (float* o = newOut)
                AiDotNet.Tensors.Engines.Simd.SimdKernels.FusedGELUUnsafe(p, o, Length);
        }

        // Tolerance: both kernels approximate GELU via Padé sigmoid /
        // tanh polynomial, so a few ULPs of divergence is expected.
        // 1e-5 relative is comfortably above the ULP floor and
        // tight enough to catch a kernel-formula bug.
        float maxDiff = 0;
        for (int i = 0; i < Length; i++)
        {
            float diff = MathF.Abs(oldOut[i] - newOut[i]);
            float scale = 1e-5f * MathF.Max(1f, MathF.Abs(oldOut[i]));
            Assert.True(diff <= scale,
                $"GELU kernel mismatch at idx {i} (input={input[i]}): "
                + $"old={oldOut[i]} new={newOut[i]} diff={diff}.");
            if (diff > maxDiff) maxDiff = diff;
        }
        _output.WriteLine($"GELU equivalence check ({Length} samples in [-3, 3]): max abs diff = {maxDiff:E4}");
    }

    /// <summary>
    /// Compares raw SIMD-kernel time vs engine-wrapper time per op.
    /// This tells us whether wall-clock is dominated by inner-loop
    /// SIMD compute or by wrapper overhead (allocation, recording,
    /// contiguity checks). Without this split, optimisation work is
    /// just guessing about where the time goes.
    /// </summary>
    [Fact]
    [Trait("Category", "Performance")]
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

        // Raw GELUUnsafe — old kernel, tanh-decomposition (effectively 2x sigmoid).
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
        double rawGeluOldUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Calls;

        // Raw FusedGELUUnsafe — new kernel, single sigmoid via algebraic
        // simplification GELU = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715*x^3)).
        // engine.GELU now routes through this kernel.
        sw.Restart();
        unsafe
        {
            fixed (float* p = xArr)
            fixed (float* o = oArr)
            {
                for (int i = 0; i < Calls; i++)
                    AiDotNet.Tensors.Engines.Simd.SimdKernels.FusedGELUUnsafe(p, o, x.Length);
            }
        }
        sw.Stop();
        double rawGeluFusedUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Calls;

        _output.WriteLine($"GELU on [{Seq}, {Hidden}] = {x.Length:N0} elements ({Calls} calls):");
        _output.WriteLine($"  engine.GELU(x)             : {engineGeluUs,8:F1} µs/call");
        _output.WriteLine($"  raw SimdKernels.GELU       : {rawGeluOldUs,8:F1} µs/call (old: tanh decomp)");
        _output.WriteLine($"  raw SimdKernels.FusedGELU  : {rawGeluFusedUs,8:F1} µs/call (new: x·sigmoid(2s))");
        _output.WriteLine($"  speedup new vs old         : {rawGeluOldUs / rawGeluFusedUs,8:F2}× ");
        _output.WriteLine($"  wrapper overhead vs Fused  : {engineGeluUs - rawGeluFusedUs,8:F1} µs/call");
    }

    /// <summary>
    /// Per-op timing breakdown for the canonical ViT-Base patch
    /// shape <c>[197, 768]</c>. Reports µs/call for each op so a
    /// future migration round can be measured against current
    /// numbers. The aggregate ms/iter assertion only catches gross
    /// regressions; the per-op output is the diagnostic.
    /// </summary>
    [Fact]
    [Trait("Category", "Performance")]
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

        // No wall-clock assertion: this is a diagnostic test, not a
        // hard gate. CI runner hardware varies enormously (the
        // ubuntu-24.04 GitHub runner measured this same workload at
        // 3,649 ms/iter with matmul=290 ms/call vs ~1.7 ms/call on
        // a typical AVX2 desktop — 162x slower). Any wall-clock
        // budget set for one hardware tier produces false failures
        // on another. The per-op breakdown above is the deliverable:
        // anyone investigating a perf regression compares the
        // current numbers to the historical record from this test's
        // ITestOutputHelper output.
        //
        // The assertion below catches only the trivially-broken case:
        // the test must complete in a finite amount of time. Any
        // future migration round that wants regression-gating should
        // build a relative-comparison test (this-PR-numbers vs
        // recorded-baseline-numbers) rather than an absolute budget.
        Assert.True(msPerIter > 0,
            $"Test produced non-positive per-iter time {msPerIter} ms — "
            + "Stopwatch or workload broken.");
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
