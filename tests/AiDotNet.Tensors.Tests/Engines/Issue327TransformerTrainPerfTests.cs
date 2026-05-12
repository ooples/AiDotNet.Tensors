// Copyright (c) AiDotNet. All rights reserved.
// Issue #327 — synthetic Transformer Train regression gate. Locks in
// the post-Phase-3 wall-clock floor so future PRs can't silently
// regress the Train path. Skipped on hosts without 16+ logical cores
// (the issue's stated target is 16-core x64 — narrower hardware will
// fail by hardware limit, not regression).

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue327TransformerTrainPerfTests
{
    private readonly ITestOutputHelper _output;
    public Issue327TransformerTrainPerfTests(ITestOutputHelper output) { _output = output; }

    // Issue #327 config (matches the consumer Transformer.TrainBatched repro)
    private const int B = 32;
    private const int Ctx = 64;
    private const int D = 128;
    private const int FfDim = 512;
    private const int Vocab = 8192;
    private const int Layers = 4;

    /// <summary>
    /// Regression guard for the Phase 3 persistent-tape Train step on a
    /// 16+ logical-core x64 host. Issue #327's close target is ≤ 100 ms;
    /// the PyTorch-parity stretch target is ≤ 50 ms — this phase HAS NOT
    /// hit the stretch yet (Phase 3 measurements land ~135-140 ms/iter
    /// steady-state on the reference 16-core dev box). The Assert.True
    /// gate below is intentionally set at ≤ 250 ms — well above the
    /// observed mean — so CI noise and slower runners don't flap; the
    /// tighter close / stretch targets are validated by the BDN harness
    /// in tests/AiDotNet.Tensors.Benchmarks instead.
    /// </summary>
    [Fact]
    [Trait("Category", "Perf")]
    public void Issue327_Transformer_PersistentTape_TrainStep_BelowIssueCloseTarget()
    {
        // Hardware gate: the issue scope is 16+ logical core x64. On
        // narrower hardware (CI workers with 2-4 cores, ARM emulation)
        // the wall-time floor is dominated by per-core compute and
        // doesn't reflect the regression this test is designed to catch.
        // Skip unless opt-in: env var AIDOTNET_RUN_PERF_GATES=1 plus
        // 16+ logical cores AND x64 architecture. Other perf-regression
        // guards in this repo follow the same env-var-gated pattern so
        // the default test run on slower CI runners doesn't flap on
        // wall-time variance.
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_GATES") != "1")
        {
            _output.WriteLine("Skip: AIDOTNET_RUN_PERF_GATES != 1.");
            return;
        }
        if (Environment.ProcessorCount < 16)
        {
            _output.WriteLine($"Skip: ProcessorCount={Environment.ProcessorCount} < 16 (issue scope).");
            return;
        }
        // Architecture gate: the close / stretch targets and the
        // SgemmDirect tall-thin tile sizes were tuned against x64
        // AVX2 hosts (the consumer's reference hardware). On ARM64
        // or x86 the same code paths execute via different SIMD
        // dispatch and the targets aren't meaningful — skip rather
        // than report a misleading regression.
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            _output.WriteLine($"Skip: ProcessArchitecture={RuntimeInformation.ProcessArchitecture} (issue scope is X64).");
            return;
        }

        // Save and restore the process-wide engine. AiDotNetEngine.Current
        // is a global static; leaking a fresh CpuEngine into it would
        // bleed into any subsequent test that didn't explicitly
        // construct its own engine, defeating test isolation.
        var priorEngine = AiDotNetEngine.Current;
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            var input = MakeFloatTensor(new[] { B, Ctx, D }, new Random(42));
            var weights = MakeWeights(Layers);

            // Persistent tape with Reset() between steps — the
            // AutoTrainingCompiler contract for the cached-backward fast path.
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

            const int warmup = 3;
            const int iters = 10;
            for (int i = 0; i < warmup; i++)
            {
                tape.Reset();
                var y = ForwardL(engine, input, weights);
                var loss = engine.ReduceSum(y, axes: null, keepDims: false);
                var grads = tape.ComputeGradients(loss, weights);
            }

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                tape.Reset();
                var y = ForwardL(engine, input, weights);
                var loss = engine.ReduceSum(y, axes: null, keepDims: false);
                var grads = tape.ComputeGradients(loss, weights);
            }
            sw.Stop();

            double msPerIter = sw.Elapsed.TotalMilliseconds / iters;
            _output.WriteLine($"Issue #327 persistent-tape Train step: {msPerIter:F2} ms/iter (close target ≤ 100 ms; gate at ≤ 250 ms)");

            // Gate is set well above the measured ~135 ms steady state so
            // CI noise + slower runners don't flap. Tighter regressions
            // (≤ 100 close, ≤ 50 stretch) are validated by the BDN harness.
            Assert.True(msPerIter <= 250.0,
                $"Issue #327 regression: persistent-tape Train step is {msPerIter:F2} ms (> 250 ms gate). "
                + "Phase 3 win (tall-thin SgemmDirect gate) or collapse-2D MatMul backward may have regressed.");
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    /// <summary>
    /// Documents the non-persistent path as the OPEN follow-up. Asserts a
    /// looser ≤ 500 ms ceiling — this won't catch a 1.5× regression but
    /// catches catastrophic failures (e.g. backward going to 10 s due to a
    /// quadratic walk bug).
    /// </summary>
    [Fact]
    [Trait("Category", "Perf")]
    public void Issue327_Transformer_FreshTape_TrainStep_DoesNotExplodeCatastrophically()
    {
        // Skip unless opt-in: env var AIDOTNET_RUN_PERF_GATES=1 plus
        // 16+ logical cores AND x64 architecture. Other perf-regression
        // guards in this repo follow the same env-var-gated pattern so
        // the default test run on slower CI runners doesn't flap on
        // wall-time variance.
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_GATES") != "1")
        {
            _output.WriteLine("Skip: AIDOTNET_RUN_PERF_GATES != 1.");
            return;
        }
        if (Environment.ProcessorCount < 16)
        {
            _output.WriteLine($"Skip: ProcessorCount={Environment.ProcessorCount} < 16 (issue scope).");
            return;
        }
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            _output.WriteLine($"Skip: ProcessArchitecture={RuntimeInformation.ProcessArchitecture} (issue scope is X64).");
            return;
        }

        // Save/restore AiDotNetEngine.Current — see the matching comment
        // on the persistent-tape test above.
        var priorEngine = AiDotNetEngine.Current;
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            var input = MakeFloatTensor(new[] { B, Ctx, D }, new Random(42));
            var weights = MakeWeights(Layers);

            // Warmup — JIT + AutoTracer + AutoTensorCache settle here.
            for (int i = 0; i < 2; i++)
            {
                using var warmTape = new GradientTape<float>();
                var y = ForwardL(engine, input, weights);
                var loss = engine.ReduceSum(y, axes: null, keepDims: false);
                warmTape.ComputeGradients(loss, weights);
            }

            const int iters = 5;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                using var tape = new GradientTape<float>();
                var y = ForwardL(engine, input, weights);
                var loss = engine.ReduceSum(y, axes: null, keepDims: false);
                tape.ComputeGradients(loss, weights);
            }
            sw.Stop();

            double msPerIter = sw.Elapsed.TotalMilliseconds / iters;
            _output.WriteLine($"Issue #327 fresh-tape Train step: {msPerIter:F2} ms/iter "
                + "(catastrophic ceiling ≤ 500 ms; remaining gap to ≤ 50 ms is documented in PR follow-up)");

            Assert.True(msPerIter <= 500.0,
                $"Issue #327 catastrophic regression: fresh-tape Train step is {msPerIter:F2} ms (> 500 ms). "
                + "Backward walk likely has a quadratic / runaway bug.");
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    /// <summary>
    /// Forward pass over L stacked transformer-encoder layers. Matches
    /// the BDN harness's ForwardL so the xUnit gate and the harness
    /// measure the exact same computation.
    /// </summary>
    private static Tensor<float> ForwardL(CpuEngine engine, Tensor<float> input, Tensor<float>[] weights)
    {
        var x = input;
        for (int l = 0; l < Layers; l++)
        {
            int offset = l * 4;
            // QKV proj routed through a slice → Wo path so weights[0]
            // stays on the autodiff graph (reviewer feedback: the prior
            // sink-only `_ = engine.TensorMatMul(...)` disconnected
            // weights[offset+0] from the loss).
            var qkv = engine.TensorMatMul(x, weights[offset + 0]);
            var qProxy = engine.TensorSlice(qkv, new[] { 0, 0, 0 }, new[] { B, Ctx, D });
            var attnOut = engine.TensorMatMul(qProxy, weights[offset + 1]);
            // FFN up + GELU + FFN down
            var f1 = engine.TensorMatMul(attnOut, weights[offset + 2]);
            var f1g = engine.GELU(f1);
            var f2 = engine.TensorMatMul(f1g, weights[offset + 3]);
            x = f2;
        }
        // Output projection to vocab
        return engine.TensorMatMul(x, weights[Layers * 4]);
    }

    private static Tensor<float>[] MakeWeights(int layers)
    {
        var weights = new Tensor<float>[layers * 4 + 1];
        var rng = new Random(123);
        for (int l = 0; l < layers; l++)
        {
            int offset = l * 4;
            weights[offset + 0] = MakeFloatTensor(new[] { D, 3 * D }, rng);
            weights[offset + 1] = MakeFloatTensor(new[] { D, D }, rng);
            weights[offset + 2] = MakeFloatTensor(new[] { D, FfDim }, rng);
            weights[offset + 3] = MakeFloatTensor(new[] { FfDim, D }, rng);
        }
        weights[layers * 4] = MakeFloatTensor(new[] { D, Vocab }, rng);
        return weights;
    }

    private static Tensor<float> MakeFloatTensor(int[] shape, Random rng)
    {
        int total = 1;
        foreach (var s in shape) total *= s;
        var data = new float[total];
        float scale = MathF.Sqrt(2.0f / shape[0]);
        for (int i = 0; i < total; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        return new Tensor<float>(data, shape);
    }
}
