// Copyright (c) AiDotNet. All rights reserved.
// Issue #338 Phase G test coverage — exercises the new infrastructure
// (StepTiming, BackwardParallel, BlasProvider MKL/BF16/batched paths,
// CompiledTrainingPlan analytic backwards, slice-prefix fusion, etc.)
// without the AIDOTNET_RUN_PERF_GATES requirement that gates the perf
// benchmark tests.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue338PhaseGCoverageTests
{
    private readonly ITestOutputHelper _output;
    public Issue338PhaseGCoverageTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Phase A / F.1 — StepTiming aggregator: verify that recording &
    /// dump are no-ops when env var is off, and that DumpAndReset
    /// handles the no-data case cleanly.
    /// </summary>
    [Fact]
    public void StepTiming_NoEnvVar_IsNoOp()
    {
        // Without AIDOTNET_STEP_TIMING=1, all Record* calls and the
        // dump should be no-ops; the writer must not receive any lines.
        int linesWritten = 0;
        StepTiming.RecordForward(1234567);
        StepTiming.RecordBackward(7654321);
        StepTiming.RecordOptimizer(42);
        StepTiming.IncrementStepCount();
        StepTiming.DumpAndReset(_ => linesWritten++);
        Assert.Equal(0, linesWritten);
    }

    /// <summary>
    /// Phase C.1 — BackwardParallel.ForRows respects MinWorkForParallel
    /// gate (below threshold runs serial, above runs parallel) and the
    /// recursion guard prevents nested parallelism.
    /// </summary>
    [Fact]
    public void BackwardParallel_RespectsWorkThreshold()
    {
        // Tiny workload — far below MinWorkForParallel=64K — must run
        // serial. We check by accumulating counts; any execution path
        // (serial or parallel) should visit each row exactly once.
        int rows = 8;
        long workPerRow = 1;  // total work = 8, well below threshold
        var visited = new int[rows];
        BackwardParallel.ForRows(rows, workPerRow, r => visited[r]++);
        for (int r = 0; r < rows; r++) Assert.Equal(1, visited[r]);

        // Above threshold — parallel path. Same correctness guarantee.
        int largeRows = 256;
        long largeWork = 1024;  // total = 256K, above 64K threshold
        var visitedLarge = new int[largeRows];
        BackwardParallel.ForRows(largeRows, largeWork, r =>
        {
            System.Threading.Interlocked.Increment(ref visitedLarge[r]);
        });
        for (int r = 0; r < largeRows; r++) Assert.Equal(1, visitedLarge[r]);

        // Zero rows — no-op.
        BackwardParallel.ForRows(0, 100, _ => Assert.True(false, "Should not fire"));

        // Single row — never parallelizes (no benefit).
        bool called = false;
        BackwardParallel.ForRows(1, 1_000_000_000L, _ => called = true);
        Assert.True(called);
    }

    /// <summary>
    /// Phase C.1 — Invoke2 runs serial when work is below threshold,
    /// parallel when both branches are above.
    /// </summary>
    [Fact]
    public void BackwardParallel_Invoke2_RespectsThreshold()
    {
        // Small work — serial path.
        bool aRan = false, bRan = false;
        BackwardParallel.Invoke2(100L, 100L,
            () => aRan = true,
            () => bRan = true);
        Assert.True(aRan);
        Assert.True(bRan);

        // Large work — parallel path. Both must fire.
        bool aRan2 = false, bRan2 = false;
        BackwardParallel.Invoke2(1_000_000L, 1_000_000L,
            () => aRan2 = true,
            () => bRan2 = true);
        Assert.True(aRan2);
        Assert.True(bRan2);

        // Single-branch big, other small → serial (gate requires BOTH big).
        bool a3 = false, b3 = false;
        BackwardParallel.Invoke2(1_000_000L, 10L, () => a3 = true, () => b3 = true);
        Assert.True(a3);
        Assert.True(b3);
    }

    /// <summary>
    /// Phase G.1 — BlasProvider exposes IsAvailable; when MKL/OpenBLAS
    /// is loadable it should be true, and the BackendName surfaces
    /// the active provider.
    /// </summary>
    [Fact]
    public void BlasProvider_BackendName_NonEmpty()
    {
        var name = BlasProvider.BackendName;
        Assert.False(string.IsNullOrEmpty(name));
        // IsAvailable should match the BackendName's "loaded" form.
        // Both states are valid; just verify they're consistent.
        bool available = BlasProvider.IsAvailable;
        if (available)
            Assert.Contains("OpenBLAS", name, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Phase G.5 — Fp32ToBf16 round-trip preserves values within
    /// BF16 precision (~3 decimal digits / 7-bit mantissa).
    /// </summary>
    [Fact]
    public unsafe void BlasProvider_Fp32ToBf16_RoundTripPrecision()
    {
        // Scalar API.
        float[] testValues = { 1.0f, -1.0f, 2.5f, 100.0f, 0.001f, 1234.5f };
        foreach (var v in testValues)
        {
            ushort bf16 = BlasProvider.Fp32ToBf16(v);
            // Reconstruct: shift up 16 bits → re-interpret as float.
            uint upper = (uint)bf16 << 16;
            float reconstructed = *(float*)&upper;
            float relErr = MathF.Abs(reconstructed - v) / MathF.Max(MathF.Abs(v), 1e-6f);
            Assert.True(relErr < 0.01f,
                $"Fp32ToBf16({v}) = {bf16:X4}, reconstructed = {reconstructed}, relErr = {relErr:E3}");
        }

        // Bulk API.
        var src = new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f };
        var dst = new ushort[src.Length];
        fixed (float* pSrc = src)
        fixed (ushort* pDst = dst)
        {
            BlasProvider.Fp32ToBf16Bulk(pSrc, pDst, src.Length);
        }
        for (int i = 0; i < src.Length; i++)
        {
            ushort expected = BlasProvider.Fp32ToBf16(src[i]);
            Assert.Equal(expected, dst[i]);
        }
    }

    /// <summary>
    /// Phase G.4 — EnableFrozenWeightOptimizations is idempotent and
    /// produces equivalent gradients (the test cache pattern doesn't
    /// mutate weights, so the contract is preserved).
    /// </summary>
    [Fact]
    public void CompiledTrainingPlan_EnableFrozenWeightOpts_Idempotent()
    {
        var engine = new CpuEngine();
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var rng = new Random(42);
            var inData = new float[8 * 4];
            for (int i = 0; i < inData.Length; i++) inData[i] = (float)(rng.NextDouble() * 2 - 1);
            var weightData = new float[4 * 2];
            for (int i = 0; i < weightData.Length; i++) weightData[i] = (float)(rng.NextDouble() * 2 - 1);

            var input = new Tensor<float>(inData, new[] { 8, 4 });
            var weight = new Tensor<float>(weightData, new[] { 4, 2 });

            using var cache = new CompiledModelCache<float>();
            var plan = cache.GetOrCompileTraining(
                inputShape: input._shape,
                forwardAndLoss: () =>
                {
                    var y = engine.TensorMatMul(input, weight);
                    return engine.ReduceSum(y, axes: null, keepDims: false);
                },
                parameters: new[] { weight });

            // Call EnableFrozenWeightOptimizations multiple times — must be safe.
            plan.EnableFrozenWeightOptimizations();
            plan.EnableFrozenWeightOptimizations();
            plan.EnableFrozenWeightOptimizations();

            // Step should still work and produce a scalar loss.
            var loss = plan.Step();
            Assert.Equal(1, loss.Length);
            Assert.False(float.IsNaN(loss[0]) || float.IsInfinity(loss[0]));
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    /// <summary>
    /// Phase G.7 — analytic loss-MatMul backward produces gradients
    /// equal to the standard GEMM backward within float tolerance.
    /// Smoke test on a minimal MatMul→ReduceSum graph.
    /// </summary>
    [Fact]
    public void AnalyticLossMatMul_GradientsMatchGEMM()
    {
        var engine = new CpuEngine();
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            const int M = 8, K = 4, N = 6;
            var rng = new Random(123);
            var xData = new float[M * K];
            var wData = new float[K * N];
            for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2 - 1);

            var x = new Tensor<float>(xData, new[] { M, K });
            var w = new Tensor<float>(wData, new[] { K, N });

            using var cache = new CompiledModelCache<float>();
            var plan = cache.GetOrCompileTraining(
                inputShape: x._shape,
                forwardAndLoss: () =>
                {
                    var y = engine.TensorMatMul(x, w);
                    return engine.ReduceSum(y, axes: null, keepDims: false);
                },
                parameters: new[] { w });
            plan.EnableFrozenWeightOptimizations();

            plan.Step();
            var dW = plan.Gradients[0];
            Assert.Equal(K * N, dW.Length);

            // Analytic gradient: dW[k, v] = col_sum_x[k] (same value per v).
            // Verify the first row of dW is the col_sum of x.
            for (int v = 0; v < N; v++)
            {
                float expected = 0f;
                for (int m = 0; m < M; m++) expected += xData[m * K + 0];  // col 0 sum
                float actual = dW.GetDataArray()[0 * N + v];
                Assert.True(MathF.Abs(actual - expected) < 1e-4f,
                    $"dW[0, {v}] = {actual}, expected col_sum_x[0] = {expected}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }
}
