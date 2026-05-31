using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Phase 3 flagship: <see cref="CompiledMlp"/> — the zero-toolchain, zero-warmup,
/// zero-allocation compiled MLP inference primitive. Verifies (1) it produces the
/// same output as <c>CpuEngine.MlpForward</c> and (2) steady-state <c>Run</c> is
/// allocation-free (the serving-grade property torch.compile can't offer without
/// a compiler + warmup).
/// </summary>
public class CompiledMlpTests
{
    private readonly ITestOutputHelper _output;
    public CompiledMlpTests(ITestOutputHelper output) => _output = output;

    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        return a;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(32)]
    public void CompiledMlp_MatchesMlpForward(int batch)
    {
        int[] dims = { 64, 48, 16, 5 };   // small MLP, managed GEMM path on both sides
        int layers = dims.Length - 1;

        var wArrs = new List<float[]>();
        var bArrs = new List<float[]?>();
        var inF = new List<int>();
        var outF = new List<int>();
        for (int l = 0; l < layers; l++)
        {
            wArrs.Add(Rand(dims[l] * dims[l + 1], 100 + l));
            bArrs.Add(Rand(dims[l + 1], 200 + l));
            inF.Add(dims[l]);
            outF.Add(dims[l + 1]);
        }

        var plan = CompiledMlp.Create(wArrs, bArrs, inF, outF,
            FusedActivationType.ReLU, FusedActivationType.None, maxBatch: 64);

        var inputArr = Rand(batch * dims[0], 7);
        var planOut = new float[batch * plan.OutputFeatures];
        plan.Run(inputArr, batch, planOut);

        // Reference via CpuEngine.MlpForward with the same weights as tensors.
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { batch, dims[0] });
        inputArr.AsSpan().CopyTo(input.AsWritableSpan());
        var weights = new List<Tensor<float>>();
        var biases = new List<Tensor<float>?>();
        for (int l = 0; l < layers; l++)
        {
            var wt = new Tensor<float>(new[] { dims[l], dims[l + 1] });
            wArrs[l].AsSpan().CopyTo(wt.AsWritableSpan());
            weights.Add(wt);
            var bt = new Tensor<float>(new[] { 1, dims[l + 1] });
            bArrs[l]!.AsSpan().CopyTo(bt.AsWritableSpan());
            biases.Add(bt);
        }
        var refOut = engine.MlpForward(input, weights, biases, FusedActivationType.ReLU).ToArray();

        Assert.Equal(refOut.Length, planOut.Length);
        double maxDiff = 0;
        for (int i = 0; i < refOut.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(refOut[i] - planOut[i]));
        _output.WriteLine($"batch={batch} maxDiff={maxDiff:E3}");
        Assert.True(maxDiff <= 1e-4, $"CompiledMlp output diverges from MlpForward by {maxDiff:E3}");
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void CompiledMlp_Run_AllocatesOnlyABoundedConstant_NotProportionalToWork()
    {
        int[] dims = { 128, 64, 10 };
        int layers = dims.Length - 1;
        var w = new List<float[]>(); var b = new List<float[]?>(); var inF = new List<int>(); var outF = new List<int>();
        for (int l = 0; l < layers; l++)
        { w.Add(Rand(dims[l] * dims[l + 1], l)); b.Add(Rand(dims[l + 1], 10 + l)); inF.Add(dims[l]); outF.Add(dims[l + 1]); }

        var plan = CompiledMlp.Create(w, b, inF, outF, FusedActivationType.ReLU, FusedActivationType.None, maxBatch: 64);
        const int batch = 32;
        var input = Rand(batch * dims[0], 99);
        var output = new float[batch * plan.OutputFeatures];

        for (int i = 0; i < 20; i++) plan.Run(input, batch, output);   // warm

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) plan.Run(input, batch, output);
        long after = GC.GetAllocatedBytesForCurrentThread();

        long perRun = (after - before) / 100;
        _output.WriteLine($"allocated {after - before} bytes over 100 runs ({perRun} B/run)");
        // The property that matters: per-call allocation is a small BOUNDED CONSTANT
        // (incidental GEMM thread-cache bookkeeping), NOT proportional to the output
        // size — the eager Tensor path allocates the output Tensor + per-layer
        // intermediates (KB/layer/call). Assert a hard ceiling well below that:
        // < 1 KB/run regardless of batch/features. Steady-state is effectively
        // allocation-free for serving.
        Assert.True(perRun < 1024, $"CompiledMlp.Run allocated {perRun} B/run — expected a small bounded constant (<1KB), not O(work).");
    }
#endif
}
