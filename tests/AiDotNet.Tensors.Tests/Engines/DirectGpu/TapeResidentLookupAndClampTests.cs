// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Ops that bailed to the CPU base under a gradient tape, measured on an LM training step:
/// <list type="bullet">
/// <item>TensorEmbeddingLookupFromFloatIndices downloaded the whole [V, E] table (107 MB/step); its dense backward
/// downloaded the upstream gradient and built the table gradient on the host, so accumulating it with the tied
/// output head's device gradient downloaded another [V, E] (100 MB/step).</item>
/// <item>TensorClamp and ScalarMinusTensor downloaded their inputs.</item>
/// </list>
/// They now run on the device under a tape and record the same backward.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class TapeResidentLookupAndClampTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public TapeResidentLookupAndClampTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Filled(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    /// <summary>Runs <paramref name="build"/> under a tape on <paramref name="e"/>; returns the gradient w.r.t. <paramref name="wrt"/>.</summary>
    private static float[] Gradient(IEngine e, Tensor<float> wrt, Func<IEngine, Tensor<float>> build, out long readback, out string sites)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = e;
        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            using var tape = new GradientTape<float>();
            GpuLaunchProbe.Reset();
            var loss = build(e);
            var grad = tape.ComputeGradients(loss, new[] { wrt })[wrt];
            readback = GpuLaunchProbe.ReadbackBytes;
            sites = string.Join("; ", GpuLaunchProbe.ReadbackSites) + " | fallbacks: " + string.Join("; ", GpuLaunchProbe.Fallbacks);
            return grad.ToArray();
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
            AiDotNetEngine.Current = previous;
        }
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"{what}[{i}]: cpu {expected[i]} gpu {actual[i]}");
    }

    [SkippableTheory]
    [InlineData(true)]    // the default: fixed-order device scatter
    [InlineData(false)]   // atomicAdd scatter
    public void TiedFloatIdEmbedding_UnderTape_MatchesCpu_WithoutHostTraffic(bool deterministic)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var previousMode = AiDotNet.Tensors.Helpers.BlasProvider.GetThreadLocalDeterministicMode();
        AiDotNet.Tensors.Helpers.BlasProvider.SetThreadLocalDeterministicMode(deterministic);
        try
        {
            TiedFloatIdEmbedding();
        }
        finally
        {
            AiDotNet.Tensors.Helpers.BlasProvider.SetThreadLocalDeterministicMode(previousMode);
        }
    }

    private void TiedFloatIdEmbedding()
    {
        const int vocab = 50, dim = 8;
        var table = Filled(new[] { vocab, dim }, 1);
        var ids = new Tensor<float>(new[] { 2, 3 });
        float[] idValues = { 3, 17, 3, 49, 0, 17 };            // repeats accumulate
        for (int i = 0; i < idValues.Length; i++) ids[i] = idValues[i];
        var weights = Filled(new[] { 2, 3, vocab }, 2);

        // Embedding lookup, then the tied output head (hidden · tableᵀ): the table gets two gradient contributions.
        Tensor<float> Loss(IEngine e)
        {
            var hidden = e.TensorTanh(e.TensorEmbeddingLookupFromFloatIndices(table, ids));
            var logits = e.TensorMatMul(e.Reshape(hidden, new[] { 6, dim }), e.TensorTranspose(table));
            return e.ReduceSum(e.TensorMultiply(e.Reshape(logits, new[] { 2, 3, vocab }), weights), new[] { 0, 1, 2 }, keepDims: false);
        }

        var cpu = Gradient(new CpuEngine(), table, Loss, out _, out _);
        var gpu = Gradient(_fixture.Engine!, table, Loss, out long readback, out string sites);
        Assert.True(readback <= 64, $"the tied embedding forward+backward read back {readback} bytes: {sites}");
        AssertClose(cpu, gpu, 1e-4f, "d/dtable");
    }

    [SkippableFact]
    public void FloatIdEmbedding_OnTheDevice_RejectsAnOutOfRangeHostId()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var ids = new Tensor<float>(new[] { 2 });
        ids[0] = 1;
        ids[1] = 50;
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorEmbeddingLookupFromFloatIndices(Filled(new[] { 50, 4 }, 3), ids));
    }

    [SkippableFact]
    public void ClampAndScalarMinus_UnderTape_MatchCpu_WithoutHostTraffic()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var x = Filled(new[] { 4, 16 }, 5);
        var weights = Filled(new[] { 4, 16 }, 6);
        Tensor<float> Loss(IEngine e)
        {
            var clamped = e.TensorClamp(e.TensorMultiplyScalar(e.TensorTanh(x), 2f), -0.5f, 0.75f);
            var flipped = e.ScalarMinusTensor(3f, clamped);
            return e.ReduceSum(e.TensorMultiply(flipped, weights), new[] { 0, 1 }, keepDims: false);
        }

        var cpu = Gradient(new CpuEngine(), x, Loss, out _, out _);
        var gpu = Gradient(_fixture.Engine!, x, Loss, out long readback, out string sites);
        Assert.True(readback <= 64, $"clamp + scalar-minus forward+backward read back {readback} bytes: {sites}");
        AssertClose(cpu, gpu, 1e-5f, "d/dx");
    }
}
