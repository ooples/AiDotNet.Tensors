// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Model parameters are views (offset into one flat ParameterBuffer). The fused GPU AdamW refused every view, so
/// eager training ran op-by-op AdamW on the host copies (measured: 25 of 26 parameters of an LM, every step). The
/// flat array now lives on the device as one buffer and each parameter is updated through a view: training through
/// views must match training standalone copies, move nothing to the host, and leave the parameters device-resident.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GpuOptimizerFlatParameterViewTests : IClassFixture<DirectGpuTensorEngineTestFixture>, IDisposable
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public GpuOptimizerFlatParameterViewTests(DirectGpuTensorEngineTestFixture fixture)
    {
        _fixture = fixture;
        if (fixture.IsAvailable) AiDotNetEngine.Current = fixture.Engine!;
    }

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    // loss = sum(tanh(x·w1)·w2)
    private static void Train(IEngine e, Tensor<float>[] ps, Tensor<float>[] m, Tensor<float>[] v, Tensor<float> x, int steps)
    {
        for (int s = 0; s < steps; s++)
        {
            using var tape = new GradientTape<float>();
            var h = e.TensorTanh(e.TensorMatMul(x, ps[0]));
            var y = e.TensorMatMul(h, ps[1]);
            var loss = e.ReduceSum(y, new[] { 0, 1 }, keepDims: false);
            var grads = tape.ComputeGradients(loss, ps);
            for (int i = 0; i < ps.Length; i++)
                Assert.True(GpuOptimizer.TryAdamWStep(ps[i], grads[ps[i]], m[i], v[i], 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, s + 1),
                    $"the fused GPU AdamW declined parameter {i}: " + string.Join("; ", GpuLaunchProbe.Fallbacks));
        }
    }

    [SkippableFact]
    public void AdamWThroughParameterBufferViews_MatchesStandaloneParameters_WithoutHostTraffic()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        int[][] shapes = { new[] { 8, 16 }, new[] { 16, 4 } };
        var w1 = Rand(shapes[0], 1);
        var w2 = Rand(shapes[1], 2);
        var x = Rand(new[] { 32, 8 }, 3);

        var buffer = new ParameterBuffer<float>(shapes);
        var views = buffer.CreateAllViews();
        buffer.CopyFrom(new[] { w1, w2 });
        var standalone = new[] { new Tensor<float>(shapes[0]), new Tensor<float>(shapes[1]) };
        for (int i = 0; i < w1.Length; i++) standalone[0][i] = w1[i];
        for (int i = 0; i < w2.Length; i++) standalone[1][i] = w2[i];

        Tensor<float>[] State() => new[] { GpuOptimizer.CreateStateTensor(shapes[0]), GpuOptimizer.CreateStateTensor(shapes[1]) };
        Train(gpu, standalone, State(), State(), x, 3);

        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        long readbackBytes;
        string sites;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            GpuLaunchProbe.Reset();
            Train(gpu, views, State(), State(), x, 3);
            readbackBytes = GpuLaunchProbe.ReadbackBytes;
            sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
        }
        Assert.True(readbackBytes == 0, $"training through views read back {readbackBytes} bytes: {sites}");
        Assert.True(gpu.IsDeviceAuthoritative(views[0]) && gpu.IsDeviceAuthoritative(views[1]),
            "after on-device AdamW the parameter views are not device-authoritative");

        for (int p = 0; p < 2; p++)
        {
            var got = views[p].ToArray();          // host read: downloads the flat buffer once
            var expected = standalone[p].ToArray();
            for (int i = 0; i < got.Length; i++)
                Assert.True(Math.Abs(got[i] - expected[i]) < 1e-5f, $"param {p}[{i}]: view {got[i]} standalone {expected[i]}");
        }
        // The flat vector reflects the same values (views and buffer share storage).
        Assert.Equal(views[1][0], buffer.AsVector()[shapes[0][0] * shapes[0][1]]);
    }
}
