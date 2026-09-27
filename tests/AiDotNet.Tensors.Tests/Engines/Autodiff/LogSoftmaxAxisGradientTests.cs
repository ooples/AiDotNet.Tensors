// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// LogSoftmax's backward reduced over the LAST axis regardless of the axis the forward normalised, so a log-softmax
/// over any other axis (a class axis of 1 in [N, C, L]) got the gradient of a different function. Checked against
/// central finite differences for every axis of a rank-3 tensor, and the GPU engine against the CPU engine.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class LogSoftmaxAxisGradientTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public LogSoftmaxAxisGradientTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static readonly int[] Shape = { 2, 3, 4 };

    private static Tensor<T> Rand<T>(int seed, Func<double, T> conv)
    {
        var rng = new Random(seed);
        var t = new Tensor<T>(Shape);
        for (int i = 0; i < t.Length; i++) t[i] = conv(rng.NextDouble() * 4 - 2);
        return t;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(-2)]
    public void Gradient_MatchesFiniteDifferences_OnEveryAxis(int axis)
    {
        IEngine e = new CpuEngine();
        var x = Rand(1, v => v);
        var g = Rand(2, v => v);   // loss = Σ g · log_softmax(x)

        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var y = e.TensorLogSoftmax(x, axis);
            grads = tape.ComputeGradients(y, new[] { x }, createGraph: false,
                seedOverride: new[] { new KeyValuePair<Tensor<double>, Tensor<double>>(y, g) });
        }

        double Loss()
        {
            var y = e.TensorLogSoftmax(x, axis);
            double l = 0;
            for (int i = 0; i < y.Length; i++) l += g[i] * y[i];
            return l;
        }

        const double h = 1e-6;
        var grad = grads[x];
        for (int i = 0; i < x.Length; i++)
        {
            double orig = x[i];
            x[i] = orig + h; double lp = Loss();
            x[i] = orig - h; double lm = Loss();
            x[i] = orig;
            double fd = (lp - lm) / (2 * h);
            Assert.True(Math.Abs(grad[i] - fd) < 1e-6, $"axis {axis} d/dx[{i}]: tape {grad[i]:G8} vs finite differences {fd:G8}");
        }
    }

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void GpuEngine_MatchesCpuEngine_OnEveryAxis(int axis)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        float[] Run(IEngine e)
        {
            var x = Rand(1, v => (float)v);
            var g = Rand(2, v => (float)v);
            using var tape = new GradientTape<float>();
            var y = e.TensorLogSoftmax(x, axis);
            var grads = tape.ComputeGradients(y, new[] { x }, createGraph: false,
                seedOverride: new[] { new KeyValuePair<Tensor<float>, Tensor<float>>(y, g) });
            return grads[x].ToArray();
        }
        var cpu = Run(new CpuEngine());
        var gpu = Run(_fixture.Engine!);
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(Math.Abs(cpu[i] - gpu[i]) < 1e-4, $"axis {axis} d/dx[{i}]: cpu {cpu[i]} gpu {gpu[i]}");
    }
}
