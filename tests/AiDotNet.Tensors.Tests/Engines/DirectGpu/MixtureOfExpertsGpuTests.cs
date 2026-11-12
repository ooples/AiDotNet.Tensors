#if NETCOREAPP
// GpuCpuCorrectnessFixture (GpuCpuCorrectnessTests.cs) is compiled only off .NET Framework.
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Ref = AiDotNet.Tensors.Tests.Engines.MixtureOfExpertsTests;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="MixtureOfExperts.SwiGluExperts"/> on the DirectGpu engine, forward and tape backward, against the host
/// engine; the device path must be the one taken (the fused kernels ready, the routing kept resident).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class MixtureOfExpertsGpuTests : IClassFixture<GpuCpuCorrectnessFixture>
{
    private readonly GpuCpuCorrectnessFixture _fixture;

    public MixtureOfExpertsGpuTests(GpuCpuCorrectnessFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ForwardAndGradients_MatchHost(bool withBias)
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        var gpu = _fixture.Gpu;
        Assert.True(gpu.GetBackend() is IMoeKernelBackend { MoeKernelsReady: true },
            "the device has no fused MoE kernels");
        const int n = 96, d = 40, g = 12, k = 3, f = 24, b = 2;
        var x = Ref.Random<float>(1, -1, 1, n, d);
        var s = Ref.Random<float>(2, 0.05, 0.95, n, g);
        var bias = withBias ? Ref.Random<float>(3, -0.3, 0.3, g) : null;
        var w1 = Ref.Random<float>(4, -0.5, 0.5, g, d, f);
        var w3 = Ref.Random<float>(5, -0.5, 0.5, g, d, f);
        var w2 = Ref.Random<float>(6, -0.5, 0.5, g, f, d);
        var wy = Ref.Random<float>(7, -1, 1, n, d);
        var inputs = new[] { x, s, w1, w3, w2 };

        (double[] Y, double[] Loads, double[][] Grads) Run(IEngine e)
        {
            using var tape = new GradientTape<float>();
            var y = MixtureOfExperts.SwiGluExperts(e, x, s, bias, w1, w3, w2, k, b, out var loads);
            var loss = e.ReduceSum(e.TensorMultiply(y, wy), null);
            var grads = tape.ComputeGradients(loss, inputs);
            return (Ref.D(y), Ref.D(loads), inputs.Select(t => Ref.D(grads[t])).ToArray());
        }

        var host = Run(new CpuEngine());
        var dev = Run(gpu);
        Assert.Equal(host.Loads, dev.Loads);
        Close("y", host.Y, dev.Y);
        string[] names = { "dx", "dscores", "dw1", "dw3", "dw2" };
        for (int i = 0; i < names.Length; i++) Close(names[i], host.Grads[i], dev.Grads[i]);
    }

    private static void Close(string what, double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1, expected.Max(Math.Abs));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 * scale, $"{what}[{i}]: host {expected[i]:R}, device {actual[i]:R}");
    }
}
#endif
