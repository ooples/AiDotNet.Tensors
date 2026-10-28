// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Conv, inference batch norm and activation as one op (#1432): folded into the convolution outside a tape, the
/// unfused ops under one. Each case is checked against the unfused composition of the same engine.
/// </summary>
public sealed class FusedConv2DBatchNormTests
{
    private const int Batch = 2, InC = 3, OutC = 5, H = 9, W = 7, K = 3;

    private static Tensor<float> Rand(int seed, float lo, float hi, params int[] shape)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return new Tensor<float>(data, shape);
    }

    private sealed record Case(Tensor<float> Input, Tensor<float> Kernel, Tensor<float>? Bias, Tensor<float> Gamma,
        Tensor<float> Beta, Tensor<float> Mean, Tensor<float> Variance);

    private static Case Make(bool withBias) => new(
        Rand(1, -1f, 1f, Batch, InC, H, W), Rand(2, -0.5f, 0.5f, OutC, InC, K, K),
        withBias ? Rand(3, -0.2f, 0.2f, OutC) : null, Rand(4, 0.5f, 1.5f, OutC), Rand(5, -0.3f, 0.3f, OutC),
        Rand(6, -0.2f, 0.2f, OutC), Rand(7, 0.1f, 2f, OutC));

    private static Tensor<float> Unfused(IEngine e, Case c, int stride, int pad, FusedActivationType act)
        => e.BatchNormInference(
            e.FusedConv2D(c.Input, c.Kernel, c.Bias, stride, stride, pad, pad, 1, 1, FusedActivationType.None),
            c.Gamma, c.Beta, c.Mean, c.Variance, 1e-5, act);

    // The differentiable reference: eval-mode batch norm as an affine op with gradients for x, gamma and beta.
    private static Tensor<float> UnfusedAffine(IEngine e, Case c)
        => e.ReLU(e.BatchNormAffine(
            e.FusedConv2D(c.Input, c.Kernel, c.Bias, 1, 1, 1, 1, 1, 1, FusedActivationType.None),
            c.Gamma, c.Beta, c.Mean, c.Variance, 1e-5));

    private static Tensor<float> Fused(IEngine e, Case c, int stride, int pad, FusedActivationType act)
        => e.FusedConv2DBatchNorm(c.Input, c.Kernel, c.Bias, c.Gamma, c.Beta, c.Mean, c.Variance, 1e-5,
            stride, stride, pad, pad, 1, 1, act);

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tol * (1 + Math.Abs(e[i])), $"element {i}: expected {e[i]}, got {a[i]}");
    }

    [Theory]
    [InlineData(true, 1, 1, FusedActivationType.ReLU)]
    [InlineData(false, 1, 1, FusedActivationType.None)]
    [InlineData(true, 2, 0, FusedActivationType.None)]
    [InlineData(false, 2, 1, FusedActivationType.ReLU)]
    public void Cpu_MatchesConvThenBatchNormThenActivation(bool withBias, int stride, int pad, FusedActivationType act)
    {
        var cpu = new CpuEngine();
        var c = Make(withBias);
        AssertClose(Unfused(cpu, c, stride, pad, act), Fused(cpu, c, stride, pad, act));
    }

    [SkippableTheory]
    [InlineData(true, FusedActivationType.ReLU)]
    [InlineData(false, FusedActivationType.None)]
    public void Gpu_MatchesTheCpuComposition(bool withBias, FusedActivationType act)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend.");
        using var _ = gpu;
        var c = Make(withBias);
        AssertClose(Unfused(new CpuEngine(), c, 1, 1, act), Fused(gpu, c, 1, 1, act), 1e-3f);
        // A second call reuses the cached fold and must give the same values.
        AssertClose(Unfused(new CpuEngine(), c, 1, 1, act), Fused(gpu, c, 1, 1, act), 1e-3f);
    }

    // Parameters whose value exists only on the device (after on-device training) fold on the device: the host fold would
    // download each one. A warm call moves nothing across the host/device boundary.
    [SkippableFact]
    public void Gpu_DeviceOnlyParameters_FoldOnTheDevice_WithoutReadbacks()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend.");
        using var _ = gpu;
        var host = Make(withBias: true);
        Tensor<float> OnDevice(Tensor<float> t) => gpu.TensorAddScalar(t, 0f);   // a resident result, no host copy
        var device = new Case(OnDevice(host.Input), OnDevice(host.Kernel), OnDevice(host.Bias!), OnDevice(host.Gamma),
            OnDevice(host.Beta), OnDevice(host.Mean), OnDevice(host.Variance));
        var expected = Unfused(new CpuEngine(), host, 1, 1, FusedActivationType.ReLU);

        using (var cold = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin())
        {
            Fused(gpu, device, 1, 1, FusedActivationType.ReLU);   // folds and caches
            Assert.Equal(0, cold.Downloads);                       // the host fold would download all six parameters
        }
        Tensor<float> warm;
        using (var scope = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin())
        {
            warm = Fused(gpu, device, 1, 1, FusedActivationType.ReLU);
            Assert.Equal(0, scope.Downloads);
            Assert.Equal(0, scope.Uploads);
        }
        AssertClose(expected, warm, 1e-3f);
    }

    [Fact]
    public void ChangingABatchNormParameter_RebuildsTheFold()
    {
        var cpu = new CpuEngine();
        var c = Make(withBias: true);
        AssertClose(Unfused(cpu, c, 1, 1, FusedActivationType.None), Fused(cpu, c, 1, 1, FusedActivationType.None));

        c.Gamma[0] = 3f;            // an in-place update, as an optimizer or a weight load makes
        c.Mean[OutC - 1] = 0.7f;
        AssertClose(Unfused(cpu, c, 1, 1, FusedActivationType.None), Fused(cpu, c, 1, 1, FusedActivationType.None));

        var replacedVariance = c with { Variance = Rand(8, 0.1f, 2f, OutC) };   // a different tensor, same kernel
        AssertClose(Unfused(cpu, replacedVariance, 1, 1, FusedActivationType.None),
            Fused(cpu, replacedVariance, 1, 1, FusedActivationType.None));
    }

    [Fact]
    public void UnderATape_GradientsMatchTheUnfusedOps()
    {
        var cpu = new CpuEngine();
        var c = Make(withBias: true);
        Tensor<float> FusedLoss() => cpu.ReduceMean(Fused(cpu, c, 1, 1, FusedActivationType.ReLU), new[] { 0, 1, 2, 3 }, keepDims: false);
        Tensor<float> UnfusedLoss() => cpu.ReduceMean(UnfusedAffine(cpu, c), new[] { 0, 1, 2, 3 }, keepDims: false);

        var sources = new[] { c.Input, c.Kernel, c.Gamma, c.Beta };
        Tensor<float>[] Grads(Func<Tensor<float>> loss)
        {
            using var tape = new GradientTape<float>();
            var l = loss();
            var g = tape.ComputeGradients(l, sources);
            return Array.ConvertAll(sources, s => g[s]);
        }

        var fused = Grads(FusedLoss);
        var unfused = Grads(UnfusedLoss);
        for (int i = 0; i < sources.Length; i++) AssertClose(unfused[i], fused[i]);
    }
}
