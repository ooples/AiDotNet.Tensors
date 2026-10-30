// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
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

    private sealed class Case
    {
        public Case(Tensor<float> input, Tensor<float> kernel, Tensor<float>? bias, Tensor<float> gamma, Tensor<float> beta,
            Tensor<float> mean, Tensor<float> variance)
        {
            Input = input; Kernel = kernel; Bias = bias; Gamma = gamma; Beta = beta; Mean = mean; Variance = variance;
        }

        public Tensor<float> Input { get; }
        public Tensor<float> Kernel { get; }
        public Tensor<float>? Bias { get; }
        public Tensor<float> Gamma { get; }
        public Tensor<float> Beta { get; }
        public Tensor<float> Mean { get; }
        public Tensor<float> Variance { get; }

        public Case WithVariance(Tensor<float> variance) => new(Input, Kernel, Bias, Gamma, Beta, Mean, variance);
    }

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

        var replacedVariance = c.WithVariance(Rand(8, 0.1f, 2f, OutC));   // a different tensor, same kernel
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

    // Device weight caches (and the Winograd filter transform) are keyed by the folded kernel's identity, so an
    // unchanged call must return the same folded tensors, and a changed parameter new ones.
    [Fact]
    public void TheFold_KeepsItsIdentity_UntilAParameterChanges()
    {
        var c = Make(withBias: true);
        var first = CpuEngine.ConvBatchNormFold<float>.Get(c.Kernel, c.Bias, c.Gamma, c.Beta, c.Mean, c.Variance, 1e-5,
            CpuEngine.ConvBatchNormFold<float>.FoldOnHost);
        var again = CpuEngine.ConvBatchNormFold<float>.Get(c.Kernel, c.Bias, c.Gamma, c.Beta, c.Mean, c.Variance, 1e-5,
            CpuEngine.ConvBatchNormFold<float>.FoldOnHost);
        Assert.Same(first.Kernel, again.Kernel);
        Assert.Same(first.Bias, again.Bias);

        c.Gamma[0] = 2.5f;
        var rebuilt = CpuEngine.ConvBatchNormFold<float>.Get(c.Kernel, c.Bias, c.Gamma, c.Beta, c.Mean, c.Variance, 1e-5,
            CpuEngine.ConvBatchNormFold<float>.FoldOnHost);
        Assert.NotSame(first.Kernel, rebuilt.Kernel);
        Assert.NotSame(first.Bias, rebuilt.Bias);
        var epsilonChanged = CpuEngine.ConvBatchNormFold<float>.Get(c.Kernel, c.Bias, c.Gamma, c.Beta, c.Mean, c.Variance, 1e-3,
            CpuEngine.ConvBatchNormFold<float>.FoldOnHost);
        Assert.NotSame(rebuilt.Kernel, epsilonChanged.Kernel);
    }

    // Under a tape with recording suppressed the op is inference: it folds and records no node, so nothing upstream
    // receives a gradient through it.
    [Fact]
    public void UnderATape_WithNoGrad_FoldsAndRecordsNothing()
    {
        var cpu = new CpuEngine();
        var c = Make(withBias: true);
        var expected = Unfused(cpu, c, 1, 1, FusedActivationType.ReLU);
        using var tape = new GradientTape<float>();
        Tensor<float> y;
        using (new NoGradScope<float>()) y = Fused(cpu, c, 1, 1, FusedActivationType.ReLU);
        AssertClose(expected, y);
        var loss = cpu.ReduceMean(cpu.TensorMultiply(y, y), new[] { 0, 1, 2, 3 }, keepDims: false);
        var grads = tape.ComputeGradients(loss, new[] { c.Kernel, c.Gamma });
        Assert.False(grads.ContainsKey(c.Kernel), "a gradient reached the kernel through a no-grad call.");
        Assert.False(grads.ContainsKey(c.Gamma), "a gradient reached gamma through a no-grad call.");
    }

    // The tape path's gradients against central differences of the op itself, in double precision: numerics, not just
    // the routing a comparison with the same composition would check.
    [Fact]
    public void UnderATape_GradientsMatchCentralDifferences()
    {
        var cpu = new CpuEngine();
        Tensor<double> D(Tensor<float> t) => new(Array.ConvertAll(t.ToArray(), v => (double)v), t.Shape.ToArray());
        var f = Make(withBias: true);
        var x = D(f.Input); var k = D(f.Kernel); var b = D(f.Bias!); var g = D(f.Gamma); var be = D(f.Beta);
        var m = D(f.Mean); var v = D(f.Variance);
        var weights = new Tensor<double>(Array.ConvertAll(Rand(9, -1f, 1f, Batch, OutC, H, W).ToArray(), q => (double)q),
            new[] { Batch, OutC, H, W });
        Tensor<double> Loss() => cpu.ReduceMean(cpu.TensorMultiply(
            cpu.FusedConv2DBatchNorm(x, k, b, g, be, m, v, 1e-5, 1, 1, 1, 1, 1, 1, FusedActivationType.None), weights),
            new[] { 0, 1, 2, 3 }, keepDims: false);

        var sources = new[] { x, k, b, g, be };
        Dictionary<Tensor<double>, Tensor<double>> analytic;
        using (var tape = new GradientTape<double>()) analytic = tape.ComputeGradients(Loss(), sources);

        const double Eps = 1e-6;
        foreach (var source in sources)
        {
            var a = analytic[source].ToArray();
            for (int i = 0; i < source.Length; i += Math.Max(1, source.Length / 17))
            {
                double keep = source.GetFlatIndexValue(i);
                source.SetFlatIndexValue(i, keep + Eps); double up = Loss().ToArray()[0];
                source.SetFlatIndexValue(i, keep - Eps); double down = Loss().ToArray()[0];
                source.SetFlatIndexValue(i, keep);
                double numeric = (up - down) / (2 * Eps);
                Assert.True(Math.Abs(a[i] - numeric) <= 1e-6 + 1e-4 * Math.Abs(numeric),
                    $"gradient [{i}] of a [{string.Join(",", source.Shape.ToArray())}] source: analytic {a[i]}, numeric {numeric}.");
            }
        }
    }

    [SkippableFact]
    public void Gpu_UnderATape_GradientsMatchTheCpu()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend.");
        using var _ = gpu;
        var c = Make(withBias: true);
        var sources = new[] { c.Input, c.Kernel, c.Gamma, c.Beta };
        Tensor<float>[] Grads(IEngine e)
        {
            using var tape = new GradientTape<float>();
            var y = Fused(e, c, 1, 1, FusedActivationType.ReLU);
            var g = tape.ComputeGradients(e.ReduceMean(y, new[] { 0, 1, 2, 3 }, keepDims: false), sources);
            return Array.ConvertAll(sources, s => g[s]);
        }
        var expected = Grads(new CpuEngine());
        var actual = Grads(gpu!);
        for (int i = 0; i < sources.Length; i++) AssertClose(expected[i], actual[i], 1e-3f);
    }
}