// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Pre-activated convolution (#2561): conv(relu(bn(x))), with an optional batch norm after the convolution folded into
/// it. On CUDA the first batch norm and ReLU are applied as the convolution engine loads its input. Each case is checked
/// against the unfused composition on the CPU.
/// </summary>
public sealed class BatchNormActivationConv2DTests
{
    private static Tensor<float> Rand(int seed, float lo, float hi, params int[] shape)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return new Tensor<float>(data, shape);
    }

    private sealed record Bn(Tensor<float> Gamma, Tensor<float> Beta, Tensor<float> Mean, Tensor<float> Variance);

    private static Bn MakeBn(int seed, int c) => new(Rand(seed, 0.5f, 1.5f, c), Rand(seed + 1, -0.3f, 0.3f, c),
        Rand(seed + 2, -0.2f, 0.2f, c), Rand(seed + 3, 0.1f, 2f, c));

    private static Tensor<float> Reference(IEngine e, Tensor<float> x, Bn pre, Tensor<float> kernel, Tensor<float>? bias,
        Bn? post, int stride, int pad, FusedActivationType act)
    {
        var activated = e.BatchNormInference(x, pre.Gamma, pre.Beta, pre.Mean, pre.Variance, 1e-5, FusedActivationType.ReLU);
        if (post is null) return e.FusedConv2D(activated, kernel, bias, stride, stride, pad, pad, 1, 1, act);
        var conv = e.FusedConv2D(activated, kernel, bias, stride, stride, pad, pad, 1, 1, FusedActivationType.None);
        return e.BatchNormInference(conv, post.Gamma, post.Beta, post.Mean, post.Variance, 1e-5, act);
    }

    private static Tensor<float> Fused(IEngine e, Tensor<float> x, Bn pre, Tensor<float> kernel, Tensor<float>? bias,
        Bn? post, int stride, int pad, FusedActivationType act)
        => post is null
            ? e.BatchNormActivationConv2D(x, pre.Gamma, pre.Beta, pre.Mean, pre.Variance, 1e-5, kernel, bias,
                stride, stride, pad, pad, 1, 1, act)
            : e.BatchNormActivationConv2DBatchNorm(x, pre.Gamma, pre.Beta, pre.Mean, pre.Variance, 1e-5, kernel, bias,
                post.Gamma, post.Beta, post.Mean, post.Variance, 1e-5, stride, stride, pad, pad, 1, 1, act);

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tol)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tol * (1 + Math.Abs(e[i])), $"element {i}: expected {e[i]}, got {a[i]}");
    }

    public static TheoryData<int, int, int, bool, bool, FusedActivationType> Cases => new()
    {
        // inC, kernel, stride, bias, post batch norm, activation. inC 16 takes the engine's tap-uniform kernels, 3 the
        // per-element ones; 3x3 with padding checks that padding stays zero after the prologue.
        { 16, 1, 1, false, true, FusedActivationType.ReLU },     // DenseNet's dense layer
        { 16, 1, 1, false, false, FusedActivationType.None },    // DenseNet's transition
        { 16, 3, 1, true, false, FusedActivationType.ReLU },
        { 3, 3, 2, true, true, FusedActivationType.None },
        { 3, 1, 1, true, false, FusedActivationType.ReLU },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Cpu_MatchesBatchNormReluThenConvolution(int inC, int k, int stride, bool withBias, bool withPost,
        FusedActivationType act)
    {
        var cpu = new CpuEngine();
        const int outC = 12;
        var x = Rand(1, -1f, 1f, 2, inC, 9, 7);
        var pre = MakeBn(10, inC);
        var kernel = Rand(2, -0.5f, 0.5f, outC, inC, k, k);
        var bias = withBias ? Rand(3, -0.2f, 0.2f, outC) : null;
        var post = withPost ? MakeBn(20, outC) : null;
        int pad = k / 2;
        AssertClose(Reference(cpu, x, pre, kernel, bias, post, stride, pad, act),
            Fused(cpu, x, pre, kernel, bias, post, stride, pad, act), 1e-4f);
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Gpu_MatchesTheCpuComposition(int inC, int k, int stride, bool withBias, bool withPost,
        FusedActivationType act)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend.");
        using var _ = gpu;
        const int outC = 12;
        var x = Rand(1, -1f, 1f, 2, inC, 9, 7);
        var pre = MakeBn(10, inC);
        var kernel = Rand(2, -0.5f, 0.5f, outC, inC, k, k);
        var bias = withBias ? Rand(3, -0.2f, 0.2f, outC) : null;
        var post = withPost ? MakeBn(20, outC) : null;
        int pad = k / 2;
        var expected = Reference(new CpuEngine(), x, pre, kernel, bias, post, stride, pad, act);
        // TF32 convolutions (the CUDA default, as PyTorch's): ~1e-3 of the output scale.
        AssertClose(expected, Fused(gpu, x, pre, kernel, bias, post, stride, pad, act), 5e-3f);
        // A second call reuses the cached batch-norm affine and fold.
        AssertClose(expected, Fused(gpu, x, pre, kernel, bias, post, stride, pad, act), 5e-3f);
    }

    [Fact]
    public void ChangingThePreActivationStatistics_RebuildsTheAffine()
    {
        var cpu = new CpuEngine();
        var x = Rand(1, -1f, 1f, 2, 4, 5, 5);
        var pre = MakeBn(10, 4);
        var kernel = Rand(2, -0.5f, 0.5f, 6, 4, 1, 1);
        Fused(cpu, x, pre, kernel, null, null, 1, 0, FusedActivationType.None);
        var changed = pre with { Variance = Rand(99, 0.1f, 2f, 4) };
        AssertClose(Reference(cpu, x, changed, kernel, null, null, 1, 0, FusedActivationType.None),
            Fused(cpu, x, changed, kernel, null, null, 1, 0, FusedActivationType.None), 1e-4f);
        // In place: the same tensor written with new values.
        pre.Mean[0] = 0.75f;
        AssertClose(Reference(cpu, x, pre, kernel, null, null, 1, 0, FusedActivationType.None),
            Fused(cpu, x, pre, kernel, null, null, 1, 0, FusedActivationType.None), 1e-4f);
    }
}
