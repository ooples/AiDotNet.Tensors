// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// A standard model's inference on the GPU engine at batch 1 and 128, against the CPU engine (#1569). Op-level parity
/// tests run small shapes, and a batch-derived launch dimension past a device limit only appears at model scale: at
/// batch 128, ResNet-18's last stage puts batch x channels = 65,536 on one grid axis, the case that crashed CUDA
/// inference before #1567. The convolutions run through FusedConv2DBatchNorm, so the folded path is exercised too.
/// </summary>
[Collection("DirectGpuSerial")]
[Trait("Category", "ModelSmoke")]
public sealed class ModelInferenceSmokeTests
{
    private sealed class Conv
    {
        internal Tensor<float> Kernel = new(new[] { 0 });
        internal Tensor<float> Gamma = new(new[] { 0 }), Beta = new(new[] { 0 }), Mean = new(new[] { 0 }), Variance = new(new[] { 0 });
        internal int Stride, Pad;
    }

    private static Tensor<float> Rand(Random rng, float scale, params int[] shape)
    {
        int n = 1; foreach (var d in shape) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return new Tensor<float>(a, shape);
    }

    private static Conv MakeConv(Random rng, int inC, int outC, int k, int stride)
    {
        var variance = Rand(rng, 0.4f, outC);
        var v = variance.ToArray();
        for (int i = 0; i < v.Length; i++) v[i] = 0.6f + Math.Abs(v[i]);   // positive running variance
        return new Conv
        {
            Kernel = Rand(rng, (float)Math.Sqrt(2.0 / (inC * k * k)), outC, inC, k, k),
            Gamma = new Tensor<float>(Array.ConvertAll(Rand(rng, 0.2f, outC).ToArray(), g => 1f + g), new[] { outC }),
            Beta = Rand(rng, 0.1f, outC), Mean = Rand(rng, 0.1f, outC), Variance = new Tensor<float>(v, new[] { outC }),
            Stride = stride, Pad = k / 2,
        };
    }

    // ResNet-18 for 32x32 inputs: a 3x3 stem, four stages of two basic blocks (64, 128, 256, 512 channels), global
    // average pooling and a linear head.
    private sealed class ResNet18
    {
        internal Conv Stem = new();
        internal List<(Conv A, Conv B, Conv? Down)> Blocks = new();
        internal Tensor<float> Fc = new(new[] { 0 });
        internal Tensor<float> FcBias = new(new[] { 0 });

        internal static ResNet18 Create(int seed)
        {
            var rng = new Random(seed);
            var net = new ResNet18 { Stem = MakeConv(rng, 3, 64, 3, 1) };
            int inC = 64;
            foreach (var (outC, stride) in new[] { (64, 1), (128, 2), (256, 2), (512, 2) })
            {
                for (int block = 0; block < 2; block++)
                {
                    int s = block == 0 ? stride : 1;
                    var down = s != 1 || inC != outC ? MakeConv(rng, inC, outC, 1, s) : null;
                    net.Blocks.Add((MakeConv(rng, inC, outC, 3, s), MakeConv(rng, outC, outC, 3, 1), down));
                    inC = outC;
                }
            }
            net.Fc = Rand(rng, (float)Math.Sqrt(1.0 / 512), 10, 512);
            net.FcBias = Rand(rng, 0.05f, 10);
            return net;
        }

        private static Tensor<float> Apply(IEngine e, Conv c, Tensor<float> x, FusedActivationType act)
            => e.FusedConv2DBatchNorm(x, c.Kernel, null, c.Gamma, c.Beta, c.Mean, c.Variance, 1e-5,
                c.Stride, c.Stride, c.Pad, c.Pad, 1, 1, act);

        internal Tensor<float> Forward(IEngine e, Tensor<float> x)
        {
            var h = Apply(e, Stem, x, FusedActivationType.ReLU);
            foreach (var (a, b, down) in Blocks)
            {
                var y = Apply(e, b, Apply(e, a, h, FusedActivationType.ReLU), FusedActivationType.None);
                var shortcut = down is null ? h : Apply(e, down, h, FusedActivationType.None);
                h = e.ReLU(e.TensorAdd(y, shortcut));
            }
            var pooled = e.ReduceMean(h, new[] { 2, 3 }, keepDims: false);   // [B, 512]
            return e.TensorAdd(e.TensorMatMul(pooled, e.TensorTranspose(Fc)),
                e.TensorBroadcastTo(e.Reshape(FcBias, new[] { 1, 10 }), new[] { pooled.Shape[0], 10 }));
        }
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(128)]
    public void ResNet18_Inference_OnTheGpu_MatchesTheCpu(int batch)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        if (gpu is null || !gpu.IsGpuAvailable)
        {
            gpu?.Dispose();
            if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
                throw new InvalidOperationException("GPU required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no DirectGpu backend is available.");
            Skip.If(true, "No DirectGpu backend.");
        }
        using var _ = gpu;
        var net = ResNet18.Create(seed: 18);
        var input = Rand(new Random(batch), 1f, batch, 3, 32, 32);

        var expected = net.Forward(new CpuEngine(), input).ToArray();
        var actual = net.Forward(gpu!, input);
        Assert.Equal(new[] { batch, 10 }, actual.Shape.ToArray());
        var values = actual.ToArray();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(values[i]) || float.IsInfinity(values[i]), $"logit {i} is {values[i]}.");
            Assert.True(Math.Abs(expected[i] - values[i]) <= 1e-3f + 1e-3f * Math.Abs(expected[i]),
                $"batch {batch}, logit {i}: CPU {expected[i]}, GPU {values[i]}.");
        }
    }
}
