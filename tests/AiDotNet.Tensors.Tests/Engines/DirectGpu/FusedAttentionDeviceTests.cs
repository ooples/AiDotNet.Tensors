using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1141: the fused device attention (weightless SDPA and MultiHeadAttentionCore) equals the CPU fused kernels in its
/// output and its Q / K / V gradients -- masks (broadcast, with fully masked rows), causal masking (top-left for SDPA,
/// bottom-right for the attention core), soft-cap, unequal query / key lengths -- runs without a host fallback, and is
/// deterministic.
/// </summary>
[Collection("VulkanGlobalState")]
public class FusedAttentionDeviceTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public FusedAttentionDeviceTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private IEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        DirectGpuTensorEngine gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is IFusedAttentionBackend fused && fused.SupportsFusedAttention(64),
            "The fused attention kernels are CUDA-only.");
        return gpu;
    }

    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        int n = shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<float>(Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(), shape);
    }

    private static Tensor<bool> Mask(int seed, double keep, params int[] shape)
    {
        var rng = new Random(seed);
        int n = shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<bool>(Enumerable.Range(0, n).Select(_ => rng.NextDouble() < keep).ToArray(), shape);
    }

    // Output and the three gradients of sum(out * w) through the tape.
    private static (float[] O, float[] DQ, float[] DK, float[] DV) Step(IEngine e, Func<Tensor<float>, Tensor<float>, Tensor<float>, Tensor<float>> f,
        Tensor<float> q, Tensor<float> k, Tensor<float> v, Tensor<float> w)
    {
        var (qc, kc, vc) = ((Tensor<float>)q.Clone(), (Tensor<float>)k.Clone(), (Tensor<float>)v.Clone());
        using var tape = new GradientTape<float>();
        var o = f(qc, kc, vc);
        var loss = e.ReduceSum(e.TensorMultiply(o, w), null);
        var g = tape.ComputeGradients(loss, new[] { qc, kc, vc });
        return (o.ToArray(), g[qc].ToArray(), g[kc].ToArray(), g[vc].ToArray());
    }

    private static void AssertClose(string what, float[] cpu, float[] gpu, float tol = 2e-4f)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.False(float.IsNaN(gpu[i]) || float.IsInfinity(gpu[i]), $"{what}[{i}]: gpu {gpu[i]} is not finite");
            Assert.True(Math.Abs(cpu[i] - gpu[i]) <= tol * (1 + Math.Abs(cpu[i])), $"{what}[{i}]: cpu {cpu[i]} gpu {gpu[i]}");
        }
    }

    private void CheckSdpa(IEngine gpu, int b, int h, int sq, int sk, int d, Tensor<bool>? mask, bool causal, double softcap, string name)
    {
        var cpu = new CpuEngine();
        var q = Random(1, b, h, sq, d); var k = Random(2, b, h, sk, d); var v = Random(3, b, h, sk, d); var w = Random(4, b, h, sq, d);
        Tensor<float> F(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float> z) => e.ScaledDotProductAttention(x, y, z, mask, null, causal, softcap);
        bool saved = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var c = Step(cpu, (x, y, z) => F(cpu, x, y, z), q, k, v, w);
            var g = Step(gpu, (x, y, z) => F(gpu, x, y, z), q, k, v, w);
            AssertClose($"{name} out", c.O, g.O);
            AssertClose($"{name} dQ", c.DQ, g.DQ);
            AssertClose($"{name} dK", c.DK, g.DK);
            AssertClose($"{name} dV", c.DV, g.DV);
        }
        finally { DirectGpuTensorEngine.ThrowOnGpuKernelFallback = saved; }
    }

    [SkippableTheory]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(64)]
    public void Sdpa_MatchesCpu_EveryOption(int d)
    {
        var gpu = GpuOrSkip();
        CheckSdpa(gpu, 2, 3, 37, 37, d, null, false, 0, "plain");
        CheckSdpa(gpu, 2, 3, 37, 37, d, null, true, 0, "causal");
        CheckSdpa(gpu, 2, 3, 21, 45, d, null, true, 0, "causal Sq<Sk");
        CheckSdpa(gpu, 2, 3, 45, 21, d, null, true, 0, "causal Sq>Sk");
        CheckSdpa(gpu, 2, 3, 37, 70, d, Mask(5, 0.6, 37, 70), false, 0, "mask [Sq, Sk]");
        // A [B, 1, Sq, Sk] mask with some rows fully masked: those rows output zeros and pass no gradient.
        var rowMask = Mask(6, 0.5, 2, 1, 33, 40);
        for (int j = 0; j < 40; j++) { rowMask[0, 0, 4, j] = false; rowMask[1, 0, 17, j] = false; }
        CheckSdpa(gpu, 2, 3, 33, 40, d, rowMask, false, 0, "mask [B, 1, Sq, Sk] with empty rows");
        CheckSdpa(gpu, 2, 3, 37, 37, d, null, false, 5.0, "softcap");
        CheckSdpa(gpu, 2, 3, 37, 70, d, Mask(7, 0.7, 2, 1, 1, 70), true, 3.0, "mask + causal + softcap");
    }

    [SkippableFact]
    public void Sdpa_LongSequence_MatchesCpu()
    {
        var gpu = GpuOrSkip();
        CheckSdpa(gpu, 1, 2, 300, 300, 64, null, true, 0, "S=300 causal");
    }

    [SkippableTheory]
    [InlineData(false, 29, 29)]
    [InlineData(true, 29, 29)]
    [InlineData(true, 13, 41)]   // bottom-right causal: the last query sees every key
    public void MultiHeadAttentionCore_MatchesCpu(bool causal, int sq, int sk)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        const int b = 2, h = 4, d = 32;
        var q = Random(11, b, sq, h * d); var k = Random(12, b, sk, h * d); var v = Random(13, b, sk, h * d); var w = Random(14, b, sq, h * d);
        bool saved = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var c = Step(cpu, (x, y, z) => cpu.MultiHeadAttentionCore(x, y, z, h, null, causal), q, k, v, w);
            var g = Step(gpu, (x, y, z) => gpu.MultiHeadAttentionCore(x, y, z, h, null, causal), q, k, v, w);
            AssertClose("mha out", c.O, g.O);
            AssertClose("mha dQ", c.DQ, g.DQ);
            AssertClose("mha dK", c.DK, g.DK);
            AssertClose("mha dV", c.DV, g.DV);
        }
        finally { DirectGpuTensorEngine.ThrowOnGpuKernelFallback = saved; }
    }

    [SkippableFact]
    public void Sdpa_RunsOnDevice_AndIsDeterministic()
    {
        var gpu = GpuOrSkip();
        var q = Random(21, 1, 4, 129, 64); var k = Random(22, 1, 4, 129, 64); var v = Random(23, 1, 4, 129, 64); var w = Random(24, 1, 4, 129, 64);
        GpuLaunchProbe.Reset();
        var first = Step(gpu, (x, y, z) => gpu.ScaledDotProductAttention(x, y, z, null, null, true, 0.0), q, k, v, w);
        Assert.Empty(GpuLaunchProbe.Fallbacks);
        var second = Step(gpu, (x, y, z) => gpu.ScaledDotProductAttention(x, y, z, null, null, true, 0.0), q, k, v, w);
        Assert.Equal(first.O, second.O);
        Assert.Equal(first.DQ, second.DQ);
        Assert.Equal(first.DK, second.DK);
        Assert.Equal(first.DV, second.DV);
    }
}
