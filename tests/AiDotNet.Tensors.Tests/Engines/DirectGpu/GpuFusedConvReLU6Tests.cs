using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A fused convolution with a ReLU6 epilogue (MobileNetV2's activation, which AiDotNet's conv + BatchNorm folding passes
/// to the engine) runs on the GPU engine and equals the CPU engine's result. The GPU activation switch had no ReLU6
/// case and threw, so every GPU MobileNetV2 predict crashed.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuFusedConvReLU6Tests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    [SkippableFact]
    public void FusedConv2D_WithReLU6_MatchesTheCpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        using (gpu)
        {
            // Large values so the clamp at 6 is exercised as well as the one at 0.
            var input = Filled(new[] { 2, 3, 9, 9 }, 1, 8f);
            var kernel = Filled(new[] { 5, 3, 3, 3 }, 2, 2f);
            var bias = Filled(new[] { 5 }, 3, 4f);
            var expected = new CpuEngine().FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU6).ToArray();
            var actual = gpu.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU6).ToArray();

            Assert.Contains(expected, v => v == 6f);
            Assert.Contains(expected, v => v == 0f);
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3f * (1f + Math.Abs(expected[i])), $"[{i}]: cpu {expected[i]}, gpu {actual[i]}");
        }
    }
}
