#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The GPU engine's inference BatchNorm with a fused activation against the CPU engine: ReLU and ReLU6 through the
/// one-pass inference kernel (MobileNet's ReLU6 had no device route and ran on the host), and an activation with no
/// fused kernel through the device batch norm followed by the engine's activation.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuBatchNormInferenceActivationTests
{
    [SkippableTheory]
    [InlineData(FusedActivationType.ReLU)]
    [InlineData(FusedActivationType.ReLU6)]
    [InlineData(FusedActivationType.GELU)]
    [InlineData(FusedActivationType.HardSwish)]
    public void Gpu_MatchesCpu(FusedActivationType activation)
    {
        DirectGpuTensorEngine? engine = null;
        try { engine = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or TypeInitializationException) { }
        Skip.If(engine is null || !engine.IsGpuAvailable, "No DirectGpu backend available.");
        using (engine)
        {
            const int n = 2, c = 5, h = 3, w = 4;
            var xv = new float[n * c * h * w];
            for (int i = 0; i < xv.Length; i++) xv[i] = ((i * 37 % 41) - 20) * 0.6f;   // spans below 0 and above 6
            var x = new Tensor<float>(xv, new[] { n, c, h, w });
            var gamma = new Tensor<float>(new[] { 1.5f, 0.5f, 2f, 1f, 3f }, new[] { c });
            var beta = new Tensor<float>(new[] { 0.5f, -1f, 2f, 0f, 1f }, new[] { c });
            var mean = new Tensor<float>(new[] { 0.1f, -0.2f, 0.3f, 0f, 1f }, new[] { c });
            var variance = new Tensor<float>(new[] { 1f, 2f, 0.5f, 4f, 3f }, new[] { c });
            var expected = new CpuEngine().BatchNormInference(x, gamma, beta, mean, variance, 1e-5, activation).ToArray();
            var actual = engine.BatchNormInference(x, gamma, beta, mean, variance, 1e-5, activation).ToArray();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * (1 + Math.Abs(expected[i])),
                    $"{activation} [{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }
}
#endif