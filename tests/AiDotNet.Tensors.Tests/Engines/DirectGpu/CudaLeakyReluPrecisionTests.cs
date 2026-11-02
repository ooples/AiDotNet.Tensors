using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CudaLeakyReluPrecisionTests
{
    [SkippableTheory]
    [InlineData(0.01f)]
    [InlineData(0.2f)]
    public void Fp32OperandsAreNotRoundedToHalf(float alpha)
    {
        CudaBackend backend;
        try { backend = new CudaBackend(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException or TypeInitializationException)
        {
            Skip.If(true, "CUDA unavailable: " + ex.GetType().Name);
            return;
        }
        using (backend)
        using (var a = backend.AllocateBuffer(new[] { 1.0003f }))
        using (var b = backend.AllocateBuffer(new[] { -1.0003f }))
        using (var bias = backend.AllocateBuffer(new[] { 0f }))
        using (var output = backend.GemmBiasLeakyRelu(a, b, bias, 1, 1, 1, alpha))
        {
            float expected = 1.0003f * -1.0003f * alpha;
            float actual = backend.DownloadBuffer(output)[0];
            Assert.True(Math.Abs(actual - expected) <= 1e-6f,
                $"FP32 product must survive the epilogue: expected {expected}, got {actual}.");
        }
    }
}
