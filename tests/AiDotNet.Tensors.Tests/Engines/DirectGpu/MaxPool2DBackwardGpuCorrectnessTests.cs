using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Regression guard for the GPU MaxPool2D backward bugs that crashed CNN training
/// through the AiDotNet facade (AiDotNet#1468):
///   1. The engine uploaded pooling indices as a <c>float[]</c>, but every backend's
///      <c>maxpool2d_backward</c> kernel reads them as <c>const int*</c>. The kernel
///      reinterpreted float bit-patterns as ints (5 → 5.0f = 0x40A00000 ≈ 1.08e9),
///      producing an out-of-bounds scatter into gradInput — a hard CUDA access
///      violation (0xC0000005) and OpenCL memory corruption surfacing as
///      "Failed to read OpenCL buffer: -5".
///   2. The CUDA backend launched the 9-parameter kernel with only 8 arguments
///      (omitting outHeight), so the kernel's outWidth read a garbage arg slot.
///
/// Both bugs corrupt or crash the backward pass, so a GPU-vs-CPU equality check on the
/// input gradient catches them. NON-SQUARE spatial dims are used deliberately: a square
/// output (outHeight == outWidth) coincidentally masked bug #2.
/// </summary>
[Collection("DirectGpuSerial")]
public class MaxPool2DBackwardGpuCorrectnessTests
{
    [Fact]
    public void MaxPool2DBackward_Gpu_MatchesCpu_NonSquare()
    {
        var prior = AiDotNetEngine.Current;
        DirectGpuTensorEngine? gpu = null;
        try
        {
            gpu = new DirectGpuTensorEngine();
            if (!gpu.SupportsGpu) return; // CPU-only host: nothing to compare

            // Non-square input [batch=2, channels=3, H=6, W=4] → pool 2x2 stride 2
            // → output [2,3,3,2] (outHeight=3 != outWidth=2).
            const int batch = 2, channels = 3, inH = 6, inW = 4;
            var rng = new Random(1468);
            var inputData = new float[batch * channels * inH * inW];
            for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() * 4 - 2);
            var input = new Tensor<float>(inputData, new[] { batch, channels, inH, inW });

            var poolSize = new[] { 2, 2 };
            var stride = new[] { 2, 2 };

            // Forward (with indices) on CPU to get a stable index set + output shape.
            var cpu = new CpuEngine();
            var pooled = ((IEngine)cpu).MaxPool2DWithIndices(input, poolSize, stride, out var indices);

            // Upstream gradient shaped like the pooled output.
            var goData = new float[pooled.Length];
            for (int i = 0; i < goData.Length; i++) goData[i] = (float)(rng.NextDouble() + 0.1);
            var gradOutput = new Tensor<float>(goData, pooled.Shape.ToArray());

            var inputShape = new[] { batch, channels, inH, inW };
            var cpuGrad = ((IEngine)cpu).MaxPool2DBackward(gradOutput, indices, inputShape, poolSize, stride);

            AiDotNetEngine.Current = gpu;
            var gpuGrad = ((IEngine)gpu).MaxPool2DBackward(gradOutput, indices, inputShape, poolSize, stride);

            Assert.Equal(cpuGrad.Length, gpuGrad.Length);
            for (int i = 0; i < cpuGrad.Length; i++)
            {
                Assert.False(float.IsNaN(gpuGrad.GetFlatIndexValue(i)), $"GPU grad[{i}] is NaN");
                Assert.Equal(cpuGrad.GetFlatIndexValue(i), gpuGrad.GetFlatIndexValue(i), 3);
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            gpu?.Dispose();
        }
    }
}
