using System;
using System.Linq;
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

    /// <summary>
    /// A winnerless window (every element -inf, or NaN) routes its gradient to its own first in-bounds cell, as the CPU
    /// engine and PyTorch do. The CUDA forward stored cell 0 of the plane for such a window, and the deterministic
    /// backward scanned the whole output plane for cell 0 to honour it -- one slow thread per plane, ~135 us per step on
    /// the parity CNN. Padded and unpadded pools; the backward runs under GpuDeterminism (the bounded gather kernel).
    /// </summary>
    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    public void WinnerlessWindow_RoutesToItsFirstCell_InTheForwardAndTheDeterministicBackward(int pad)
    {
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        const int batch = 1, channels = 2, inH = 6, inW = 6, k = 2, s = 2;
        int outH = (inH + 2 * pad - k) / s + 1, outW = (inW + 2 * pad - k) / s + 1;
        var x = new float[batch * channels * inH * inW];
        for (int i = 0; i < x.Length; i++) x[i] = (i % 7) * 0.25f - 1f;
        // Channel 0: window (1,1) -- rows and cols [2 - pad, 3 - pad] -- is all -inf; channel 1: the same cells all NaN.
        foreach (int h in new[] { 2 - pad, 3 - pad })
            foreach (int w in new[] { 2 - pad, 3 - pad })
            {
                x[h * inW + w] = float.NegativeInfinity;
                x[inH * inW + h * inW + w] = float.NaN;
            }
        var g = new float[batch * channels * outH * outW];
        for (int i = 0; i < g.Length; i++) g[i] = 1f + i;

        using var xBuf = backend.AllocateBuffer(x);
        using var outBuf = backend.AllocateBuffer(g.Length);
        using var idxBuf = backend.AllocateBuffer(g.Length);
        using var gBuf = backend.AllocateBuffer(g);
        using var dxBuf = backend.AllocateBuffer(Enumerable.Repeat(123.25f, x.Length).ToArray());
        backend.MaxPool2D(xBuf, outBuf, idxBuf, batch, channels, inH, inW, outH, outW, k, k, s, s, pad, pad);
        var prior = AiDotNet.Tensors.Helpers.BlasProvider.GetThreadLocalDeterministicMode();
        AiDotNet.Tensors.Helpers.BlasProvider.SetThreadLocalDeterministicMode(true);
        try
        {
            backend.MaxPool2DBackward(gBuf, idxBuf, dxBuf, batch, channels, inH, inW, outH, outW, k, k, s, s, pad, pad);
        }
        finally
        {
            AiDotNet.Tensors.Helpers.BlasProvider.SetThreadLocalDeterministicMode(prior);
        }

        // Reference: each window's winner (first strict maximum in scan order, else its first in-bounds cell) takes the
        // window's gradient.
        var indices = backend.DownloadBuffer(idxBuf);
        var expected = new float[x.Length];
        for (int c = 0; c < channels; c++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    int firstH = Math.Min(Math.Max(oh * s - pad, 0), inH - 1), firstW = Math.Min(Math.Max(ow * s - pad, 0), inW - 1);
                    int winner = firstH * inW + firstW;
                    float best = float.NegativeInfinity;
                    for (int kh = 0; kh < k; kh++)
                        for (int kw = 0; kw < k; kw++)
                        {
                            int ih = oh * s - pad + kh, iw = ow * s - pad + kw;
                            if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                            float v = x[c * inH * inW + ih * inW + iw];
                            if (v > best) { best = v; winner = ih * inW + iw; }
                        }
                    int o = (c * outH + oh) * outW + ow;
                    int stored = BitConverter.ToInt32(BitConverter.GetBytes(indices[o]), 0);
                    Assert.True(stored == winner, $"pad {pad}, channel {c}, window ({oh},{ow}): index {stored}, expected {winner}");
                    expected[c * inH * inW + winner] += g[o];
                }
        var got = backend.DownloadBuffer(dxBuf);
        for (int i = 0; i < x.Length; i++)
            Assert.True(got[i] == expected[i], $"pad {pad}: gradInput[{i}] = {got[i]}, expected {expected[i]}");
    }
}
