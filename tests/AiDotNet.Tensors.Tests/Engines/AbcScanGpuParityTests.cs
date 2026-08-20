using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// GPU parity for the ABC slot recurrence (issue ooples/AiDotNet#1464).
/// <see cref="DirectGpuTensorEngine.AbcScanForward{T}"/> has no dedicated device kernel; it composes
/// the recurrence from GPU-resident primitives (one batched GEMM plus a softmax for the whole
/// sequence of write scores, then a per-step state update and read). These tests run that composed
/// path on a real device and require it to agree with the differentiable
/// <see cref="CpuEngine.AbcScanForward{T}"/> that owns the tape node — if the two disagree, a model
/// silently changes behaviour between training and GPU inference.
/// They SKIP when no backend is present rather than passing vacuously.
/// </summary>
[Collection("DirectGpuSerial")]
public class AbcScanGpuParityTests
{
    private const double InitScale = 0.1;

    private static float[] Gen(int n, int s, float scale = 0.5f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.5 * (i + 1) + 1.3 * s) * scale);
        return a;
    }

    // The forget gate is a sigmoid output in (0,1).
    private static float[] GenGate(int n, int s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(1.0 / (1.0 + Math.Exp(-Math.Sin(0.5 * (i + 1) + 1.3 * s))));
        return a;
    }

    /// <summary>
    /// Runs the composed op sequence itself — the exact code the GPU override executes — through a
    /// CpuEngine and requires it to reproduce the fused kernel. This is what actually validates the
    /// composition's shape algebra and math; it needs no device, so unlike the parity test below it
    /// can never pass vacuously. On real hardware only the dispatch target changes.
    /// </summary>
    [Theory]
    [InlineData(1, 4, 8, 2, 3)]
    [InlineData(2, 5, 12, 3, 4)]
    [InlineData(2, 1, 8, 4, 2)]   // seqLen 1 takes the no-concat path
    [InlineData(3, 6, 4, 1, 5)]   // single head
    public void ComposedOpSequence_MatchesFusedKernel(int batch, int seqLen, int modelDim, int numHeads, int numSlots)
    {
        var cpu = new CpuEngine();
        int headDim = modelDim / numHeads;
        var (q, k, v, fg, sk) = MakeInputs(batch, seqLen, modelDim, numHeads, numSlots, headDim);
        using var qOwner = q;
        using var kOwner = k;
        using var vOwner = v;
        using var fgOwner = fg;
        using var skOwner = sk;

        using var expectedTensor = cpu.AbcScanForward(q, k, v, fg, sk, numHeads, InitScale);
        using var composedTensor = DirectGpuTensorEngine.AbcScanComposed(
            cpu, q, k, v, fg, sk, batch, seqLen, modelDim, numHeads, headDim, numSlots, InitScale);
        var expected = (float[])(object)expectedTensor.GetDataArray()!;
        var composed = (float[])(object)composedTensor.GetDataArray()!;

        Assert.Equal(expected.Length, composed.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - composed[i]) < 1e-4f,
                $"[{batch}x{seqLen}x{modelDim}, H={numHeads}, N={numSlots}] element {i}: " +
                $"fused={expected[i]} composed={composed[i]}");
    }

    private static (Tensor<float> q, Tensor<float> k, Tensor<float> v, Tensor<float> fg, Tensor<float> sk)
        MakeInputs(int batch, int seqLen, int modelDim, int numHeads, int numSlots, int headDim)
    {
        var shape = new[] { batch, seqLen, modelDim };
        return (new Tensor<float>(Gen(batch * seqLen * modelDim, 1), shape),
                new Tensor<float>(Gen(batch * seqLen * modelDim, 2), shape),
                new Tensor<float>(Gen(batch * seqLen * modelDim, 3), shape),
                new Tensor<float>(GenGate(batch * seqLen * numHeads, 4), new[] { batch, seqLen, numHeads }),
                new Tensor<float>(Gen(numHeads * numSlots * headDim, 5), new[] { numHeads, numSlots, headDim }));
    }

    [SkippableTheory]
    [InlineData(1, 4, 8, 2, 3)]
    [InlineData(2, 5, 12, 3, 4)]
    [InlineData(2, 1, 8, 4, 2)]   // seqLen 1 takes the no-concat path
    public void GpuComposedPath_MatchesCpuEngine(int batch, int seqLen, int modelDim, int numHeads, int numSlots)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        int headDim = modelDim / numHeads;
        var (q, k, v, fg, sk) = MakeInputs(batch, seqLen, modelDim, numHeads, numSlots, headDim);
        using var qOwner = q;
        using var kOwner = k;
        using var vOwner = v;
        using var fgOwner = fg;
        using var skOwner = sk;

        using var expectedTensor = new CpuEngine().AbcScanForward(q, k, v, fg, sk, numHeads, InitScale);
        var expected = (float[])(object)expectedTensor.GetDataArray()!;

        bool savedThrowOnFallback = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        bool savedCaptureReadbackSites = GpuLaunchProbe.CaptureReadbackSites;
        Tensor<float>? gpuResult = null;
        try
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
            GpuLaunchProbe.CaptureReadbackSites = true;
            GpuLaunchProbe.Reset();
            // Invoke the composition directly: the public wrapper must not be able to hide a
            // primitive failure by returning the CPU reference result.
            gpuResult = DirectGpuTensorEngine.AbcScanComposed(
                gpu, q, k, v, fg, sk, batch, seqLen, modelDim, numHeads, headDim, numSlots, InitScale);

            Assert.True(GpuLaunchProbe.Count > 0, "ABC composition launched no GPU work.");
            Assert.True(GpuLaunchProbe.Readbacks == 0,
                $"ABC composition performed {GpuLaunchProbe.Readbacks} device-to-host transfers: " +
                string.Join("; ", GpuLaunchProbe.ReadbackSites));
            Assert.Empty(GpuLaunchProbe.Fallbacks);

            var got = (float[])(object)gpuResult.GetDataArray()!;

            Assert.Equal(expected.Length, got.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - got[i]) < 1e-4f,
                    $"[{batch}x{seqLen}x{modelDim}, H={numHeads}, N={numSlots}] element {i}: cpu={expected[i]} gpu={got[i]}");
        }
        finally
        {
            gpuResult?.Dispose();
            GpuLaunchProbe.CaptureReadbackSites = savedCaptureReadbackSites;
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = savedThrowOnFallback;
        }
    }
}
