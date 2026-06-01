using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.Cpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity test for the GPU RG-LRU scan path (#1464): the shared
/// <see cref="RecurrenceCpuKernels.RgLruForward"/> host reference (Metal/Vulkan/WebGpu fallback)
/// must match <see cref="CpuEngine.RgLruScanForward{T}"/>.
/// </summary>
public class RgLruScanGpuParityTests
{
    private static float[] Gen(int n, int s, float scale = 0.5f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.5 * (i + 1) + 0.9 * s) * scale);
        return a;
    }
    private static float[] GenSig(int n, int s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(1.0 / (1.0 + Math.Exp(-Math.Sin(0.5 * (i + 1) + 0.9 * s))));
        return a;
    }

    [Fact]
    public void HostReferenceForward_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 6, recDim = 5;
        var shape = new[] { batch, seqLen, recDim };
        var v = new Tensor<float>(Gen(batch * seqLen * recDim, 1), shape);
        var r = new Tensor<float>(GenSig(batch * seqLen * recDim, 2), shape);
        var ig = new Tensor<float>(GenSig(batch * seqLen * recDim, 3), shape);
        var decay = new Tensor<float>(Gen(recDim, 4, 0.8f), new[] { recDim });

        var reference = (float[])(object)engine.RgLruScanForward(v, r, ig, decay).GetDataArray()!;
        var hostOut = new float[batch * seqLen * recDim];
        RecurrenceCpuKernels.RgLruForward(F(v), F(r), F(ig), F(decay), hostOut, batch, seqLen, recDim);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - hostOut[i]) < 1e-5f,
                $"forward[{i}] cpu={reference[i]} host={hostOut[i]}");
    }

    private static float[] F(Tensor<float> t) => (float[])(object)t.GetDataArray()!;
}
