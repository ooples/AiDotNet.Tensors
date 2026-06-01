using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.Cpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity test for the GPU Gated DeltaNet scan path (#1464): the shared
/// <see cref="RecurrenceCpuKernels.GatedDeltaNetForward"/> host reference (used by the
/// Metal/Vulkan/WebGpu fallback) must match <see cref="CpuEngine.GatedDeltaNetScanForward{T}"/>.
/// </summary>
public class GatedDeltaNetScanGpuParityTests
{
    private static float[] Gen(int n, int s, float scale = 0.4f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.5 * (i + 1) + 1.1 * s) * scale);
        return a;
    }
    private static float[] GenSig(int n, int s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(1.0 / (1.0 + Math.Exp(-Math.Sin(0.5 * (i + 1) + 1.1 * s))));
        return a;
    }

    [Fact]
    public void HostReferenceForward_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, modelDim = 6, numHeads = 2;
        var shape = new[] { batch, seqLen, modelDim };
        var gShape = new[] { batch, seqLen, numHeads };
        var q = new Tensor<float>(Gen(batch * seqLen * modelDim, 1), shape);
        var k = new Tensor<float>(Gen(batch * seqLen * modelDim, 2), shape);
        var v = new Tensor<float>(Gen(batch * seqLen * modelDim, 3), shape);
        var a = new Tensor<float>(GenSig(batch * seqLen * numHeads, 4), gShape);
        var bta = new Tensor<float>(GenSig(batch * seqLen * numHeads, 5), gShape);

        var reference = (float[])(object)engine.GatedDeltaNetScanForward(q, k, v, a, bta, numHeads).GetDataArray()!;
        var hostOut = new float[batch * seqLen * modelDim];
        RecurrenceCpuKernels.GatedDeltaNetForward(
            F(q), F(k), F(v), F(a), F(bta), hostOut, batch, seqLen, modelDim, numHeads, modelDim / numHeads);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - hostOut[i]) < 1e-4f,
                $"forward[{i}] cpu={reference[i]} host={hostOut[i]}");
    }

    private static float[] F(Tensor<float> t) => (float[])(object)t.GetDataArray()!;
}
