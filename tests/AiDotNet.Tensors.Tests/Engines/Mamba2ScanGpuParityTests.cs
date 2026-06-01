using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.Cpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity test for the GPU Mamba-2 SSD scan path (#1464): the shared
/// <see cref="RecurrenceCpuKernels.Mamba2SsdScanForward"/> host reference must match
/// <see cref="CpuEngine.Mamba2SsdScanForward{T}"/>.
/// </summary>
public class Mamba2ScanGpuParityTests
{
    private static float[] Gen(int n, int s, float scale = 0.4f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.5 * (i + 1) + 0.55 * s) * scale);
        return a;
    }
    private static float[] GenPos(int n, int s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(0.1 + 0.4 * (0.5 + 0.5 * Math.Sin(0.5 * (i + 1) + 0.55 * s)));
        return a;
    }

    [Fact]
    public void HostReferenceForward_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, numHeads = 2, headDim = 3, stateDim = 3;
        int innerDim = numHeads * headDim;
        var x = new Tensor<float>(Gen(batch * seqLen * innerDim, 1), new[] { batch, seqLen, innerDim });
        var delta = new Tensor<float>(GenPos(batch * seqLen * numHeads, 2), new[] { batch, seqLen, numHeads });
        var aLog = new Tensor<float>(Gen(numHeads, 3, 0.3f), new[] { numHeads });
        var bP = new Tensor<float>(Gen(batch * seqLen * stateDim, 4), new[] { batch, seqLen, stateDim });
        var cP = new Tensor<float>(Gen(batch * seqLen * stateDim, 5), new[] { batch, seqLen, stateDim });
        var dP = new Tensor<float>(Gen(numHeads, 6, 0.3f), new[] { numHeads });

        var reference = (float[])(object)engine.Mamba2SsdScanForward(x, delta, aLog, bP, cP, dP, numHeads).GetDataArray()!;
        var hostOut = new float[batch * seqLen * innerDim];
        RecurrenceCpuKernels.Mamba2SsdScanForward(
            F(x), F(delta), F(aLog), F(bP), F(cP), F(dP), hostOut, batch, seqLen, innerDim, numHeads, headDim, stateDim);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - hostOut[i]) < 1e-4f,
                $"forward[{i}] cpu={reference[i]} host={hostOut[i]}");
    }

    private static float[] F(Tensor<float> t) => (float[])(object)t.GetDataArray()!;
}
