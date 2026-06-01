using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.Cpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity test for the GPU Mamba selective scan path (#1464): the shared
/// <see cref="RecurrenceCpuKernels.MambaSelectiveScanForward"/> host reference must match
/// <see cref="CpuEngine.MambaSelectiveScanForward{T}"/>.
/// </summary>
public class MambaScanGpuParityTests
{
    private static float[] Gen(int n, int s, float scale = 0.4f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.5 * (i + 1) + 0.6 * s) * scale);
        return a;
    }
    private static float[] GenPos(int n, int s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(0.1 + 0.4 * (0.5 + 0.5 * Math.Sin(0.5 * (i + 1) + 0.6 * s)));
        return a;
    }

    [Fact]
    public void HostReferenceForward_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, innerDim = 4, stateDim = 3;
        var x = new Tensor<float>(Gen(batch * seqLen * innerDim, 1), new[] { batch, seqLen, innerDim });
        var delta = new Tensor<float>(GenPos(batch * seqLen * innerDim, 2), new[] { batch, seqLen, innerDim });
        var aLog = new Tensor<float>(Gen(innerDim * stateDim, 3, 0.3f), new[] { innerDim, stateDim });
        var bP = new Tensor<float>(Gen(batch * seqLen * stateDim, 4), new[] { batch, seqLen, stateDim });
        var cP = new Tensor<float>(Gen(batch * seqLen * stateDim, 5), new[] { batch, seqLen, stateDim });
        var dP = new Tensor<float>(Gen(innerDim, 6, 0.3f), new[] { innerDim });

        var reference = (float[])(object)engine.MambaSelectiveScanForward(x, delta, aLog, bP, cP, dP).GetDataArray()!;
        var hostOut = new float[batch * seqLen * innerDim];
        RecurrenceCpuKernels.MambaSelectiveScanForward(
            F(x), F(delta), F(aLog), F(bP), F(cP), F(dP), hostOut, batch, seqLen, innerDim, stateDim);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - hostOut[i]) < 1e-4f,
                $"forward[{i}] cpu={reference[i]} host={hostOut[i]}");
    }

    private static float[] F(Tensor<float> t) => (float[])(object)t.GetDataArray()!;
}
