using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.Cpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity test for the GPU RWKV-4 WKV scan path (#1464): the shared
/// <see cref="RecurrenceCpuKernels.Rwkv4WkvForward"/> host reference (Metal/Vulkan/WebGpu fallback)
/// must match <see cref="CpuEngine.Rwkv4WkvForward{T}"/>.
/// </summary>
public class Rwkv4WkvGpuParityTests
{
    private static float[] Gen(int n, int s, float scale = 0.5f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.5 * (i + 1) + 0.7 * s) * scale);
        return a;
    }

    [Fact]
    public void HostReferenceForward_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 6, modelDim = 5;
        var shape = new[] { batch, seqLen, modelDim };
        var r = new Tensor<float>(Gen(batch * seqLen * modelDim, 1), shape);
        var k = new Tensor<float>(Gen(batch * seqLen * modelDim, 2), shape);
        var v = new Tensor<float>(Gen(batch * seqLen * modelDim, 3), shape);
        var td = new Tensor<float>(Gen(modelDim, 4, 0.3f), new[] { modelDim });
        var tf = new Tensor<float>(Gen(modelDim, 5, 0.3f), new[] { modelDim });

        var reference = (float[])(object)engine.Rwkv4WkvForward(r, k, v, td, tf).GetDataArray()!;
        var hostOut = new float[batch * seqLen * modelDim];
        RecurrenceCpuKernels.Rwkv4WkvForward(F(r), F(k), F(v), F(td), F(tf), hostOut, batch, seqLen, modelDim);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - hostOut[i]) < 1e-4f,
                $"forward[{i}] cpu={reference[i]} host={hostOut[i]}");
    }

    private static float[] F(Tensor<float> t) => (float[])(object)t.GetDataArray()!;
}
