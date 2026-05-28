using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Regression guard for the lean output-only SDPA in the fused MHA float forward
/// (<see cref="CpuEngine.ScaledDotProductAttentionFloatInto"/>). MHA discards the attention
/// weights, so the forward routes through the output-only kernel instead of the general
/// ScaledDotProductAttention, which would allocate a weightsData array, an attentionWeights
/// tensor and an output tensor and then copy out. On [128,32,64] h=4 that cut per-call
/// allocation from ~7.2 MB to ~1.0 MB (~6.9×) and sustained mean latency ~2.7×. This test
/// fails if the forward regresses back to the weight-materializing path.
/// </summary>
public class MhaLeanSdpaAllocTests
{
    private static Tensor<float> Rand(Random r, params int[] s)
    {
        var t = new Tensor<float>(s);
        var sp = t.AsWritableSpan();
        for (int i = 0; i < sp.Length; i++) sp[i] = (float)(r.NextDouble() * 2 - 1);
        return t;
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void MhaForward_AllocatesFarLessThanWeightMaterializingPath()
    {
        var eng = new CpuEngine();
        const int batch = 128, seq = 32, dModel = 64, numHeads = 4;
        var r = new Random(2026);
        var input = Rand(r, batch, seq, dModel);
        var qW = Rand(r, dModel, dModel); var kW = Rand(r, dModel, dModel);
        var vW = Rand(r, dModel, dModel); var oW = Rand(r, dModel, dModel);

        for (int w = 0; w < 5; w++) _ = eng.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);

        const int n = 20;
        long a0 = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < n; i++) _ = eng.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        long a1 = GC.GetAllocatedBytesForCurrentThread();
        double kbPerCall = (a1 - a0) / 1024.0 / n;

        // Lean path measures ~1.0 MB/call; the old weight-materializing path was ~7.2 MB/call.
        // 3 MB threshold catches a regression with wide margin against measurement noise.
        Assert.True(kbPerCall < 3072,
            $"MHA forward allocated {kbPerCall:F0} KB/call; expected < 3072 KB (lean ~1024, " +
            "old weight-materializing path ~7200). Regressed to the non-lean SDPA?");
    }
#endif
}
