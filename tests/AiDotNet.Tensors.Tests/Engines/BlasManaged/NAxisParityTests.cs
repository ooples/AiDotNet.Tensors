using System;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasMgd = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using PB = AiDotNet.Tensors.Engines.BlasManaged.PackBothStrategy;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// The N-axis private-B parallel path (wide-N moderate-K float, panel kernel) must be
/// BIT-IDENTICAL to the M-axis shared-B path — same per-element K-reduction order, jc blocks
/// own disjoint C columns. Toggled via <c>PackBothStrategy.s_disableNAxis</c>.
/// </summary>
public class NAxisParityTests
{
    private static float[] Rand(int n, int seed)
    {
        var r = new Random(seed); var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() - 0.5);
        return a;
    }

    [Theory]
    [InlineData(384, 4096, 512)]   // wide-N, m mr-aligned → N-axis fires
    [InlineData(6, 2048, 777)]     // thin-M
    [InlineData(192, 3000, 1024)]  // non-pow2 N, odd K
    [InlineData(48, 8192, 256)]    // very wide N
    [InlineData(252, 1536, 1536)]  // mr-aligned moderate
    public void NAxis_BitIdentical_ToMAxis(int m, int n, int k)
    {
        var prior = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = 4;
        var a = Rand(m * k, 1);
        var b = Rand(k * n, 2);
        var cN = new float[m * n];
        var cM = new float[m * n];
        try
        {
            PB.s_disableNAxis = true;
            BlasMgd.Gemm<float>(a, k, false, b, n, false, cM, n, m, n, k);
            PB.s_disableNAxis = false;
            BlasMgd.Gemm<float>(a, k, false, b, n, false, cN, n, m, n, k);
        }
        finally
        {
            PB.s_disableNAxis = false;
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }

        for (int i = 0; i < cM.Length; i++)
            if (cM[i] != cN[i])
                Assert.Fail($"N-axis diverged at [{i / n},{i % n}] m={m} n={n} k={k}: Maxis={cM[i]:R} Naxis={cN[i]:R}");
    }
}
