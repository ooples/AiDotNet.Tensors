#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged.Jit;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #685: the intermittent AVX-512 FP32 GEMM drift (Auto vs PackBoth off by ~500x the tolerance at 192x768x768 and
/// 193x770x384). Both shapes are wide-N (n >= k), so PackBoth took its N-axis path, which computes with the 6x16 FP32
/// panel kernel but packed A with the strategy's tile: under AVX-512 that tile is 16x16, so the panel kernel read
/// 16-row A stripes as 6-row ones. Whether the path was taken depended on IsFp32PanelAvailable's lazy init (racy until
/// #1103) and the fan-out, hence intermittent. #700/#702 gated the path to mr == MachineKernelGemm.Fp32Mr; the CI job
/// that could have caught a regression ran on non-AVX-512 runners. This pins the gate on any x86 host by calling the
/// strategy with the AVX-512 tile directly.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class PackBothNAxisTileGateTests
{
    private static (float[] A, float[] B, double[] C) Problem(int m, int n, int k)
    {
        var rng = new Random(685 + m + n + k);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++)
            {
                double av = a[i * k + p];
                for (int j = 0; j < n; j++) c[i * n + j] += av * b[p * n + j];
            }
        return (a, b, c);
    }

    // The issue's aligned shape, with the AVX-512 FP32 tile (16x16): the N-axis plan must be refused, not run with a
    // packed A it would misread.
    [SkippableFact]
    public void NAxisPlan_IsRefused_ForTheAvx512Tile()
    {
        Skip.IfNot(MachineKernelGemm.IsFp32PanelAvailable, "Needs the FP32 machine-code panel kernel (x86 AVX2+FMA).");
        var (a, b, _) = Problem(192, 768, 768);
        var c = new float[192 * 768];
        Assert.Throws<NotSupportedException>(() => PackBothStrategy.Run<float>(
            a, 768, false, b, 768, false, c, 768, 192, 768, 768,
            mc: 96, nc: 768, kc: 256, mr: 16, nr: 16,
            new BlasOptions<float> { ParallelismAxis = ParallelismAxis.N }));
    }

    // Positive control: with the panel kernel's own tile the N-axis plan runs and is correct, so the refusal above is
    // the tile gate, not the path being unavailable.
    [SkippableFact]
    public void NAxisPlan_RunsAndMatchesReference_ForThePanelTile()
    {
        Skip.IfNot(MachineKernelGemm.IsFp32PanelAvailable, "Needs the FP32 machine-code panel kernel (x86 AVX2+FMA).");
        const int m = 192, n = 768, k = 768;
        var (a, b, expected) = Problem(m, n, k);
        var c = new float[m * n];
        PackBothStrategy.Run<float>(
            a, k, false, b, n, false, c, n, m, n, k,
            mc: 96, nc: 768, kc: 256, mr: MachineKernelGemm.Fp32Mr, nr: MachineKernelGemm.Fp32Nr,
            new BlasOptions<float> { ParallelismAxis = ParallelismAxis.N });
        double scale = 0, drift = 0;
        for (int i = 0; i < c.Length; i++) { scale = Math.Max(scale, Math.Abs(expected[i])); drift = Math.Max(drift, Math.Abs(c[i] - expected[i])); }
        Assert.True(drift < 1e-4 * Math.Max(1, scale) * Math.Sqrt(k), $"N-axis drift {drift}");
    }
}
#endif