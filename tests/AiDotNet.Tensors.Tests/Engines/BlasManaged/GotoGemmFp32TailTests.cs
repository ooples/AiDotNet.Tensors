// Copyright (c) AiDotNet. All rights reserved.
// #653: the GotoBLAS-style FP32 GEMM runs a C block's M-tail rows (effMc % 6) on the microkernel through a zero-padded
// A sub-panel. Validates RunSingle and RunTile (via RunParallel) against a double reference on every M-tail length,
// an N-tail, K crossing the 256 K-block, and blocks that leave a tail in more than one M-block.

#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class GotoGemmFp32TailTests
{
    public static TheoryData<int, int, int, int, int> Shapes => new()
    {
        // m, n, k, mc, nc
        { 7, 32, 9, 120, 128 },     // one full 6-row tile + 1 tail row
        { 8, 48, 300, 120, 128 },   // 2 tail rows, K over two K-blocks
        { 9, 16, 1, 120, 128 },     // 3 tail rows, K = 1
        { 10, 64, 257, 120, 128 },  // 4 tail rows, K-block remainder of 1
        { 11, 50, 64, 120, 32 },    // 5 tail rows, N-tail of 2 in the second N-block
        { 5, 32, 40, 120, 128 },    // only tail rows (no full tile)
        { 256, 96, 520, 128, 32 },  // the DiT M=256 split into two 128-row blocks: 2 tail rows in each
        { 77, 80, 70, 77, 48 },     // the cross-attention M=77: 5 tail rows
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void RunSingle_MatchesDoubleReference(int m, int n, int k, int mc, int nc)
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The machine-code microkernel is unavailable on this CPU/OS.");
        var (a, b, expected) = Problem(m, n, k);
        var c = Garbage(m * n);
        unsafe
        {
            fixed (float* pa = a, pb = b, pc = c)
                GotoGemmFp32.RunSingle(pa, k, pb, n, pc, n, m, n, k, mc, nc, 256);
        }
        AssertClose(expected, c, k, m, n);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void RunParallel_MatchesDoubleReference(int m, int n, int k, int mc, int nc)
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The machine-code microkernel is unavailable on this CPU/OS.");
        var (a, b, expected) = Problem(m, n, k);
        var c = Garbage(m * n);
        unsafe
        {
            fixed (float* pa = a, pb = b, pc = c)
                GotoGemmFp32.RunParallel(pa, k, pb, n, pc, n, m, n, k, mc, nc, 256);
        }
        AssertClose(expected, c, k, m, n);
    }

    // The CCX pool's pre-packed-B routes run the M-tail on the same microkernel as the live pack, so a weight packed
    // once gives exactly the output of the same GEMM packed per call.
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void PrePackedBRoutes_MatchTheLivePackExactly(int m, int n, int k, int mc, int nc)
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The machine-code microkernel is unavailable on this CPU/OS.");
        _ = mc; _ = nc; // one block covering the whole problem, so the three routes tile it identically
        const int kc = 256;
        var (a, b, _) = Problem(m, n, k);
        var live = Garbage(m * n);
        var tilePacked = Garbage(m * n);
        var panelStep = Garbage(m * n);
        var packed = new float[GotoGemmFp32.PackedBLen(n, k, kc)];
        var panel = new float[GotoGemmFp32.PackedBPanelLen(n, kc)];
        unsafe
        {
            fixed (float* pa = a, pb = b, pLive = live, pTile = tilePacked, pStep = panelStep, pPacked = packed, pPanel = panel)
            {
                GotoGemmFp32.RunSingle(pa, k, pb, n, pLive, n, m, n, k, m, n, kc);

                GotoGemmFp32.PackBPanel(pb, n, 0, n, k, kc, pPacked);
                GotoGemmFp32.RunTilePackedB(pa, k, pb, n, pTile, n, 0, 0, m, n, k, m, n, kc, pPacked);

                for (int pc = 0; pc < k; pc += kc)
                {
                    int effKc = Math.Min(kc, k - pc);
                    GotoGemmFp32.PackBPanelPc(pb, n, 0, n, pc, effKc, kc, pPanel, 0, 1);
                    GotoGemmFp32.RunMacroPanelStep(pa, k, pb, n, pStep, n, 0, 0, m, n, pc, effKc, m, kc, pPanel, pc == 0);
                }
            }
        }
        Assert.Equal(live, tilePacked);
        Assert.Equal(live, panelStep);
    }

    private static (float[] A, float[] B, double[] Expected) Problem(int m, int n, int k)
    {
        var rng = new Random(653 + 31 * m + 7 * n + k);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var expected = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int p = 0; p < k; p++) acc += (double)a[i * k + p] * b[p * n + j];
                expected[i * n + j] = acc;
            }
        return (a, b, expected);
    }

    // C starts as garbage: the GEMM writes C = A·B, so nothing it leaves unwritten (or accumulates onto) may survive.
    private static float[] Garbage(int length)
    {
        var c = new float[length];
        for (int i = 0; i < c.Length; i++) c[i] = 1e6f + i;
        return c;
    }

    private static void AssertClose(double[] expected, float[] actual, int k, int m, int n)
    {
        double tol = 1e-5 * Math.Max(1, k) + 1e-5;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"C[{i / n},{i % n}] of {m}x{n} (k={k}): expected {expected[i]}, got {actual[i]} (tol {tol})");
    }
}
#endif
