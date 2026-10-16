using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Float Adam and double Adam/AdamW split a large parameter into SIMD-aligned chunks across the pool, as float AdamW
/// already did. Each chunk is a disjoint element range, so the result must equal the single-threaded pass bit for bit.
/// The reference is the kernel's own range body run once over the whole length, the pass the dispatcher used to make.
/// </summary>
public class FusedAdamParallelChunkParityTests
{
    public static TheoryData<int> ParallelLengths => new() { 65_539, 100_003, 401_413 };

    [Theory]
    [MemberData(nameof(ParallelLengths))]
    public unsafe void Adam_Float_ParallelChunks_BitIdenticalToSerial(int len)
    {
        Assert.True(len >= 2 * FusedOptimizer.ParallelThreshold,
            $"length {len} no longer splits into parallel chunks at threshold {FusedOptimizer.ParallelThreshold}");
        const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        var (p0, g, m0, v0) = MakeFloat(len, 17 + len);
        for (int step = 1; step <= 3; step++)
        {
            float bc1 = 1f - (float)Math.Pow(b1, step), bc2 = 1f - (float)Math.Pow(b2, step);
            float[] pS = (float[])p0.Clone(), mS = (float[])m0.Clone(), vS = (float[])v0.Clone();
            float[] pP = (float[])p0.Clone(), mP = (float[])m0.Clone(), vP = (float[])v0.Clone();
            fixed (float* p = pS, gg = g, mm = mS, vv = vS) FusedOptimizer.AdamUpdateRangeF(p, gg, mm, vv, 0, len, lr, b1, b2, eps, bc1, bc2);
            fixed (float* p = pP, gg = g, mm = mP, vv = vP) FusedOptimizer.AdamUpdateSimd(p, gg, mm, vv, len, lr, b1, b2, eps, bc1, bc2);
            AssertBitEqual(pS, pP); AssertBitEqual(mS, mP); AssertBitEqual(vS, vP);
            Assert.NotEqual(p0[len - 1], pP[len - 1]); // the scalar tail of the last chunk was updated
        }
    }

    [Theory]
    [MemberData(nameof(ParallelLengths))]
    public unsafe void AdamAndAdamW_Double_ParallelChunks_BitIdenticalToSerial(int len)
    {
        Assert.True(len >= 2 * FusedOptimizer.ParallelThreshold,
            $"length {len} no longer splits into parallel chunks at threshold {FusedOptimizer.ParallelThreshold}");
        const double lr = 5e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-2;
        var (p0, g, m0, v0) = MakeDouble(len, 29 + len);
        for (int step = 1; step <= 3; step++)
        {
            double bc1 = 1.0 - Math.Pow(b1, step), bc2 = 1.0 - Math.Pow(b2, step);
            foreach (bool decoupled in new[] { false, true })
            {
                double[] pS = (double[])p0.Clone(), mS = (double[])m0.Clone(), vS = (double[])v0.Clone();
                double[] pP = (double[])p0.Clone(), mP = (double[])m0.Clone(), vP = (double[])v0.Clone();
                fixed (double* p = pS, gg = g, mm = mS, vv = vS)
                {
                    if (decoupled) FusedOptimizer.AdamWUpdateRangeD(p, gg, mm, vv, 0, len, lr, b1, b2, eps, wd, bc1, bc2);
                    else FusedOptimizer.AdamUpdateRangeD(p, gg, mm, vv, 0, len, lr, b1, b2, eps, bc1, bc2);
                }
                fixed (double* p = pP, gg = g, mm = mP, vv = vP)
                {
                    if (decoupled) FusedOptimizer.AdamWUpdateSimd(p, gg, mm, vv, len, lr, b1, b2, eps, wd, bc1, bc2);
                    else FusedOptimizer.AdamUpdateSimd(p, gg, mm, vv, len, lr, b1, b2, eps, bc1, bc2);
                }
                AssertBitEqual(pS, pP); AssertBitEqual(mS, mP); AssertBitEqual(vS, vP);
                Assert.NotEqual(p0[len - 1], pP[len - 1]);
            }
        }
    }

    private static void AssertBitEqual(float[] a, float[] b)
    {
        for (int i = 0; i < a.Length; i++)
            Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(a[i]), 0) == BitConverter.ToInt32(BitConverter.GetBytes(b[i]), 0),
                $"element {i}: {a[i]:R} vs {b[i]:R}");
    }

    private static void AssertBitEqual(double[] a, double[] b)
    {
        for (int i = 0; i < a.Length; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(a[i]) == BitConverter.DoubleToInt64Bits(b[i]),
                $"element {i}: {a[i]:R} vs {b[i]:R}");
    }

    private static (double[] p, double[] g, double[] m, double[] v) MakeDouble(int len, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        double[] p = new double[len], g = new double[len], m = new double[len], v = new double[len];
        for (int i = 0; i < len; i++)
        {
            p[i] = rng.NextDouble() * 2 - 1;
            g[i] = (rng.NextDouble() * 2 - 1) * 0.1;
            m[i] = (rng.NextDouble() * 2 - 1) * 0.05;
            v[i] = rng.NextDouble() * 0.01;
        }
        return (p, g, m, v);
    }

    private static (float[] p, float[] g, float[] m, float[] v) MakeFloat(int len, int seed)
    {
        var (p, g, m, v) = MakeDouble(len, seed);
        return (Array.ConvertAll(p, x => (float)x), Array.ConvertAll(g, x => (float)x),
            Array.ConvertAll(m, x => (float)x), Array.ConvertAll(v, x => (float)x));
    }
}