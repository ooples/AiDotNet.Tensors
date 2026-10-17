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

    /// <summary>
    /// The multi-tensor AdamW (one dispatch over every tensor's segments) must equal the per-tensor kernel bit for bit,
    /// honour each tensor's parameter and gradient offsets, and skip a tensor with no gradient backing -- for a total both
    /// above the parallel threshold (pool) and below it (calling thread).
    /// </summary>
    [Theory]
    [InlineData(new[] { 64, 4096, 8192, 10, 65_539, 3, 8191, 128 })]   // 86K elements: one parallel dispatch
    [InlineData(new[] { 64, 1000, 17, 8 })]                              // under the threshold: serial
    public unsafe void AdamW_Float_MultiTensorParallel_BitIdenticalToPerTensorKernel(int[] lengths)
    {
        const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 1e-2f;
        int n = lengths.Length + 1;   // plus one tensor whose gradient backing is empty
        var allLengths = new int[n];
        Array.Copy(lengths, allLengths, lengths.Length);
        allLengths[n - 1] = 32;
        var pRef = new float[n][]; var mRef = new float[n][]; var vRef = new float[n][];
        var pMul = new float[n][]; var mMul = new float[n][]; var vMul = new float[n][];
        var grads = new float[n][];
        var pOff = new int[n]; var gOff = new int[n];
        for (int t = 0; t < n; t++)
        {
            int len = allLengths[t];
            var (p, g, m, v) = MakeFloat(len, 300 + t);
            pOff[t] = t % 3; gOff[t] = (t + 1) % 4;   // the plan binds views into larger backings
            pRef[t] = new float[len + pOff[t]]; Array.Copy(p, 0, pRef[t], pOff[t], len);
            pMul[t] = (float[])pRef[t].Clone();
            grads[t] = t == n - 1 ? Array.Empty<float>() : new float[len + gOff[t]];
            if (t != n - 1) Array.Copy(g, 0, grads[t], gOff[t], len);
            mRef[t] = m; mMul[t] = (float[])m.Clone();
            vRef[t] = v; vMul[t] = (float[])v.Clone();
        }
        var untouched = (float[])pRef[n - 1].Clone();
        int[]? units = null;
        for (int step = 1; step <= 3; step++)
        {
            float bc1 = 1f - (float)Math.Pow(b1, step), bc2 = 1f - (float)Math.Pow(b2, step);
            for (int t = 0; t < n - 1; t++)
                fixed (float* p = &pRef[t][pOff[t]], g = &grads[t][gOff[t]], m = mRef[t], v = vRef[t])
                    FusedOptimizer.AdamWUpdateSimd(p, g, m, v, allLengths[t], lr, b1, b2, eps, wd, bc1, bc2);
            FusedOptimizer.AdamWUpdateMultiParallel(pMul, pOff, grads, gOff, mMul, vMul, allLengths, n,
                lr, b1, b2, eps, wd, bc1, bc2, ref units);
            for (int t = 0; t < n; t++)
            {
                AssertBitEqual(pRef[t], pMul[t]); AssertBitEqual(mRef[t], mMul[t]); AssertBitEqual(vRef[t], vMul[t]);
            }
        }
        AssertBitEqual(untouched, pMul[n - 1]);   // no gradient: not updated
        for (int t = 0; t < n - 1; t++)
            Assert.NotEqual(BitConverter.ToInt32(BitConverter.GetBytes(pRef[t][pOff[t] + allLengths[t] - 1]), 0),
                BitConverter.ToInt32(BitConverter.GetBytes(MakeFloat(allLengths[t], 300 + t).p[allLengths[t] - 1]), 0));
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