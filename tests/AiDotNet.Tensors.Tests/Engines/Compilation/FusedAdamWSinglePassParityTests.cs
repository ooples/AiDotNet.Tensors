using System;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The fused single-pass AdamW SIMD kernel must be BIT-IDENTICAL to the prior
/// two-pass form (a full weight-decay pre-scale pass over <c>param</c>, then
/// <see cref="FusedOptimizer"/>.AdamUpdateSimd). The fusion only removes the
/// redundant second read+write of the parameter array; it changes no arithmetic,
/// so training trajectories must be unchanged to the last bit. These tests
/// reconstruct the exact two-pass reference and compare raw bit patterns across
/// SIMD-tail lengths and a range of steps / learning rates / weight decays.
/// </summary>
public class FusedAdamWSinglePassParityTests
{
    public static TheoryData<int> Lengths => new() { 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 100, 257, 1024, 4099 };

    // The step-taking Adam/AdamW kernels now delegate to a bc1/bc2-taking overload
    // so a per-parameter loop can hoist the two step-global Math.Pow to once/step.
    // These pin the wrapper's bias-correction formula: the step overload must be
    // bit-identical to the bc overload fed bc1=1-β1^step, bc2=1-β2^step.
    [Theory]
    [MemberData(nameof(Lengths))]
    public unsafe void AdamAndAdamW_BcOverload_BitIdenticalToStepOverload_Double(int len)
    {
        const double lr = 5e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-2;
        for (int step = 1; step <= 4; step++)
        {
            double bc1 = 1.0 - Math.Pow(b1, step), bc2 = 1.0 - Math.Pow(b2, step);
            var (param, grad, m, v) = MakeDouble(len, 55 + len + step);

            // Adam: step overload vs bc overload.
            double[] pA = (double[])param.Clone(), mA = (double[])m.Clone(), vA = (double[])v.Clone();
            double[] pB = (double[])param.Clone(), mB = (double[])m.Clone(), vB = (double[])v.Clone();
            fixed (double* p = pA, g = grad, mm = mA, vv = vA) FusedOptimizer.AdamUpdateSimd(p, g, mm, vv, len, lr, b1, b2, eps, step);
            fixed (double* p = pB, g = grad, mm = mB, vv = vB) FusedOptimizer.AdamUpdateSimd(p, g, mm, vv, len, lr, b1, b2, eps, bc1, bc2);
            AssertBitEqual(pA, pB); AssertBitEqual(mA, mB); AssertBitEqual(vA, vB);

            // AdamW: step overload vs bc overload.
            double[] pC = (double[])param.Clone(), mC = (double[])m.Clone(), vC = (double[])v.Clone();
            double[] pD = (double[])param.Clone(), mD = (double[])m.Clone(), vD = (double[])v.Clone();
            fixed (double* p = pC, g = grad, mm = mC, vv = vC) FusedOptimizer.AdamWUpdateSimd(p, g, mm, vv, len, lr, b1, b2, eps, wd, step);
            fixed (double* p = pD, g = grad, mm = mD, vv = vD) FusedOptimizer.AdamWUpdateSimd(p, g, mm, vv, len, lr, b1, b2, eps, wd, bc1, bc2);
            AssertBitEqual(pC, pD); AssertBitEqual(mC, mD); AssertBitEqual(vC, vD);
        }
    }

    private static void AssertBitEqual(double[] a, double[] b)
    {
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(a[i]), BitConverter.DoubleToInt64Bits(b[i]));
    }

    // The multi-tensor (foreach) AdamW kernel must be BIT-IDENTICAL to calling the
    // single-tensor bc1/bc2 overload once per parameter — it only hoists the SIMD
    // constant setup out of the per-tensor loop, changing no arithmetic. Exercises a
    // list mixing SIMD-tail lengths and empty (len==0, unexercised) tensors.
    [Fact]
    public unsafe void AdamWMultiTensor_BitIdenticalToPerParam_Double()
    {
        const double lr = 5e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-2;
        int[] lens = { 1, 3, 4, 7, 8, 15, 16, 17, 31, 100, 4096, 0, 5, 8193 };
        int count = lens.Length;
        for (int step = 1; step <= 3; step++)
        {
            double bc1 = 1.0 - Math.Pow(b1, step), bc2 = 1.0 - Math.Pow(b2, step);
            var rng = new Random(31 + step);
            double[][] P = new double[count][], G = new double[count][], M = new double[count][], V = new double[count][];
            double[][] Pr = new double[count][], Mr = new double[count][], Vr = new double[count][];
            for (int t = 0; t < count; t++)
            {
                int L = lens[t];
                P[t] = new double[L]; G[t] = new double[L]; M[t] = new double[L]; V[t] = new double[L];
                for (int i = 0; i < L; i++)
                {
                    P[t][i] = rng.NextDouble() * 2 - 1;
                    G[t][i] = (rng.NextDouble() * 2 - 1) * 0.1;
                    M[t][i] = (rng.NextDouble() * 2 - 1) * 0.05;
                    V[t][i] = rng.NextDouble() * 0.01;
                }
                Pr[t] = (double[])P[t].Clone(); Mr[t] = (double[])M[t].Clone(); Vr[t] = (double[])V[t].Clone();
            }

            // Reference: per-parameter bc1/bc2 overload (grads unchanged by AdamW, so shared).
            for (int t = 0; t < count; t++)
            {
                int L = lens[t]; if (L == 0) continue;
                fixed (double* p = Pr[t], g = G[t], m = Mr[t], v = Vr[t])
                    FusedOptimizer.AdamWUpdateSimd(p, g, m, v, L, lr, b1, b2, eps, wd, bc1, bc2);
            }
            // Multi-tensor path.
            FusedOptimizer.AdamWUpdateSimdMulti(P, G, M, V, lens, count, lr, b1, b2, eps, wd, bc1, bc2);

            for (int t = 0; t < count; t++)
                for (int i = 0; i < lens[t]; i++)
                {
                    Assert.Equal(BitConverter.DoubleToInt64Bits(Pr[t][i]), BitConverter.DoubleToInt64Bits(P[t][i]));
                    Assert.Equal(BitConverter.DoubleToInt64Bits(Mr[t][i]), BitConverter.DoubleToInt64Bits(M[t][i]));
                    Assert.Equal(BitConverter.DoubleToInt64Bits(Vr[t][i]), BitConverter.DoubleToInt64Bits(V[t][i]));
                }
        }
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void FusedAdamW_Double_BitIdenticalToTwoPass(int len)
    {
        const double lr = 3e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-2;
        for (int step = 1; step <= 3; step++)
        {
            var (param, grad, m, v) = MakeDouble(len, 100 + len + step);

            // Reference: the OLD two-pass form. Weight-decay pre-scale (scalar
            // multiply is bit-identical to the old SIMD Avx.Multiply for the same
            // inputs), then the UNCHANGED AdamUpdateSimd kernel.
            var pRef = (double[])param.Clone();
            var mRef = (double[])m.Clone();
            var vRef = (double[])v.Clone();
            double wdScale = 1.0 - wd * lr;
            for (int i = 0; i < len; i++) pRef[i] *= wdScale;
            unsafe
            {
                fixed (double* pp = pRef, pg = grad, pm = mRef, pv = vRef)
                    FusedOptimizer.AdamUpdateSimd(pp, pg, pm, pv, len, lr, b1, b2, eps, step);
            }

            // Fused single-pass AdamW.
            var pFused = (double[])param.Clone();
            var mFused = (double[])m.Clone();
            var vFused = (double[])v.Clone();
            unsafe
            {
                fixed (double* pp = pFused, pg = grad, pm = mFused, pv = vFused)
                    FusedOptimizer.AdamWUpdateSimd(pp, pg, pm, pv, len, lr, b1, b2, eps, wd, step);
            }

            for (int i = 0; i < len; i++)
            {
                Assert.Equal(BitConverter.DoubleToInt64Bits(pRef[i]), BitConverter.DoubleToInt64Bits(pFused[i]));
                Assert.Equal(BitConverter.DoubleToInt64Bits(mRef[i]), BitConverter.DoubleToInt64Bits(mFused[i]));
                Assert.Equal(BitConverter.DoubleToInt64Bits(vRef[i]), BitConverter.DoubleToInt64Bits(vFused[i]));
            }
        }
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void FusedAdamW_Float_BitIdenticalToTwoPass(int len)
    {
        const float lr = 3e-4f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 1e-2f;
        for (int step = 1; step <= 3; step++)
        {
            var (param, grad, m, v) = MakeFloat(len, 200 + len + step);

            var pRef = (float[])param.Clone();
            var mRef = (float[])m.Clone();
            var vRef = (float[])v.Clone();
            float wdScale = 1f - wd * lr;
            for (int i = 0; i < len; i++) pRef[i] *= wdScale;
            unsafe
            {
                fixed (float* pp = pRef, pg = grad, pm = mRef, pv = vRef)
                    FusedOptimizer.AdamUpdateSimd(pp, pg, pm, pv, len, lr, b1, b2, eps, step);
            }

            var pFused = (float[])param.Clone();
            var mFused = (float[])m.Clone();
            var vFused = (float[])v.Clone();
            unsafe
            {
                fixed (float* pp = pFused, pg = grad, pm = mFused, pv = vFused)
                    FusedOptimizer.AdamWUpdateSimd(pp, pg, pm, pv, len, lr, b1, b2, eps, wd, step);
            }

            for (int i = 0; i < len; i++)
            {
                Assert.Equal(BitConverter.SingleToInt32Bits(pRef[i]), BitConverter.SingleToInt32Bits(pFused[i]));
                Assert.Equal(BitConverter.SingleToInt32Bits(mRef[i]), BitConverter.SingleToInt32Bits(mFused[i]));
                Assert.Equal(BitConverter.SingleToInt32Bits(vRef[i]), BitConverter.SingleToInt32Bits(vFused[i]));
            }
        }
    }

    // Adam (no weight decay) multi-tensor kernel — bit-identical to per-parameter.
    [Fact]
    public unsafe void AdamMultiTensor_BitIdenticalToPerParam_Double()
    {
        const double lr = 5e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8;
        int[] lens = { 1, 3, 4, 7, 8, 15, 16, 31, 100, 4096, 0, 5, 8193 };
        int count = lens.Length;
        for (int step = 1; step <= 3; step++)
        {
            double bc1 = 1.0 - Math.Pow(b1, step), bc2 = 1.0 - Math.Pow(b2, step);
            var rng = new Random(71 + step);
            double[][] P = new double[count][], G = new double[count][], M = new double[count][], V = new double[count][];
            double[][] Pr = new double[count][], Mr = new double[count][], Vr = new double[count][];
            for (int t = 0; t < count; t++)
            {
                int L = lens[t];
                P[t] = new double[L]; G[t] = new double[L]; M[t] = new double[L]; V[t] = new double[L];
                for (int i = 0; i < L; i++) { P[t][i] = rng.NextDouble() * 2 - 1; G[t][i] = (rng.NextDouble() * 2 - 1) * 0.1; M[t][i] = (rng.NextDouble() * 2 - 1) * 0.05; V[t][i] = rng.NextDouble() * 0.01; }
                Pr[t] = (double[])P[t].Clone(); Mr[t] = (double[])M[t].Clone(); Vr[t] = (double[])V[t].Clone();
            }
            for (int t = 0; t < count; t++) { int L = lens[t]; if (L == 0) continue; fixed (double* p = Pr[t], g = G[t], m = Mr[t], v = Vr[t]) FusedOptimizer.AdamUpdateSimd(p, g, m, v, L, lr, b1, b2, eps, bc1, bc2); }
            FusedOptimizer.AdamUpdateSimdMulti(P, G, M, V, lens, count, lr, b1, b2, eps, bc1, bc2);
            for (int t = 0; t < count; t++) for (int i = 0; i < lens[t]; i++)
            {
                Assert.Equal(BitConverter.DoubleToInt64Bits(Pr[t][i]), BitConverter.DoubleToInt64Bits(P[t][i]));
                Assert.Equal(BitConverter.DoubleToInt64Bits(Mr[t][i]), BitConverter.DoubleToInt64Bits(M[t][i]));
                Assert.Equal(BitConverter.DoubleToInt64Bits(Vr[t][i]), BitConverter.DoubleToInt64Bits(V[t][i]));
            }
        }
    }

    // Float Adam + AdamW multi-tensor kernels — bit-identical to per-parameter (raw bits).
    [Fact]
    public unsafe void AdamAndAdamWMultiTensor_BitIdenticalToPerParam_Float()
    {
        const float lr = 5e-4f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 1e-2f;
        int[] lens = { 1, 7, 8, 9, 15, 16, 17, 100, 4096, 0, 8193 };
        int count = lens.Length;
        for (int step = 1; step <= 3; step++)
        {
            float bc1 = 1f - MathF.Pow(b1, step), bc2 = 1f - MathF.Pow(b2, step);
            var rng = new Random(91 + step);
            float[][] P = new float[count][], G = new float[count][], M = new float[count][], V = new float[count][];
            for (int t = 0; t < count; t++)
            {
                int L = lens[t];
                P[t] = new float[L]; G[t] = new float[L]; M[t] = new float[L]; V[t] = new float[L];
                for (int i = 0; i < L; i++) { P[t][i] = (float)(rng.NextDouble() * 2 - 1); G[t][i] = (float)((rng.NextDouble() * 2 - 1) * 0.1); M[t][i] = (float)((rng.NextDouble() * 2 - 1) * 0.05); V[t][i] = (float)(rng.NextDouble() * 0.01); }
            }
            foreach (bool isAdamW in new[] { false, true })
            {
                float[][] Pr = new float[count][], Mr = new float[count][], Vr = new float[count][];
                float[][] Pm = new float[count][], Mm = new float[count][], Vm = new float[count][];
                for (int t = 0; t < count; t++) { Pr[t] = (float[])P[t].Clone(); Mr[t] = (float[])M[t].Clone(); Vr[t] = (float[])V[t].Clone(); Pm[t] = (float[])P[t].Clone(); Mm[t] = (float[])M[t].Clone(); Vm[t] = (float[])V[t].Clone(); }
                for (int t = 0; t < count; t++)
                {
                    int L = lens[t]; if (L == 0) continue;
                    fixed (float* p = Pr[t], g = G[t], m = Mr[t], v = Vr[t])
                        if (isAdamW) FusedOptimizer.AdamWUpdateSimd(p, g, m, v, L, lr, b1, b2, eps, wd, bc1, bc2);
                        else FusedOptimizer.AdamUpdateSimd(p, g, m, v, L, lr, b1, b2, eps, bc1, bc2);
                }
                if (isAdamW) FusedOptimizer.AdamWUpdateSimdMulti(Pm, G, Mm, Vm, lens, count, lr, b1, b2, eps, wd, bc1, bc2);
                else FusedOptimizer.AdamUpdateSimdMulti(Pm, G, Mm, Vm, lens, count, lr, b1, b2, eps, bc1, bc2);
                for (int t = 0; t < count; t++) for (int i = 0; i < lens[t]; i++)
                {
                    Assert.Equal(BitConverter.SingleToInt32Bits(Pr[t][i]), BitConverter.SingleToInt32Bits(Pm[t][i]));
                    Assert.Equal(BitConverter.SingleToInt32Bits(Mr[t][i]), BitConverter.SingleToInt32Bits(Mm[t][i]));
                    Assert.Equal(BitConverter.SingleToInt32Bits(Vr[t][i]), BitConverter.SingleToInt32Bits(Vm[t][i]));
                }
            }
        }
    }

    // ---- Phase 2 contiguous flat-buffer path -------------------------------------------

    // The flat-moment MULTI kernel (fallback for the some-tensors-skipped case) must be
    // BIT-IDENTICAL to the jagged multi kernel: same per-tensor loop, moments merely relocated
    // from per-tensor arrays into one flat buffer at offsets[t]. Any drift here would corrupt
    // training on steps where a lazy-init layer has no gradient yet.
    [Fact]
    public unsafe void FlatMomentMulti_BitIdenticalToJaggedMulti_Double()
    {
        const double lr = 5e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-2;
        int[] lens = { 1, 3, 4, 7, 8, 15, 16, 100, 4096, 0, 5 };
        int count = lens.Length;
        int total = 0; var offsets = new int[count];
        for (int t = 0; t < count; t++) { offsets[t] = total; total += lens[t]; }

        for (int step = 1; step <= 3; step++)
        {
            double bc1 = 1.0 - Math.Pow(b1, step), bc2 = 1.0 - Math.Pow(b2, step);
            foreach (bool isAdamW in new[] { false, true })
            {
                var rng = new Random(123 + step + (isAdamW ? 7 : 0));
                double[][] P = new double[count][], G = new double[count][], Mj = new double[count][], Vj = new double[count][];
                double[][] Pf = new double[count][];
                var mFlat = new double[total]; var vFlat = new double[total];
                for (int t = 0; t < count; t++)
                {
                    int L = lens[t];
                    P[t] = new double[L]; G[t] = new double[L]; Mj[t] = new double[L]; Vj[t] = new double[L]; Pf[t] = new double[L];
                    for (int i = 0; i < L; i++)
                    {
                        double pv = rng.NextDouble() * 2 - 1, gv = (rng.NextDouble() * 2 - 1) * 0.1;
                        double mv = (rng.NextDouble() * 2 - 1) * 0.05, vv = rng.NextDouble() * 0.01;
                        P[t][i] = pv; Pf[t][i] = pv; G[t][i] = gv;
                        Mj[t][i] = mv; Vj[t][i] = vv; mFlat[offsets[t] + i] = mv; vFlat[offsets[t] + i] = vv;
                    }
                }
                // jagged reference
                if (isAdamW) FusedOptimizer.AdamWUpdateSimdMulti(P, G, Mj, Vj, lens, count, lr, b1, b2, eps, wd, bc1, bc2);
                else FusedOptimizer.AdamUpdateSimdMulti(P, G, Mj, Vj, lens, count, lr, b1, b2, eps, bc1, bc2);
                // flat-moment
                if (isAdamW) FusedOptimizer.AdamWUpdateSimdMultiFlatMoments(Pf, G, mFlat, vFlat, lens, offsets, count, lr, b1, b2, eps, wd, bc1, bc2);
                else FusedOptimizer.AdamUpdateSimdMultiFlatMoments(Pf, G, mFlat, vFlat, lens, offsets, count, lr, b1, b2, eps, bc1, bc2);

                for (int t = 0; t < count; t++) for (int i = 0; i < lens[t]; i++)
                {
                    Assert.Equal(BitConverter.DoubleToInt64Bits(P[t][i]), BitConverter.DoubleToInt64Bits(Pf[t][i]));
                    Assert.Equal(BitConverter.DoubleToInt64Bits(Mj[t][i]), BitConverter.DoubleToInt64Bits(mFlat[offsets[t] + i]));
                    Assert.Equal(BitConverter.DoubleToInt64Bits(Vj[t][i]), BitConverter.DoubleToInt64Bits(vFlat[offsets[t] + i]));
                }
            }
        }
    }

    // The contiguous single-pass kernel matches the per-tensor kernels to within a tiny
    // tolerance — it is NOT bit-identical by design: tensor-tail elements get an FMA (one
    // rounding) instead of separate multiply+add (two roundings). Bound the drift tightly so a
    // real bug (wrong offset, skipped element, stale staging) can't hide behind the tolerance.
    [Fact]
    public unsafe void Contiguous_MatchesPerTensor_WithinFmaTail_Double()
    {
        const double lr = 5e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-2;
        int[] lens = { 7, 16, 3, 100, 33, 8, 1, 4095 };   // deliberate non-multiple-of-4 tails
        int count = lens.Length;
        int total = 0; var offsets = new int[count];
        for (int t = 0; t < count; t++) { offsets[t] = total; total += lens[t]; }

        for (int step = 1; step <= 5; step++)   // let any drift compound across steps
        {
            double bc1 = 1.0 - Math.Pow(b1, step), bc2 = 1.0 - Math.Pow(b2, step);
            foreach (bool isAdamW in new[] { false, true })
            {
                var rng = new Random(321 + step + (isAdamW ? 11 : 0));
                double[][] Pref = new double[count][], Pcon = new double[count][], G = new double[count][];
                var mRef = new double[total]; var vRef = new double[total];
                var mCon = new double[total]; var vCon = new double[total];
                var pFlat = new double[total]; var gFlat = new double[total];
                for (int t = 0; t < count; t++)
                {
                    int L = lens[t];
                    Pref[t] = new double[L]; Pcon[t] = new double[L]; G[t] = new double[L];
                    for (int i = 0; i < L; i++)
                    {
                        double pv = rng.NextDouble() * 2 - 1, gv = (rng.NextDouble() * 2 - 1) * 0.1;
                        double mv = (rng.NextDouble() * 2 - 1) * 0.05, vv = rng.NextDouble() * 0.01;
                        Pref[t][i] = pv; Pcon[t][i] = pv; G[t][i] = gv;
                        mRef[offsets[t] + i] = mv; vRef[offsets[t] + i] = vv;
                        mCon[offsets[t] + i] = mv; vCon[offsets[t] + i] = vv;
                    }
                }
                // reference: flat-moment multi (per-tensor tails = scalar)
                if (isAdamW) FusedOptimizer.AdamWUpdateSimdMultiFlatMoments(Pref, G, mRef, vRef, lens, offsets, count, lr, b1, b2, eps, wd, bc1, bc2);
                else FusedOptimizer.AdamUpdateSimdMultiFlatMoments(Pref, G, mRef, vRef, lens, offsets, count, lr, b1, b2, eps, bc1, bc2);
                // contiguous single pass (uniform FMA)
                if (isAdamW) FusedOptimizer.AdamWUpdateSimdContiguous(Pcon, G, lens, offsets, count, pFlat, gFlat, mCon, vCon, total, lr, b1, b2, eps, wd, bc1, bc2);
                else FusedOptimizer.AdamUpdateSimdContiguous(Pcon, G, lens, offsets, count, pFlat, gFlat, mCon, vCon, total, lr, b1, b2, eps, bc1, bc2);

                for (int t = 0; t < count; t++) for (int i = 0; i < lens[t]; i++)
                {
                    Assert.Equal(Pref[t][i], Pcon[t][i], 12);
                    Assert.Equal(mRef[offsets[t] + i], mCon[offsets[t] + i], 12);
                    Assert.Equal(vRef[offsets[t] + i], vCon[offsets[t] + i], 12);
                }
            }
        }
    }

    // A tensor with an empty gradient (inactive / lazy-init layer) must have its flat moment
    // slot left completely untouched by the flat-moment multi kernel — the guard the contiguous
    // path relies on to fall back safely.
    [Fact]
    public unsafe void FlatMomentMulti_SkipsInactiveTensor_LeavesItsMomentsUntouched_Double()
    {
        const double lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-2;
        int[] lens = { 8, 8, 8 };
        int count = 3, total = 24;
        var offsets = new[] { 0, 8, 16 };
        var P = new double[count][]; var G = new double[count][];
        var mFlat = new double[total]; var vFlat = new double[total];
        var rng = new Random(99);
        for (int t = 0; t < count; t++)
        {
            P[t] = new double[8];
            // Middle tensor is INACTIVE: empty gradient array.
            G[t] = t == 1 ? Array.Empty<double>() : new double[8];
            for (int i = 0; i < 8; i++)
            {
                P[t][i] = rng.NextDouble();
                if (G[t].Length == 8) G[t][i] = rng.NextDouble() * 0.1;
                mFlat[offsets[t] + i] = 0.5; vFlat[offsets[t] + i] = 0.25;
            }
        }
        FusedOptimizer.AdamWUpdateSimdMultiFlatMoments(P, G, mFlat, vFlat, lens, offsets, count, lr, b1, b2, eps, wd, 1.0 - b1, 1.0 - b2);
        // Inactive tensor's moments are exactly their initial values (bit-exact).
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(BitConverter.DoubleToInt64Bits(0.5), BitConverter.DoubleToInt64Bits(mFlat[8 + i]));
            Assert.Equal(BitConverter.DoubleToInt64Bits(0.25), BitConverter.DoubleToInt64Bits(vFlat[8 + i]));
        }
        // Active tensors' moments DID move.
        Assert.NotEqual(0.5, mFlat[0]);
        Assert.NotEqual(0.5, mFlat[16]);
    }

    private static (double[] p, double[] g, double[] m, double[] v) MakeDouble(int len, int seed)
    {
        var rng = new Random(seed);
        double[] p = new double[len], g = new double[len], m = new double[len], v = new double[len];
        for (int i = 0; i < len; i++)
        {
            p[i] = (rng.NextDouble() * 2 - 1);
            g[i] = (rng.NextDouble() * 2 - 1) * 0.1;
            m[i] = (rng.NextDouble() * 2 - 1) * 0.05;
            v[i] = rng.NextDouble() * 0.01;   // second moment >= 0
        }
        return (p, g, m, v);
    }

    private static (float[] p, float[] g, float[] m, float[] v) MakeFloat(int len, int seed)
    {
        var rng = new Random(seed);
        float[] p = new float[len], g = new float[len], m = new float[len], v = new float[len];
        for (int i = 0; i < len; i++)
        {
            p[i] = (float)(rng.NextDouble() * 2 - 1);
            g[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
            m[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
            v[i] = (float)(rng.NextDouble() * 0.01);
        }
        return (p, g, m, v);
    }
}
