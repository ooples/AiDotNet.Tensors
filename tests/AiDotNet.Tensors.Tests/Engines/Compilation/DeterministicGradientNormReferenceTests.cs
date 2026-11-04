using System;
using System.Collections.Generic;
using System.Numerics;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;
using N = AiDotNet.Tensors.Engines.Gpu.DeterministicGradientNorm;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The integer reference every backend's global-norm kernels must match bit for bit, checked against exact
/// big-integer arithmetic and against IEEE fp32 PyTorch arithmetic where that does not overflow.
/// </summary>
public class DeterministicGradientNormReferenceTests
{
    // Exact sum of squares as a big integer times 2^-298 (every float square is a multiple of 2^-298).
    private static BigInteger ExactSumSquares(IEnumerable<float[]> tensors)
    {
        BigInteger s = BigInteger.Zero;
        foreach (var t in tensors)
            foreach (var x in t)
            {
                uint b = N.SingleToUInt32Bits(x);
                uint e = (b >> 23) & 0xFF;
                BigInteger m = (b & 0x7FFFFF) | (e != 0 ? 0x800000u : 0u);
                if (e == 0) e = 1;
                s += m * m << (2 * (int)e - 2);   // m^2 * 2^(2e - 300) / 2^-298
            }
        return s;
    }

    private static BigInteger RecordA(uint[] r) =>
        new BigInteger(r[N.WordA0]) | new BigInteger(r[N.WordA0 + 1]) << 32
        | new BigInteger(r[N.WordA0 + 2]) << 64 | new BigInteger(r[N.WordA0 + 3]) << 96;

    private static double RecordDouble(uint[] r) =>
        BitConverter.Int64BitsToDouble((long)(((ulong)r[N.WordSumSquaresHi] << 32) | r[N.WordSumSquaresLo]));

    private static float F(uint bits) => N.UInt32BitsToSingle(bits);

    private static float[] Random(int n, double scale, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale * Math.Pow(2, rng.Next(-20, 20)));
        return a;
    }

    public static IEnumerable<object[]> Cases()
    {
        yield return new object[] { new[] { new[] { 3f, 4f } }, 1f };
        yield return new object[] { new[] { new[] { 3e10f, 4e10f } }, 1f };
        yield return new object[] { new[] { new[] { 3e20f, 4e20f } }, 1f };
        yield return new object[] { new[] { new[] { 3e37f, -3e38f, 1e-45f } }, 2.5f };
        yield return new object[] { new[] { new[] { 1e-45f, 1e-40f, -3e-39f } }, 1e-30f };
        yield return new object[] { new[] { Random(10000, 1, 1), Random(4097, 1e3, 2), new float[0], Random(3, 1e-3, 3) }, 1f };
        yield return new object[] { new[] { Random(9000, 1e15, 4), Random(5000, 1e-15, 5) }, 0.5f };
        yield return new object[] { new[] { new float[100] }, 1f };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void SumOfSquares_IsTheExactSumTruncatedOnTheDocumentedGrid(float[][] tensors, float maxNorm)
    {
        var r = N.Compute(tensors, maxNorm);
        BigInteger exact = ExactSumSquares(tensors);
        BigInteger a = RecordA(r);
        int v = (int)r[N.WordV];
        // A * 2^V <= exact, and exact - A * 2^V < partials * 2^V (each element and partial truncates < 1 unit).
        int partials = N.Partials(tensors).Length;
        if (exact.IsZero) { Assert.True(a.IsZero); return; }
        // Compare on a common grid: 2^b with b = min(V, -298).
        int b = Math.Min(v, -298);
        BigInteger approx = a << (v - b), ex = exact << (-298 - b), unit = BigInteger.One << (v - b);
        Assert.True(approx <= ex);
        // Each element and each partial truncates by less than one 2^V unit.
        Assert.True(ex - approx < unit * (Elements(tensors) + partials), "truncation exceeded the per-element bound");
        Assert.Equal(0u, r[N.WordNonFinite]);
    }

    private static int Elements(float[][] t) { int n = 0; foreach (var x in t) n += x.Length; return n; }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DoubleNormAndScale_AreCorrectlyRoundedFromTheSum(float[][] tensors, float maxNorm)
    {
        var r = N.Compute(tensors, maxNorm);
        BigInteger a = RecordA(r);
        int v = (int)r[N.WordV];
        double d = RecordDouble(r);
        // The double is the correctly rounded value of A * 2^V: compare exactly against both neighbours.
        Assert.Equal(RoundToDouble(a, v), d);

        // norm: fp32(sqrt(A * 2^V)), correctly rounded.
        float norm = F(r[N.WordNorm]);
        Assert.Equal(RoundSqrtToFloat(a, v), norm);

        if (!float.IsInfinity(norm) && norm < 1e30f && norm > 1e-30f)
        {
            // In range, the coefficient is exactly PyTorch's fp32 arithmetic on that norm.
            float coef = maxNorm / (norm + 1e-6f);
            float expected = coef < 1f ? coef : 1f;
            Assert.Equal(expected, F(r[N.WordScale]));
        }
    }

    [Fact]
    public void ANormPastFloatRange_StillClipsExactly()
    {
        // ||(2.4e38, 3.2e38)|| = 4e38 > float.MaxValue: fp32 PyTorch arithmetic gives norm inf and scale 0.
        var g = new[] { 2.4e38f, 3.2e38f };
        var r = N.Compute(new[] { g }, 1f);
        Assert.Equal(float.PositiveInfinity, F(r[N.WordNorm]));
        float scale = F(r[N.WordScale]);
        Assert.True(scale > 0f);
        Assert.Equal(0.6f, g[0] * scale, 6);
        Assert.Equal(0.8f, g[1] * scale, 6);
        Assert.Equal(0u, r[N.WordNonFinite]);
    }

    [Theory]
    [InlineData(1f), InlineData(1e10f), InlineData(1e20f), InlineData(1e30f)]
    public void ClipIsExactAtEveryMagnitude(float magnitude)
    {
        var g = new[] { 3f * magnitude, -4f * magnitude };
        var r = N.Compute(new[] { g }, 1f);
        float scale = F(r[N.WordScale]);
        Assert.Equal(0.6f, g[0] * scale, 6);
        Assert.Equal(-0.8f, g[1] * scale, 6);
    }

    [Fact]
    public void NonFinite_IsCountedAndPropagatesAsPyTorchWould()
    {
        var r = N.Compute(new[] { new[] { 1f, 2f }, new[] { 1f, float.PositiveInfinity, float.NaN, float.NaN } }, 1f);
        Assert.Equal(2u, r[N.WordNanCount]);
        Assert.Equal(1u, r[N.WordInfCount]);
        Assert.Equal(1u, r[N.WordFirstNonFiniteTensor]);
        Assert.Equal(1u, r[N.WordNonFinite]);
        Assert.True(float.IsNaN(F(r[N.WordNorm])));
        Assert.True(float.IsNaN(F(r[N.WordScale])));
        Assert.True(double.IsNaN(RecordDouble(r)));

        var inf = N.Compute(new[] { new[] { 1f, float.NegativeInfinity } }, 1f);
        Assert.Equal(float.PositiveInfinity, F(inf[N.WordNorm]));
        Assert.Equal(0f, F(inf[N.WordScale]));
        Assert.Equal(uint.MaxValue, N.Compute(new[] { new[] { 1f } }, 1f)[N.WordFirstNonFiniteTensor]);
    }

    [Fact]
    public void Result_DoesNotDependOnElementOrderWithinAChunk()
    {
        var a = Random(4096, 1e5, 9);
        var b = (float[])a.Clone();
        Array.Reverse(b);
        Assert.Equal(N.Compute(new[] { a }, 1f), N.Compute(new[] { b }, 1f));
    }

    [Fact]
    public void ScaleRounding_MatchesExactRationalRoundingOverManyNorms()
    {
        var rng = new Random(17);
        for (int i = 0; i < 20000; i++)
        {
            double mag = Math.Pow(2, rng.Next(-140, 127));   // finite inputs
            var g = new[] { (float)(rng.NextDouble() * mag), (float)(rng.NextDouble() * mag) };
            float maxNorm = (float)Math.Pow(2, rng.Next(-30, 30)) * (float)(0.5 + rng.NextDouble());
            var r = N.Compute(new[] { g }, maxNorm);
            BigInteger a = RecordA(r);
            int v = (int)r[N.WordV];
            float norm = F(r[N.WordNorm]);
            Assert.Equal(RoundSqrtToFloat(a, v), norm);
            Assert.Equal(ReferenceScale(a, v, maxNorm), F(r[N.WordScale]));
        }
    }

    // ---- big-integer oracles ----

    private static int Bits(BigInteger x)
    {
        int n = 0;
        while (!x.IsZero) { x >>= 1; n++; }
        return n;
    }

    // Rational x = num / den (positive) rounded to binary with `precision` bits, min exponent `minExp` (ulp floor),
    // returned as (q, ulpExp) with value q * 2^ulpExp.
    private static (BigInteger q, int ulp) RoundRational(BigInteger num, BigInteger den, int precision, int minUlp)
    {
        // lead exponent: floor(log2(num/den))
        int lead = Bits(num) - Bits(den);
        if ((lead >= 0 ? num >= den << lead : num << -lead >= den) == false) lead--;
        int ulp = Math.Max(lead - (precision - 1), minUlp);
        BigInteger n = ulp >= 0 ? num : num << -ulp;
        BigInteger d = ulp >= 0 ? den << ulp : den;
        BigInteger q = BigInteger.DivRem(n, d, out var rem);
        int c = (rem * 2).CompareTo(d);
        if (c > 0 || (c == 0 && !q.IsEven)) q++;
        if (Bits(q) > precision) { q >>= 1; ulp++; }
        return (q, ulp);
    }

    private static float ToFloat(BigInteger q, int ulp)
    {
        if (q.IsZero) return 0f;
        double v = (double)q * Math.Pow(2, ulp);
        return v > float.MaxValue ? float.PositiveInfinity : (float)v;   // exact: q has <= 24 bits
    }

    private static double RoundToDouble(BigInteger a, int v)
    {
        if (a.IsZero) return 0;
        var (q, ulp) = v >= 0 ? RoundRational(a << v, 1, 53, -1074) : RoundRational(a, BigInteger.One << -v, 53, -1074);
        return (double)q * Math.Pow(2, ulp);
    }

    // fp32(sqrt(a * 2^v)): with v even, sqrt = sqrt(a) * 2^(v/2); round sqrt(a) via an integer sqrt with 60 extra bits.
    private static float RoundSqrtToFloat(BigInteger a, int v)
    {
        if (a.IsZero) return 0f;
        if ((v & 1) != 0) { a <<= 1; v--; }
        const int extra = 120;   // sqrt(a * 2^(2*extra)) = sqrt(a) * 2^extra
        BigInteger s = ISqrt(a << (2 * extra));
        bool inexact = s * s != a << (2 * extra);
        // value = (s + inexact*eps) * 2^(v/2 - extra); round with sticky: represent as (2s+inexact)/2.
        var (q, ulp) = RoundRational(2 * s + (inexact ? 1 : 0), 2, 24, -149 - (v / 2 - extra));
        return ToFloat(q, ulp + v / 2 - extra);
    }

    private static BigInteger ISqrt(BigInteger n)
    {
        if (n.IsZero) return 0;
        BigInteger x = BigInteger.One << ((Bits(n) + 1) / 2);
        while (true)
        {
            BigInteger y = (x + n / x) >> 1;
            if (y >= x) return x;
            x = y;
        }
    }

    // min(1, fp32(maxNorm / fp32(n + 1e-6f))) with n = fp32(sqrt(S)) at an unbounded exponent.
    private static float ReferenceScale(BigInteger a, int v, float maxNorm)
    {
        if ((v & 1) != 0) { a <<= 1; v--; }
        const int extra = 120;
        BigInteger s = ISqrt(a << (2 * extra));
        bool inexact = s * s != a << (2 * extra);
        var (nq, nu) = RoundRational(2 * s + (inexact ? 1 : 0), 2, 24, int.MinValue / 2);
        nu += v / 2 - extra;
        // d = fp32(n + 1e-6f) unbounded: n = nq * 2^nu, eps = 0x8637BD * 2^-43
        int lo = Math.Min(nu, -43);
        BigInteger sum = (nq << (nu - lo)) + (new BigInteger(0x8637BD) << (-43 - lo));
        var (dq, du) = RoundRational(sum, 1, 24, int.MinValue / 2);
        du += lo;
        // maxNorm = mq * 2^mu
        uint b = N.SingleToUInt32Bits(maxNorm);
        uint e = (b >> 23) & 0xFF;
        BigInteger mq = (b & 0x7FFFFF) | (e != 0 ? 0x800000u : 0u);
        int mu = (e == 0 ? 1 : (int)e) - 150;
        // coef = mq * 2^mu / (dq * 2^du)
        BigInteger num = mq, den = dq;
        int sh = mu - du;
        if (sh >= 0) num <<= sh; else den <<= -sh;
        if (num >= den) return 1f;
        var (q, u) = RoundRational(num, den, 24, -149);
        return ToFloat(q, u);
    }
}
