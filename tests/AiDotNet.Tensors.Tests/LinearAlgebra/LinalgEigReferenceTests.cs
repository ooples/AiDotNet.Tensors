using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// <see cref="Linalg.Eig{T}"/> / <see cref="Linalg.Eigvals{T}"/> against numpy.linalg.eigvals (LAPACK dgeev).
///
/// <para>Before this suite the float/double path ran 200 unshifted QR sweeps, accepted the unconverged
/// result, and read any non-negligible sub-diagonal as a complex pair, so it returned wrong eigenvalues
/// (spectral radius 0.668 instead of 1.053 for the 200x200 reservoir below); its eigenvectors came from
/// a dense 2n x 2n inverse-iteration solve per eigenvalue, about 98 MB for a 100x100 matrix.</para>
///
/// <para>Expected values come from this script (numpy 2.5.4); <c>reservoir</c> ports System.Random(seed)
/// so the reservoir matrices are bit-identical to <see cref="RandomReservoir"/>:</para>
/// <code>
/// import numpy as np
/// MBIG = 2147483647; MSEED = 161803398
/// class NetRandom:  # System.Random(int seed): Knuth subtractive generator
///     def __init__(self, seed):
///         sa = [0] * 56; mj = MSEED - abs(seed); sa[55] = mj; mk = 1
///         for i in range(1, 55):
///             ii = (21 * i) % 55; sa[ii] = mk; mk = mj - mk
///             if mk &lt; 0: mk += MBIG
///             mj = sa[ii]
///         for _ in range(1, 5):
///             for i in range(1, 56):
///                 sa[i] -= sa[1 + (i + 30) % 55]
///                 if sa[i] &lt; 0: sa[i] += MBIG
///         self.sa, self.inext, self.inextp = sa, 0, 21
///     def next_double(self):
///         self.inext = 1 if self.inext + 1 &gt;= 56 else self.inext + 1
///         self.inextp = 1 if self.inextp + 1 &gt;= 56 else self.inextp + 1
///         r = self.sa[self.inext] - self.sa[self.inextp]
///         if r == MBIG: r -= 1
///         if r &lt; 0: r += MBIG
///         self.sa[self.inext] = r
///         return r * (1.0 / MBIG)
/// def reservoir(size, density, seed):
///     rnd = NetRandom(seed); v = np.zeros(size * size)
///     for i in range(v.size):
///         v[i] = rnd.next_double() - 0.5 if rnd.next_double() &lt; density else 0.0
///     return v.reshape(size, size)
/// def fmt(a): return ", ".join(repr(float(x)) for x in np.ravel(a))
/// def ev(a):
///     w = sorted(np.linalg.eigvals(a), key=lambda z: (round(z.real, 6), z.imag))
///     return fmt([c for z in w for c in (z.real, z.imag)])
/// rng = np.random.default_rng(2026)
/// A = np.round(rng.standard_normal((6, 6)), 4); print(fmt(A), ev(A))
/// S = np.array([[1., 2, 0], [0, 1, 3], [1, 0, 1]]); J = np.array([[2., 1, 0], [0, 2, 1], [0, 0, 2]])
/// D = S @ J @ np.linalg.inv(S); print(fmt(D), ev(D))
/// print(ev(np.array([[1., 1], [1e-10, 1]])))
/// B = np.round(rng.standard_normal((5, 5)), 4); Sym = B + B.T; print(fmt(Sym), fmt(np.linalg.eigvalsh(Sym)))
/// G = np.diag([1e6, 1, 1e-6]) @ np.array([[1., 2, 3], [4, 5, 6], [7, 8, 10]]) @ np.diag([1e-6, 1, 1e6])
/// print(fmt(G), ev(G))
/// print(ev(reservoir(37, .3, 4)))
/// for size, dens, seed in [(100, .1, 1), (37, .3, 4), (100, 1.0, 3), (200, .05, 5)]:
///     print(np.abs(np.linalg.eigvals(reservoir(size, dens, seed))).max())
/// # Half inputs: the same matrices rounded to float16 first
/// print(ev(A.astype(np.float16).astype(np.float64)))
/// print(np.abs(np.linalg.eigvals(reservoir(37, .3, 4).astype(np.float16).astype(np.float64))).max())
/// </code>
/// </summary>
public class LinalgEigReferenceTests
{
    private static readonly double[] RandomSixExpected =
    {
        -2.167014102762344, 0.0, -0.8887516682348182, -0.34126802245752774, -0.8887516682348182, 0.34126802245752774,
        0.8493781868627006, -0.9279408968174128, 0.8493781868627006, 0.9279408968174128, 2.02856106550658, 0.0,
    };

    [Fact]
    public void RandomNonsymmetric_WithComplexPairs_MatchesNumpy()
    {
        var a = new[]
        {
            -0.7931, 0.2406, -1.8963, 1.3958, 0.6383, -0.292,
            -0.3119, 0.3038, -0.2677, -0.2259, 0.7201, 0.5147,
            -0.0641, -0.0855, 0.1609, -0.614, -0.4038, 0.5483,
            -0.1305, -1.3744, -0.4773, 0.6566, -0.2323, -0.1487,
            0.6418, 1.8246, -0.7132, 1.3482, -1.23, 0.175,
            -1.1695, 1.3515, 0.8339, 1.1377, -0.8855, 0.6846,
        };
        var expected = new[]
        {
            -2.167014102762344, 0.0, -0.8887516682348182, -0.34126802245752774, -0.8887516682348182, 0.34126802245752774,
            0.8493781868627006, -0.9279408968174128, 0.8493781868627006, 0.9279408968174128, 2.02856106550658, 0.0,
        };
        AssertEigMatches(a, 6, expected, 1e-12, 1e-12);
    }

    [Fact]
    public void Defective_JordanBlockUnderSimilarity_MatchesNumpy()
    {
        // S·J·S⁻¹ with J the 3x3 Jordan block for 2: one eigenvalue of algebraic multiplicity 3 and a single
        // eigenvector. Any backward-stable solver perturbs it by O(eps^(1/3)) ≈ 6e-6 (numpy returns
        // 1.9999979, 2.0000010 ± 1.8e-6i), but the mean (= trace / 3) stays exact.
        var a = new[]
        {
            2.1428571428571432, 0.7142857142857142, -0.1428571428571429,
            -0.14285714285714274, 2.2857142857142856, 0.14285714285714274,
            0.4285714285714285, 0.14285714285714285, 1.5714285714285716,
        };
        var expected = new[]
        {
            1.9999979191569563, 0.0, 2.0000010404215294, -1.8022682751198351e-06, 2.0000010404215294, 1.8022682751198351e-06,
        };
        var (w, _) = AssertEigMatches(a, 3, expected, 1e-4, residualTol: 1e-8);
        double sumRe = 0, sumIm = 0;
        for (int k = 0; k < 3; k++)
        {
            Assert.True(Math.Abs(w[k * 2] - 2.0) < 1e-4, $"eigenvalue {k} = {w[k * 2]} + {w[k * 2 + 1]}i, expected 2");
            sumRe += w[k * 2];
            sumIm += w[k * 2 + 1];
        }
        Assert.Equal(6.0, sumRe, 12);
        Assert.Equal(0.0, sumIm, 12);
    }

    [Fact]
    public void NearlyDefective_TinyCoupling_MatchesNumpy()
    {
        // [[1, 1], [1e-10, 1]]: eigenvalues 1 ± 1e-5, a split 1e5 times larger than the perturbation.
        var expected = new[] { 0.99999, 0.0, 1.00001, 0.0 };
        AssertEigMatches(new[] { 1.0, 1.0, 1e-10, 1.0 }, 2, expected, 1e-11, 1e-12);
    }

    [Fact]
    public void Symmetric_MatchesNumpyEigvalshAndIsReal()
    {
        var a = new[]
        {
            -1.038, -1.0854, 2.69, 1.7038, -0.086,
            -1.0854, -1.6516, 0.6284, -0.06989999999999996, 1.0035,
            2.69, 0.6284, 5.119, 4.1454, 2.9065000000000003,
            1.7038, -0.06989999999999996, 4.1454, -0.8854, -0.5773,
            -0.086, 1.0035, 2.9065000000000003, -0.5773, -1.9708,
        };
        var eigvalsh = new[] { -4.616118558135452, -3.1025887990151793, -1.9726054373168918, 0.48923039377296207, 8.775282400694557 };
        var expected = new double[10];
        for (int i = 0; i < 5; i++) expected[i * 2] = eigvalsh[i];
        var (w, _) = AssertEigMatches(a, 5, expected, 1e-12, 1e-12);
        for (int k = 0; k < 5; k++) Assert.Equal(0.0, w[k * 2 + 1]);
    }

    [Fact]
    public void BadlyScaled_BalancingRecoversEigenvalues()
    {
        // diag(1e6, 1, 1e-6)·M·diag(1e-6, 1, 1e6): entries span 18 orders of magnitude.
        var a = new[] { 1.0, 2000000.0, 3000000000000.0, 4e-06, 5.0, 6000000.0, 6.999999999999999e-12, 8e-06, 10.0 };
        var expected = new[] { -0.9057401795217568, 0.0, 0.19824686339701092, 0.0, 16.707493316124758, 0.0 };
        AssertEigMatches(a, 3, expected, 1e-12, residualTol: 1e-10);
    }

    [Fact]
    public void AiDotNetReservoir37_FullSpectrumMatchesNumpy()
    {
        var expected = new[]
        {
            -0.8796833251351825, -0.08477831978564901, -0.8796833251351825, 0.08477831978564901, -0.6790287170768822, 0.0,
            -0.5491096188225211, -0.2732715911015583, -0.5491096188225211, 0.2732715911015583, -0.4351271298732373, -0.49957519332625155,
            -0.4351271298732373, 0.49957519332625155, -0.36051097765896256, 0.0, -0.24311920249310442, -0.4302683627991814,
            -0.24311920249310442, 0.4302683627991814, -0.1887640464551466, -0.5428862749491096, -0.1887640464551466, 0.5428862749491096,
            -0.13362813051497616, -0.1369805854630719, -0.13362813051497616, 0.1369805854630719, -0.04136394309328801, -0.3677990315115032,
            -0.04136394309328801, 0.3677990315115032, 0.013093511478692005, -0.9485461540787828, 0.013093511478692005, 0.9485461540787828,
            0.07469246219483983, -0.29927783167195376, 0.07469246219483983, 0.29927783167195376, 0.12458234345532976, -0.5299014590007493,
            0.12458234345532976, 0.5299014590007493, 0.1250192170370008, 0.0, 0.22519974341248267, -0.5555351289240452,
            0.22519974341248267, 0.5555351289240452, 0.24247382390189037, -0.211181214860845, 0.24247382390189037, 0.211181214860845,
            0.36604459903051456, -0.37967259138119647, 0.36604459903051456, 0.37967259138119647, 0.3835490515614013, 0.0,
            0.5014788039026938, -0.2363332876353232, 0.5014788039026938, 0.2363332876353232, 0.6751606094089965, 0.0,
            0.8403649948221124, -0.428198935839249, 0.8403649948221124, 0.428198935839249, 0.938396792505346, -0.08005514809157088,
            0.938396792505346, 0.08005514809157088,
        };
        AssertEigMatches(RandomReservoir(37, 0.3, 4), 37, expected, 1e-12, 1e-12);
    }

    // AiDotNet PR #2641 (commit 701892c94d) repro: the old path returned 0.9640, 0.9200,
    // 2.4715 and 0.6683 for these four reservoirs.
    [Theory]
    [InlineData(100, 0.1, 1, 0.987483918802371)]
    [InlineData(37, 0.3, 4, 0.9486365196746817)]
    [InlineData(100, 1.0, 3, 2.9365774244347898)]
    [InlineData(200, 0.05, 5, 1.053266845019182)]
    public void AiDotNetReservoir_SpectralRadiusMatchesNumpy(int size, double density, int seed, double expected)
    {
        var w = Linalg.Eigvals(new Tensor<double>(RandomReservoir(size, density, seed), new[] { size, size })).GetDataArray();
        double radius = 0;
        for (int k = 0; k < size; k++) radius = Math.Max(radius, Math.Sqrt(w[k * 2] * w[k * 2] + w[k * 2 + 1] * w[k * 2 + 1]));
        Assert.Equal(expected, radius, 12);
    }

    [Fact]
    public void Float_RandomNonsymmetric_MatchesNumpy()
    {
        // The 6x6 case above through the float entry point (computed in double, rounded on output).
        var a = new float[]
        {
            -0.7931f, 0.2406f, -1.8963f, 1.3958f, 0.6383f, -0.292f,
            -0.3119f, 0.3038f, -0.2677f, -0.2259f, 0.7201f, 0.5147f,
            -0.0641f, -0.0855f, 0.1609f, -0.614f, -0.4038f, 0.5483f,
            -0.1305f, -1.3744f, -0.4773f, 0.6566f, -0.2323f, -0.1487f,
            0.6418f, 1.8246f, -0.7132f, 1.3482f, -1.23f, 0.175f,
            -1.1695f, 1.3515f, 0.8339f, 1.1377f, -0.8855f, 0.6846f,
        };
        var expected = new[]
        {
            -2.167014102762344, 0.0, -0.8887516682348182, -0.34126802245752774, -0.8887516682348182, 0.34126802245752774,
            0.8493781868627006, -0.9279408968174128, 0.8493781868627006, 0.9279408968174128, 2.02856106550658, 0.0,
        };
        var w = Linalg.Eigvals(new Tensor<float>(a, new[] { 6, 6 })).GetDataArray();
        var actual = new double[w.Length];
        for (int i = 0; i < w.Length; i++) actual[i] = w[i];
        AssertSpectraMatch(expected, actual, 1e-5);
    }

    [Fact]
    public void Batched_EqualsPerMatrix()
    {
        var a = RandomReservoir(12, 0.5, 11);
        var b = RandomReservoir(12, 0.5, 12);
        var both = new double[2 * 144];
        Array.Copy(a, 0, both, 0, 144);
        Array.Copy(b, 0, both, 144, 144);
        var (wb, vb) = Linalg.Eig(new Tensor<double>(both, new[] { 2, 12, 12 }));
        var (wa, va) = Linalg.Eig(new Tensor<double>(a, new[] { 12, 12 }));
        var (w2, v2) = Linalg.Eig(new Tensor<double>(b, new[] { 12, 12 }));
        var wbd = wb.GetDataArray();
        var vbd = vb.GetDataArray();
        for (int i = 0; i < 24; i++)
        {
            Assert.Equal(wa.GetDataArray()[i], wbd[i]);
            Assert.Equal(w2.GetDataArray()[i], wbd[24 + i]);
        }
        for (int i = 0; i < 288; i++)
        {
            Assert.Equal(va.GetDataArray()[i], vbd[i]);
            Assert.Equal(v2.GetDataArray()[i], vbd[288 + i]);
        }
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void Census100x100_AllocatesOnlyOutputsAndOneWorkspace()
    {
        // Outputs are 100x100x2 + 100x2 doubles (~162 KB) and the workspace two 100x100 doubles (~160 KB);
        // the old per-eigenvalue 200x200 inverse-iteration solves allocated ~98 MB here.
        var input = new Tensor<double>(RandomReservoir(100, 0.1, 7), new[] { 100, 100 });
        Linalg.Eig(input);

        long before = GC.GetAllocatedBytesForCurrentThread();
        Linalg.Eig(input);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(allocated < 1_000_000, $"Eig of a 100x100 matrix allocated {allocated:N0} bytes");
    }
#endif

    // ── decimal: Hessenberg + Francis QR + Schur back-substitution carried out in decimal ──

    [Fact]
    public void Decimal_RandomNonsymmetric_MatchesNumpy()
    {
        var a = new[]
        {
            -0.7931m, 0.2406m, -1.8963m, 1.3958m, 0.6383m, -0.292m,
            -0.3119m, 0.3038m, -0.2677m, -0.2259m, 0.7201m, 0.5147m,
            -0.0641m, -0.0855m, 0.1609m, -0.614m, -0.4038m, 0.5483m,
            -0.1305m, -1.3744m, -0.4773m, 0.6566m, -0.2323m, -0.1487m,
            0.6418m, 1.8246m, -0.7132m, 1.3482m, -1.23m, 0.175m,
            -1.1695m, 1.3515m, 0.8339m, 1.1377m, -0.8855m, 0.6846m,
        };
        var (wt, vt) = Linalg.Eig(new Tensor<decimal>(a, new[] { 6, 6 }));
        var w = wt.GetDataArray();
        var v = vt.GetDataArray();
        var actual = new double[12];
        for (int i = 0; i < 12; i++) actual[i] = (double)w[i];
        AssertSpectraMatch(RandomSixExpected, actual, 1e-12);

        // Decimal eigenpairs are far tighter than the double oracle: A·v = λ·v to 1e-20 with unit norm.
        for (int k = 0; k < 6; k++)
        {
            decimal lr = w[k * 2], li = w[k * 2 + 1], norm2 = 0m;
            for (int r = 0; r < 6; r++)
            {
                decimal avr = 0m, avi = 0m;
                for (int c = 0; c < 6; c++)
                {
                    avr += a[r * 6 + c] * v[(c * 6 + k) * 2];
                    avi += a[r * 6 + c] * v[(c * 6 + k) * 2 + 1];
                }
                decimal xr = v[(r * 6 + k) * 2], xi = v[(r * 6 + k) * 2 + 1];
                Assert.True(Math.Abs(avr - (lr * xr - li * xi)) < 1e-20m, $"eigenpair {k} row {r} (re)");
                Assert.True(Math.Abs(avi - (lr * xi + li * xr)) < 1e-20m, $"eigenpair {k} row {r} (im)");
                norm2 += xr * xr + xi * xi;
            }
            Assert.True(Math.Abs(norm2 - 1m) < 1e-24m, $"eigenvector {k} squared norm {norm2}");
        }
    }

    [Fact]
    public void Decimal_AiDotNetReservoir37_SpectralRadiusMatchesNumpy()
    {
        // (decimal)double keeps 15 significant digits: a ~1e-16 relative perturbation of the numpy matrix.
        var src = RandomReservoir(37, 0.3, 4);
        var a = new decimal[src.Length];
        for (int i = 0; i < a.Length; i++) a[i] = (decimal)src[i];
        var w = Linalg.Eigvals(new Tensor<decimal>(a, new[] { 37, 37 })).GetDataArray();
        double radius = 0;
        for (int k = 0; k < 37; k++) radius = Math.Max(radius, Math.Sqrt((double)(w[k * 2] * w[k * 2] + w[k * 2 + 1] * w[k * 2 + 1])));
        Assert.Equal(0.9486365196746817, radius, 10);
    }

    // ── Half: accumulated in double, rounded to Half on output ──

    [Fact]
    public void Half_RandomNonsymmetric_MatchesNumpyOnTheRoundedMatrix()
    {
        var src = new[]
        {
            -0.7931, 0.2406, -1.8963, 1.3958, 0.6383, -0.292,
            -0.3119, 0.3038, -0.2677, -0.2259, 0.7201, 0.5147,
            -0.0641, -0.0855, 0.1609, -0.614, -0.4038, 0.5483,
            -0.1305, -1.3744, -0.4773, 0.6566, -0.2323, -0.1487,
            0.6418, 1.8246, -0.7132, 1.3482, -1.23, 0.175,
            -1.1695, 1.3515, 0.8339, 1.1377, -0.8855, 0.6846,
        };
        var a = new Half[src.Length];
        for (int i = 0; i < a.Length; i++) a[i] = (Half)src[i];
        var expected = new[]
        {
            -2.167222942525127, 0.0, -0.8886911191462048, -0.34152753624919174, -0.8886911191462048, 0.34152753624919174,
            0.8491950924991476, -0.927896888689763, 0.8491950924991476, 0.927896888689763, 2.028685698944235, 0.0,
        };
        var (wt, vt) = Linalg.Eig(new Tensor<Half>(a, new[] { 6, 6 }));
        var w = wt.GetDataArray();
        var v = vt.GetDataArray();
        var actual = new double[12];
        for (int i = 0; i < 12; i++) actual[i] = (double)w[i];
        // Only the output rounding remains: half an ulp of Half at |λ| ≈ 2 is 9.8e-4.
        AssertSpectraMatch(expected, actual, 1e-3);
        for (int k = 0; k < 6; k++)
        {
            double norm2 = 0;
            for (int r = 0; r < 6; r++)
            {
                double xr = (double)v[(r * 6 + k) * 2], xi = (double)v[(r * 6 + k) * 2 + 1];
                norm2 += xr * xr + xi * xi;
            }
            Assert.True(Math.Abs(norm2 - 1) < 4e-3, $"Half eigenvector {k} squared norm {norm2}");
        }
    }

    [Fact]
    public void Half_AiDotNetReservoir37_SpectralRadiusMatchesNumpyOnTheRoundedMatrix()
    {
        var src = RandomReservoir(37, 0.3, 4);
        var a = new Half[src.Length];
        for (int i = 0; i < a.Length; i++) a[i] = (Half)src[i];
        var w = Linalg.Eigvals(new Tensor<Half>(a, new[] { 37, 37 })).GetDataArray();
        double radius = 0;
        for (int k = 0; k < 37; k++)
        {
            double re = (double)w[k * 2], im = (double)w[k * 2 + 1];
            radius = Math.Max(radius, Math.Sqrt(re * re + im * im));
        }
        Assert.True(Math.Abs(radius - 0.9486886876240023) < 1e-3, $"Half spectral radius {radius}");
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void Decimal40x40_AllocatesOnlyOutputsAndOneWorkspace()
    {
        // Outputs 40x40x2 + 40x2 decimals (~52 KB) and two 40x40 workspaces (~51 KB); the old dense 80x80
        // inverse iteration per eigenvalue allocated O(n³).
        var src = RandomReservoir(40, 0.3, 9);
        var a = new decimal[src.Length];
        for (int i = 0; i < a.Length; i++) a[i] = (decimal)src[i];
        var input = new Tensor<decimal>(a, new[] { 40, 40 });
        Linalg.Eig(input);

        long before = GC.GetAllocatedBytesForCurrentThread();
        Linalg.Eig(input);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(allocated < 300_000, $"decimal Eig of a 40x40 matrix allocated {allocated:N0} bytes");
    }

    [Fact]
    public void Half100x100_AllocatesOnlyOutputsAndOneWorkspace()
    {
        // Outputs 100x100x2 + 100x2 Halfs (~40 KB) and the double workspace (~160 KB).
        var src = RandomReservoir(100, 0.1, 7);
        var a = new Half[src.Length];
        for (int i = 0; i < a.Length; i++) a[i] = (Half)src[i];
        var input = new Tensor<Half>(a, new[] { 100, 100 });
        Linalg.Eig(input);

        long before = GC.GetAllocatedBytesForCurrentThread();
        Linalg.Eig(input);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(allocated < 600_000, $"Half Eig of a 100x100 matrix allocated {allocated:N0} bytes");
    }
#endif

    /// <summary>
    /// Checks eigenvalues against numpy (as an unordered multiset) and every eigenpair against A·v = λ·v,
    /// ‖v‖ = 1, conjugate pairs adjacent with the positive imaginary part first, and the largest component
    /// of a complex eigenvector real (the dgeev conventions).
    /// </summary>
    private static (double[] W, double[] V) AssertEigMatches(double[] a, int n, double[] expected, double valueTol, double residualTol)
    {
        var (wt, vt) = Linalg.Eig(new Tensor<double>((double[])a.Clone(), new[] { n, n }));
        var w = wt.GetDataArray();
        var v = vt.GetDataArray();
        AssertSpectraMatch(expected, w, valueTol);
        Assert.Equal(w, Linalg.Eigvals(new Tensor<double>((double[])a.Clone(), new[] { n, n })).GetDataArray());

        double aNorm = 0;
        foreach (var x in a) aNorm = Math.Max(aNorm, Math.Abs(x));
        for (int k = 0; k < n; k++)
        {
            double lr = w[k * 2], li = w[k * 2 + 1];
            if (li > 0)
            {
                Assert.True(k + 1 < n && w[(k + 1) * 2] == lr && w[(k + 1) * 2 + 1] == -li, $"eigenvalue {k} lacks its conjugate after it");
            }
            else if (li < 0)
            {
                Assert.True(k > 0 && w[(k - 1) * 2] == lr && w[(k - 1) * 2 + 1] == -li, $"eigenvalue {k} lacks its conjugate before it");
            }

            double norm2 = 0, worst = 0, best = -1, bestIm = 0;
            for (int r = 0; r < n; r++)
            {
                double avr = 0, avi = 0;
                for (int c = 0; c < n; c++)
                {
                    avr += a[r * n + c] * v[(c * n + k) * 2];
                    avi += a[r * n + c] * v[(c * n + k) * 2 + 1];
                }
                double xr = v[(r * n + k) * 2], xi = v[(r * n + k) * 2 + 1];
                worst = Math.Max(worst, Math.Abs(avr - (lr * xr - li * xi)));
                worst = Math.Max(worst, Math.Abs(avi - (lr * xi + li * xr)));
                double m2 = xr * xr + xi * xi;
                norm2 += m2;
                if (m2 > best) { best = m2; bestIm = xi; }
            }
            Assert.True(Math.Abs(norm2 - 1) < 1e-12, $"eigenvector {k} has squared norm {norm2}");
            Assert.True(worst <= residualTol * Math.Max(1.0, aNorm) * n,
                $"eigenpair {k} ({lr} + {li}i): |A·v - λ·v| = {worst}");
            if (li != 0) Assert.True(Math.Abs(bestIm) < 1e-14, $"largest component of eigenvector {k} has Im {bestIm}");
        }
        return (w, v);
    }

    /// <summary>Greedy nearest-neighbour matching of (re, im) pairs; each expected value must be hit once.</summary>
    private static void AssertSpectraMatch(double[] expected, double[] actual, double tol)
    {
        int n = expected.Length / 2;
        Assert.Equal(expected.Length, actual.Length);
        var used = new bool[n];
        for (int e = 0; e < n; e++)
        {
            int best = -1;
            double bestDist = double.MaxValue;
            for (int k = 0; k < n; k++)
            {
                if (used[k]) continue;
                double dr = actual[k * 2] - expected[e * 2], di = actual[k * 2 + 1] - expected[e * 2 + 1];
                double dist = Math.Sqrt(dr * dr + di * di);
                if (dist < bestDist) { bestDist = dist; best = k; }
            }
            used[best] = true;
            double scale = Math.Max(1.0, Math.Sqrt(expected[e * 2] * expected[e * 2] + expected[e * 2 + 1] * expected[e * 2 + 1]));
            Assert.True(bestDist <= tol * scale,
                $"numpy eigenvalue {expected[e * 2]} + {expected[e * 2 + 1]}i: nearest is {actual[best * 2]} + {actual[best * 2 + 1]}i (|diff| {bestDist:E3})");
        }
    }

    /// <summary>AiDotNet ReservoirLayer test generator: entry is U(-0.5, 0.5) with probability <paramref name="density"/>.</summary>
    private static double[] RandomReservoir(int size, double density, int seed)
    {
        var random = new Random(seed);
        var values = new double[size * size];
        for (int i = 0; i < values.Length; i++)
            values[i] = random.NextDouble() < density ? random.NextDouble() - 0.5 : 0.0;
        return values;
    }
}
