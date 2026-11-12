using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The batched numeric recurrences on the CPU engine against reference values: the tridiagonal solve and its gradient
/// against torch.linalg.solve on the dense matrix (PyTorch 2.12.1, float64), the Fritsch-Carlson limiter against the
/// hand-computed sequential recurrence, the CTC lattice against torch.ops.aten._ctc_loss and the ctc_loss gradient
/// (PyTorch 2.12.1, float64), and the assignment against the SciPy documentation example
/// (scipy.optimize.linear_sum_assignment, SciPy 1.13 docstring) and brute force over every permutation.
/// </summary>
public class NumericRecurrenceOpsTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    internal static Tensor<float> Random(int seed, double scale, double shift, params int[] shape)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(shift + scale * (rng.NextDouble() * 2 - 1));
        return t;
    }

    internal static void AssertClose(IReadOnlyList<double> expected, Tensor<float> actual, double tol, string what)
    {
        var a = actual.ToArray();
        Assert.Equal(expected.Count, a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            if (double.IsNegativeInfinity(expected[i]))
                Assert.True(float.IsNegativeInfinity(a[i]), $"{what}[{i}]: expected -inf, got {a[i]:R}");
            else if (double.IsNaN(expected[i]))
                Assert.True(float.IsNaN(a[i]), $"{what}[{i}]: expected NaN, got {a[i]:R}");
            else
                Assert.True(Math.Abs(expected[i] - a[i]) <= tol * Math.Max(1, Math.Abs(expected[i])),
                    $"{what}[{i}]: expected {expected[i]:R}, got {a[i]:R}");
        }
    }

    // ---------------------------------------------------------------- tridiagonal solve

    // Two 4 x 4 systems; x from torch.linalg.solve on the dense matrices (PyTorch 2.12.1, float64).
    internal static readonly double[] TriLower = { 1.0, -0.5, 2.0, 0.5, 0.25, -1.5 };
    internal static readonly double[] TriDiagonal = { 4.0, 5.0, 3.0, 6.0, -3.0, 2.5, 4.0, -5.0 };
    internal static readonly double[] TriUpper = { 2.0, 1.0, -1.0, 1.0, -0.75, 0.5 };
    internal static readonly double[] TriRhs = { 1.0, 2.0, 3.0, 4.0, -2.0, 0.0, 1.5, 7.0 };
    internal static readonly double[] TriSolution =
    {
        0.18010752688172044, 0.13978494623655915, 1.1209677419354838, 0.29301075268817206,
        0.6783579115185333, 0.035073734555599825, 0.5691510561976882, -1.5707453168593064,
    };

    private static Tensor<double> D(double[] v, params int[] shape) => new Tensor<double>(v, shape);

    [Fact]
    public void Tridiagonal_MatchesTorchDenseSolve()
    {
        var x = _engine.TensorTridiagonalSolve(D(TriLower, 2, 3), D(TriDiagonal, 2, 4), D(TriUpper, 2, 3), D(TriRhs, 2, 4));
        Assert.Equal(new[] { 2, 4 }, x.Shape.ToArray());
        var a = x.ToArray();
        for (int i = 0; i < a.Length; i++) Assert.Equal(TriSolution[i], a[i], 12);
    }

    [Fact]
    public void Tridiagonal_Float_MatchesDenseReference_ForRandomDominantSystems()
    {
        int batch = 5, n = 37;
        var lo = Random(1, 1, 0, 1, batch, n - 1); var up = Random(2, 1, 0, 1, batch, n - 1);
        var dg = Random(3, 1, 4, 1, batch, n); var rh = Random(4, 1, 0, 1, batch, n);
        var x = _engine.TensorTridiagonalSolve(lo, dg, up, rh);
        Assert.Equal(new[] { 1, batch, n }, x.Shape.ToArray());
        AssertClose(DenseSolve(lo.ToArray(), dg.ToArray(), up.ToArray(), rh.ToArray(), batch, n), x, 1e-5, "x");
    }

    [Fact]
    public void Tridiagonal_SingleUnknown_IsRhsOverDiagonal()
    {
        var x = _engine.TensorTridiagonalSolve(new Tensor<double>(new[] { 2, 0 }), D(new[] { 4.0, -2.0 }, 2, 1),
            new Tensor<double>(new[] { 2, 0 }), D(new[] { 2.0, 3.0 }, 2, 1));
        Assert.Equal(new[] { 0.5, -1.5 }, x.ToArray());
    }

    [Fact]
    public void Tridiagonal_RejectsBadBandShape()
    {
        Assert.Throws<ArgumentException>(() => _engine.TensorTridiagonalSolve(D(new double[4], 4), D(new double[4], 4),
            D(new double[3], 3), D(new double[4], 4)));
    }

    // d/d(inputs) of sum(w * x) for the first system, from torch autograd through torch.linalg.solve (PyTorch 2.12.1).
    [Fact]
    public void Tridiagonal_Gradients_MatchTorchAutograd()
    {
        var lo = D(TriLower.Take(3).ToArray(), 3); var dg = D(TriDiagonal.Take(4).ToArray(), 4);
        var up = D(TriUpper.Take(3).ToArray(), 3); var rh = D(TriRhs.Take(4).ToArray(), 4);
        var w = D(new[] { 0.3, -1.0, 2.0, 0.5 }, 4);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var x = _engine.TensorTridiagonalSolve(lo, dg, up, rh);
            var loss = _engine.ReduceSum(_engine.TensorMultiply(x, w), null);
            grads = tape.ComputeGradients(loss, new[] { lo, dg, up, rh });
        }
        AssertDoubles(new[] { 0.033891201294947396, -0.08477280610475199, -0.20671609434616714 }, grads[lo], "dlower");
        AssertDoubles(new[] { -0.02198086483986588, 0.026303618915481563, -0.679812695109261, -0.05403370331830269 }, grads[dg], "ddiag");
        AssertDoubles(new[] { -0.017059775696612326, 0.21093479014915018, -0.17769684356573012 }, grads[up], "dupper");
        AssertDoubles(new[] { 0.12204301075268817, -0.1881720430107527, 0.6064516129032257, 0.18440860215053761 }, grads[rh], "drhs");
    }

    private static void AssertDoubles(double[] expected, Tensor<double> actual, string what)
    {
        var a = actual.ToArray();
        Assert.Equal(expected.Length, a.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(expected[i] - a[i]) <= 1e-12, $"{what}[{i}]: expected {expected[i]:R}, got {a[i]:R}");
    }

    // Gaussian elimination with partial pivoting on the dense matrix, in double.
    internal static double[] DenseSolve(float[] lo, float[] dg, float[] up, float[] rh, int batch, int n)
    {
        var x = new double[batch * n];
        for (int b = 0; b < batch; b++)
        {
            var a = new double[n, n + 1];
            for (int i = 0; i < n; i++)
            {
                a[i, i] = dg[b * n + i];
                a[i, n] = rh[b * n + i];
                if (i + 1 < n) { a[i + 1, i] = lo[b * (n - 1) + i]; a[i, i + 1] = up[b * (n - 1) + i]; }
            }
            for (int c = 0; c < n; c++)
            {
                int p = c;
                for (int r = c + 1; r < n; r++) if (Math.Abs(a[r, c]) > Math.Abs(a[p, c])) p = r;
                for (int k = 0; k <= n; k++) (a[c, k], a[p, k]) = (a[p, k], a[c, k]);
                for (int r = c + 1; r < n; r++)
                {
                    double f = a[r, c] / a[c, c];
                    for (int k = c; k <= n; k++) a[r, k] -= f * a[c, k];
                }
            }
            for (int i = n - 1; i >= 0; i--)
            {
                double s = a[i, n];
                for (int k = i + 1; k < n; k++) s -= a[i, k] * x[b * n + k];
                x[b * n + i] = s / a[i, i];
            }
        }
        return x;
    }

    // ---------------------------------------------------------------- Fritsch-Carlson

    // m = [1, 4, 0.5], d = [1, 1]: step 0 scales (1, 4) by 3 / sqrt(17); step 1 then reads the scaled m[1] = 12 / sqrt(17)
    // (a^2 + b^2 = 144/17 + 0.25 < 9, unchanged). A parallel limiter reading the original m[1] = 4 would also rescale
    // step 1 (16.25 > 9), so this pins the sequential order. Second series: a zero secant flattens both ends.
    [Fact]
    public void FritschCarlson_IsSequential_AndFlattensZeroSecants()
    {
        var m = D(new[] { 1.0, 4.0, 0.5, 2.0, 3.0, 1.0 }, 2, 3);
        var d = D(new[] { 1.0, 1.0, 0.0, 1.0 }, 2, 2);
        var r = _engine.TensorFritschCarlsonLimit(m, d).ToArray();
        double tau = 3.0 / Math.Sqrt(17.0);
        Assert.Equal(new[] { tau * 1.0, tau * 4.0, 0.5, 0.0, 0.0, 1.0 }, r);
    }

    [Fact]
    public void FritschCarlson_MatchesScalarRecurrence_ForRandomSeries()
    {
        int batch = 7, n = 29;
        var m = Random(11, 3, 0, batch, n); var d = Random(12, 1, 0, batch, n - 1);
        var r = _engine.TensorFritschCarlsonLimit(m, d);
        AssertClose(ScalarFritschCarlson(m.ToArray(), d.ToArray(), batch, n), r, 1e-6, "m");
    }

    // The AiDotNet MonotoneCubicInterpolation.CalculateSlopes step-2 loop, verbatim in double.
    internal static double[] ScalarFritschCarlson(float[] slopes, float[] secants, int batch, int n)
    {
        var m = slopes.Select(v => (double)v).ToArray();
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < n - 1; i++)
            {
                double di = secants[b * (n - 1) + i];
                if (di == 0) { m[b * n + i] = 0; m[b * n + i + 1] = 0; continue; }
                double alpha = m[b * n + i] / di, beta = m[b * n + i + 1] / di, sum = alpha * alpha + beta * beta;
                if (sum > 9)
                {
                    double tau = 3 / Math.Sqrt(sum);
                    m[b * n + i] = tau * (alpha * di);
                    m[b * n + i + 1] = tau * (beta * di);
                }
            }
        return m;
    }

    // ---------------------------------------------------------------- CTC lattice

    // logProbs = log_softmax(sin(1 + 7t + 3n + c)), T 4, N 2, C 3, targets [1, 2] and [1, 1] (a repeat: no skip), blank 0.
    internal static Tensor<double> CtcLogProbs() => D(new[]
    {
        -0.9425784028566115, -0.8747519608388263, -1.6429293796046407, -1.2322256676323229, -1.4343474469875332, -0.7548386705233207,
        -0.5750678478262817, -1.1523076092079068, -2.1084472053390333, -1.9061690447043704, -1.442751756154102, -0.48601180132702604,
        -0.46429114407686584, -1.402482300899048, -2.0759764761135395, -2.168112004670142, -1.2672475482355132, -0.504179507170838,
        -0.610173276024239, -1.4475423709090056, -1.5069003287404588, -1.8590455056222133, -0.9641353050448375, -0.7703178271199371,
    }, 4, 2, 3);

    internal static readonly Tensor<int> CtcTargets = new Tensor<int>(new[] { 1, 2, 1, 1 }, new[] { 4 });
    internal static readonly int[] CtcInputLengths = { 4, 4 };
    internal static readonly int[] CtcTargetLengths = { 2, 2 };

    // torch.ops.aten._ctc_loss(lp, targets, [4, 4], [2, 2], 0, False): (nll, log_alpha [N, T, 5]).
    internal static readonly double[] TorchNll = { 1.8133197504455754, 4.392980097406002 };
    internal static readonly double[] TorchLogAlpha =
    {
        -0.9425784028566115, -0.8747519608388263, double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity,
        -1.5176462506828932, -1.3672506674110447, -1.449819808665108, -2.98319916617786, double.NegativeInfinity,
        -1.981937394759759, -2.148958886084843, -1.1788272356451874, -2.692070576096862, -3.4474903102547256,
        -2.592110670783998, -2.8163603540371995, -1.4676178849092807, -2.2162066845942276, -2.9171084140717483,
        -1.2322256676323229, -1.4343474469875332, double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity,
        -3.1383947123366935, -2.0777931502218046, -3.3405164916919037, double.NegativeInfinity, double.NegativeInfinity,
        -5.306506717006835, -3.0477196369448616, -3.996795585024487, -4.607764039927417, double.NegativeInfinity,
        -7.1655522226290485, -3.912482904230778, -4.579550945788844, -4.52731579494472, -6.466809545549631,
    };

    // exp(lp) - d ctc_loss(sum) / d lp: torch's ctc_loss gradient assumes log_softmax inputs, so this is the posterior.
    internal static readonly double[] TorchPosteriors =
    {
        0.2918993044908422, 0.7081006955091582, 0.0, 0.24310918000969317, 0.7568908199903075, 0.0,
        0.41160536867822095, 0.4526178569010056, 0.13577677442077357, 0.5582717312455916, 0.44172826875440807, 0.0,
        0.5239410743814612, 0.15841462657444186, 0.31764429904409713, 0.5666855057806336, 0.4333144942193659, 0.0,
        0.3316123331279897, 0.0, 0.6683876668720105, 0.12570348385423563, 0.8742965161457643, 0.0,
    };

    [Fact]
    public void Ctc_LossAndLogAlpha_MatchTorch()
    {
        var lp = CtcLogProbs();
        var loss = _engine.TensorCTCLoss(lp, CtcTargets, CtcInputLengths, CtcTargetLengths, 0).ToArray();
        for (int n = 0; n < 2; n++) Assert.Equal(TorchNll[n], loss[n], 12);
        var (alpha, beta) = _engine.TensorCTCForwardBackward(lp, CtcTargets, CtcInputLengths, CtcTargetLengths, 0);
        Assert.Equal(new[] { 2, 4, 5 }, alpha.Shape.ToArray());
        Assert.Equal(new[] { 2, 4, 5 }, beta.Shape.ToArray());
        var a = alpha.ToArray();
        for (int i = 0; i < a.Length; i++)
        {
            if (double.IsNegativeInfinity(TorchLogAlpha[i])) Assert.True(double.IsNegativeInfinity(a[i]), $"alpha[{i}] = {a[i]}");
            else Assert.Equal(TorchLogAlpha[i], a[i], 12);
        }
    }

    // alpha + beta summed over the lattice is log P at every valid time step, and beta's last step is 0 at the final
    // blank and label (Graves et al. 2006, eq. 14).
    [Fact]
    public void Ctc_AlphaBeta_SumToLogLikelihoodAtEveryStep()
    {
        var lp = CtcLogProbs();
        var (alpha, beta) = _engine.TensorCTCForwardBackward(lp, CtcTargets, CtcInputLengths, CtcTargetLengths, 0);
        var a = alpha.ToArray(); var b = beta.ToArray();
        for (int n = 0; n < 2; n++)
        {
            for (int t = 0; t < 4; t++)
            {
                double sum = double.NegativeInfinity;
                for (int s = 0; s < 5; s++) sum = CtcLattice.LogSumExp(sum, a[(n * 4 + t) * 5 + s] + b[(n * 4 + t) * 5 + s]);
                Assert.Equal(-TorchNll[n], sum, 12);
            }
            Assert.Equal(0.0, b[(n * 4 + 3) * 5 + 4]);
            Assert.Equal(0.0, b[(n * 4 + 3) * 5 + 3]);
        }
    }

    [Fact]
    public void Ctc_Posteriors_MatchTorch_AndAreMinusTheLossGradient()
    {
        var lp = CtcLogProbs();
        var gamma = _engine.TensorCTCLabelPosteriors(lp, CtcTargets, CtcInputLengths, CtcTargetLengths, 0);
        Assert.Equal(new[] { 4, 2, 3 }, gamma.Shape.ToArray());
        var g = gamma.ToArray();
        for (int i = 0; i < g.Length; i++) Assert.Equal(TorchPosteriors[i], g[i], 12);

        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var loss = _engine.TensorCTCLoss(lp, CtcTargets, CtcInputLengths, CtcTargetLengths, 0);
            grads = tape.ComputeGradients(_engine.ReduceSum(loss, null), new[] { lp });
        }
        var d = grads[lp].ToArray();
        for (int i = 0; i < d.Length; i++) Assert.Equal(-TorchPosteriors[i], d[i], 12);
    }

    // Shorter inputs: rows past an item's input length are zero in gamma and -inf in alpha and beta; an item whose
    // target cannot fit (2 labels needing 3 steps: [1, 1]) has an infinite loss and NaN posterior rows.
    [Fact]
    public void Ctc_ShortAndInfeasibleItems()
    {
        var lp = CtcLogProbs();
        int[] inputLengths = { 3, 1 };
        var loss = _engine.TensorCTCLoss(lp, CtcTargets, inputLengths, CtcTargetLengths, 0).ToArray();
        Assert.True(double.IsPositiveInfinity(loss[1]));
        var gamma = _engine.TensorCTCLabelPosteriors(lp, CtcTargets, inputLengths, CtcTargetLengths, 0).ToArray();
        for (int c = 0; c < 3; c++)
        {
            Assert.Equal(0.0, gamma[(3 * 2 + 0) * 3 + c]);      // t = 3 >= 3 for item 0
            Assert.True(double.IsNaN(gamma[(0 * 2 + 1) * 3 + c]));   // item 1 at t = 0: no alignment
            Assert.Equal(0.0, gamma[(1 * 2 + 1) * 3 + c]);      // item 1 past its single step
        }
        for (int t = 0; t < 3; t++)
            Assert.Equal(1.0, gamma[(t * 2) * 3] + gamma[(t * 2) * 3 + 1] + gamma[(t * 2) * 3 + 2], 12);
        var (alpha, beta) = _engine.TensorCTCForwardBackward(lp, CtcTargets, inputLengths, CtcTargetLengths, 0);
        var a = alpha.ToArray(); var b = beta.ToArray();
        for (int s = 0; s < 5; s++)
        {
            Assert.True(double.IsNegativeInfinity(a[(0 * 4 + 3) * 5 + s]));
            Assert.True(double.IsNegativeInfinity(b[(0 * 4 + 3) * 5 + s]));
        }
    }

    // ---------------------------------------------------------------- linear sum assignment

    // scipy.optimize.linear_sum_assignment docstring: cost [[4, 1, 3], [2, 0, 5], [3, 2, 2]] -> col_ind [1, 0, 2], total 5.
    [Fact]
    public void Assignment_MatchesScipyDocExample()
    {
        var cost = new Tensor<double>(new[] { 4.0, 1, 3, 2, 0, 5, 3, 2, 2 }, new[] { 3, 3 });
        var r = _engine.TensorLinearSumAssignment(cost);
        Assert.Equal(new[] { 3 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1, 0, 2 }, r.ToArray());
    }

    [Fact]
    public void Assignment_IsOptimal_ForRandomBatches()
    {
        int batch = 6, size = 6;
        var cost = Random(21, 5, 0, batch, size, size);
        var r = _engine.TensorLinearSumAssignment(cost).ToArray();
        var c = cost.ToArray();
        for (int b = 0; b < batch; b++)
        {
            var perm = r.Skip(b * size).Take(size).ToArray();
            Assert.Equal(Enumerable.Range(0, size), perm.OrderBy(v => v));
            Assert.Equal(BruteForceAssignment(c, b, size), perm);
        }
    }

    // NaN entries are a last resort: avoided whenever any assignment can avoid them, used only where none can.
    [Fact]
    public void Assignment_TreatsNaNAsLastResort()
    {
        float nan = float.NaN;
        var avoidable = new Tensor<float>(new[] { nan, 9f, 1f, nan }, new[] { 2, 2 });
        Assert.Equal(new[] { 1, 0 }, _engine.TensorLinearSumAssignment(avoidable).ToArray());
        // Row 0 is all NaN: it takes the column the rest of the optimum leaves.
        var forced = new Tensor<float>(new[] { nan, nan, nan, 5f, 1f, 7f, 2f, 8f, 3f }, new[] { 3, 3 });
        Assert.Equal(new[] { 2, 1, 0 }, _engine.TensorLinearSumAssignment(forced).ToArray());
    }

    // The unique optimum over all S! permutations (the random costs have no ties).
    [Fact]
    public void Assignment_ExtremeFiniteCostsAndNonFinitePenalties_StayFiniteInternally()
    {
        double h = double.MaxValue;
        var finite = new Tensor<double>(new[] { h, -h, -h, h }, new[] { 2, 2 });
        Assert.Equal(new[] { 1, 0 }, _engine.TensorLinearSumAssignment(finite).ToArray());
        var forced = new Tensor<double>(new[] { double.NaN, double.NaN, h, -h }, new[] { 2, 2 });
        Assert.Equal(new[] { 0, 1 }, _engine.TensorLinearSumAssignment(forced).ToArray());
    }

    internal static int[] BruteForceAssignment(float[] cost, int b, int size)
    {
        int[] best = Array.Empty<int>();
        double bestCost = double.PositiveInfinity;
        foreach (var perm in Permutations(Enumerable.Range(0, size).ToArray(), 0))
        {
            double s = 0;
            for (int i = 0; i < size; i++) s += cost[(b * size + i) * size + perm[i]];
            if (s < bestCost) { bestCost = s; best = (int[])perm.Clone(); }
        }
        return best;
    }

    private static IEnumerable<int[]> Permutations(int[] a, int k)
    {
        if (k == a.Length) { yield return a; yield break; }
        for (int i = k; i < a.Length; i++)
        {
            (a[k], a[i]) = (a[i], a[k]);
            foreach (var p in Permutations(a, k + 1)) yield return p;
            (a[k], a[i]) = (a[i], a[k]);
        }
    }
}
