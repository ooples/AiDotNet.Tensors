using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// <c>TensorPDist</c> and <c>TensorCDist</c> must record gradients for their inputs.
/// </summary>
/// <remarks>
/// <para>
/// Both were NO-GRADIENT leads from the gradcheck sweep: each computed its result and returned it with
/// no <c>DifferentiableOps.Record*</c> call, and neither preserved the user-facing reference before
/// <c>.Contiguous()</c>, so even a naive record would have attached the tape to a throwaway copy.
/// </para>
/// <para>
/// For dist = (Σ|d_k|^p)^(1/p) the per-term derivative is
/// |d_k|^(p−1)·sign(d_k)·dist^(1−p). At dist = 0 the p-norm has a kink and only a subgradient set
/// exists; 0 is chosen, matching PyTorch. Probe points below keep all distances well away from 0 so
/// finite differences are valid, and the degenerate case is asserted separately.
/// </para>
/// </remarks>
public class DistanceOpGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public DistanceOpGradTests(ITestOutputHelper o) => _out = o;

    private static Tensor<double> Rand(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([rows, cols]);
        for (int i = 0; i < t.Length; i++) t[i] = -1.0 + rng.NextDouble() * 2.0;
        return t;
    }

    private void CheckFd(Tensor<double>[] wrt, Func<Tensor<double>> fwd, string label, double tol = 1e-6)
    {
        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(fwd(), null);
        var grads = tape.ComputeGradients(loss, wrt);

        const double h = 1e-6;
        for (int w = 0; w < wrt.Length; w++)
        {
            var t = wrt[w];
            Assert.True(grads.TryGetValue(t, out var g) && g is not null, $"{label}: no gradient for operand {w}");
            for (int i = 0; i < t.Length; i++)
            {
                double orig = t[i];
                t[i] = orig + h; double lp = _engine.TensorSum(fwd());
                t[i] = orig - h; double lm = _engine.TensorSum(fwd());
                t[i] = orig;
                double numerical = (lp - lm) / (2 * h);
                double denom = Math.Max(1.0, Math.Max(Math.Abs(g![i]), Math.Abs(numerical)));
                double rel = Math.Abs(g[i] - numerical) / denom;
                _out.WriteLine($"{label} d/d[{w}][{i}] analytical={g[i]:G10} numerical={numerical:G10} rel={rel:E3}");
                Assert.True(rel < tol,
                    $"{label}: operand {w} index {i} analytical {g[i]:G10} vs numerical {numerical:G10}");
            }
        }
    }

    [Theory]
    [InlineData(2.0)]
    [InlineData(1.0)]
    [InlineData(3.0)]
    public void PDist_GradientMatchesFiniteDifferences(double p)
    {
        var x = Rand(4, 3, seed: 21 + (int)(p * 10));
        CheckFd([x], () => _engine.TensorPDist(x, p), $"pdist-p{p}");
    }

    [Theory]
    [InlineData(2.0)]
    [InlineData(1.0)]
    [InlineData(3.0)]
    public void CDist_GradientMatchesFiniteDifferences(double p)
    {
        var x1 = Rand(3, 4, seed: 31 + (int)(p * 10));
        var x2 = Rand(2, 4, seed: 41 + (int)(p * 10));
        CheckFd([x1, x2], () => _engine.TensorCDist(x1, x2, p), $"cdist-p{p}");
    }

    /// <summary>
    /// Coincident rows put the p-norm exactly on its kink. There is no derivative there, only a
    /// subgradient set containing 0; the gradient must be finite (0), never NaN or infinity.
    /// </summary>
    [Fact]
    public void PDist_CoincidentRows_YieldZeroSubgradientNotNaN()
    {
        var x = new Tensor<double>([2, 2]);
        x[0] = 0.5; x[1] = -0.25;
        x[2] = 0.5; x[3] = -0.25;   // identical to row 0 -> distance exactly 0

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var dist = _engine.TensorPDist(x, 2.0);
        Assert.Equal(0.0, dist[0]);

        var loss = _engine.ReduceSum(dist, null);
        var grads = tape.ComputeGradients(loss, [x]);
        var g = grads[x];
        for (int i = 0; i < x.Length; i++)
        {
            Assert.False(double.IsNaN(g![i]), $"grad[{i}] is NaN at the p-norm kink");
            Assert.False(double.IsInfinity(g[i]), $"grad[{i}] is Infinity at the p-norm kink");
            Assert.Equal(0.0, g[i]);
        }
    }

    [Fact]
    public void CDist_CoincidentRows_YieldZeroSubgradientNotNaN()
    {
        var x1 = new Tensor<double>([1, 2]); x1[0] = 0.3; x1[1] = 0.7;
        var x2 = new Tensor<double>([1, 2]); x2[0] = 0.3; x2[1] = 0.7;

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(_engine.TensorCDist(x1, x2, 2.0), null);
        var grads = tape.ComputeGradients(loss, [x1, x2]);

        foreach (var (name, t) in new[] { ("x1", x1), ("x2", x2) })
        {
            var g = grads[t];
            for (int i = 0; i < t.Length; i++)
            {
                Assert.False(double.IsNaN(g![i]), $"{name} grad[{i}] is NaN at the kink");
                Assert.Equal(0.0, g[i]);
            }
        }
    }

    /// <summary>
    /// PDist pairs are ordered (0,1),(0,2),…,(1,2),… like torch.pdist, and row i / row j receive equal
    /// and opposite gradient. Weighting a SINGLE pair therefore isolates exactly two rows, which pins
    /// the pair ordering — a transposed or off-by-one enumeration would move gradient onto row 2.
    /// </summary>
    [Fact]
    public void PDist_PairOrdering_IsolatesTheExpectedRowPair()
    {
        var x = Rand(3, 2, seed: 77);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var dist = _engine.TensorPDist(x, 2.0);   // pairs: [0]=(0,1) [1]=(0,2) [2]=(1,2)
        // Select only pair index 2 == (row1, row2), so row 0 must receive nothing.
        var mask = new Tensor<double>([3]);
        mask[0] = 0.0; mask[1] = 0.0; mask[2] = 1.0;
        var loss = _engine.ReduceSum(_engine.TensorMultiply(dist, mask), null);
        var grads = tape.ComputeGradients(loss, [x]);
        var g = grads[x];

        Assert.Equal(0.0, g![0]);
        Assert.Equal(0.0, g[1]);
        Assert.NotEqual(0.0, g[2]);   // row 1
        Assert.NotEqual(0.0, g[4]);   // row 2
        // Equal and opposite between the two rows of the selected pair.
        Assert.Equal(g[2], -g[4], 12);
        Assert.Equal(g[3], -g[5], 12);
    }
}
