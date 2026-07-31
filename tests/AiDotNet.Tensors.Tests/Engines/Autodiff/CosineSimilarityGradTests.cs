using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// <c>TensorCosineSimilarity(x1, x2, dim, eps)</c> must record a gradient, and it must be the
/// per-slice derivative rather than the whole-tensor scalar one.
/// </summary>
/// <remarks>
/// Found by the gradcheck sweep ("TensorCosineSimilarity: no gradient for ANY of its 2 tensor
/// input(s)"). The op recorded nothing. A <c>CosineSimilarityBackward</c> did exist, but it belongs to
/// <c>TensorCosineSimilarityLoss</c> — a whole-tensor SCALAR loss — so it reads <c>gradOutput[0]</c>,
/// sums over every element, and hardcodes eps = 1e-8. Wiring that one up would have been correct only
/// for a single-slice input, so this op gets its own dim-aware backward and the loss op's is untouched.
/// </remarks>
public class CosineSimilarityGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public CosineSimilarityGradTests(ITestOutputHelper o) => _out = o;

    private static Tensor<double> Rand(int[] shape, int seed, double lo = -1.0, double hi = 1.0)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = lo + rng.NextDouble() * (hi - lo);
        return t;
    }

    private void Check(Tensor<double> x1, Tensor<double> x2, int dim, double eps, string label)
    {
        Func<Tensor<double>> fwd = () => _engine.TensorCosineSimilarity(x1, x2, dim, eps);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(fwd(), null);
        var grads = tape.ComputeGradients(loss, [x1, x2]);

        Assert.True(grads.TryGetValue(x1, out var g1) && g1 is not null, $"{label}: no gradient for x1");
        Assert.True(grads.TryGetValue(x2, out var g2) && g2 is not null, $"{label}: no gradient for x2");

        const double h = 1e-6;
        foreach (var (name, t, g) in new[] { ("x1", x1, g1!), ("x2", x2, g2!) })
        {
            for (int i = 0; i < t.Length; i++)
            {
                double orig = t[i];
                t[i] = orig + h; double lp = _engine.TensorSum(fwd());
                t[i] = orig - h; double lm = _engine.TensorSum(fwd());
                t[i] = orig;
                double numerical = (lp - lm) / (2 * h);
                double denom = Math.Max(1.0, Math.Max(Math.Abs(g[i]), Math.Abs(numerical)));
                double rel = Math.Abs(g[i] - numerical) / denom;
                _out.WriteLine($"{label} d/d{name}[{i}] analytical={g[i]:G10} numerical={numerical:G10} rel={rel:E3}");
                Assert.True(rel < 1e-6,
                    $"{label}: d/d{name}[{i}] analytical {g[i]:G10} vs numerical {numerical:G10}");
            }
        }
    }

    [Fact]
    public void Rank1_LastDim_GradientMatchesFiniteDifferences()
        => Check(Rand([5], 1), Rand([5], 2), dim: -1, eps: 1e-8, "rank1");

    /// <summary>
    /// The case the whole-tensor backward could never get right: 3 independent similarity slices, each
    /// with its own norms. A scalar-style backward would mix all three.
    /// </summary>
    [Fact]
    public void Rank2_ReduceLastDim_MultipleSlices_GradientMatchesFiniteDifferences()
        => Check(Rand([3, 4], 3), Rand([3, 4], 4), dim: -1, eps: 1e-8, "rank2-dim1");

    /// <summary>Reducing an INTERIOR/leading axis exercises the outer/inner striding.</summary>
    [Fact]
    public void Rank2_ReduceFirstDim_GradientMatchesFiniteDifferences()
        => Check(Rand([4, 3], 5), Rand([4, 3], 6), dim: 0, eps: 1e-8, "rank2-dim0");

    [Fact]
    public void Rank3_ReduceMiddleDim_GradientMatchesFiniteDifferences()
        => Check(Rand([2, 3, 2], 7), Rand([2, 3, 2], 8), dim: 1, eps: 1e-8, "rank3-dim1");

    /// <summary>
    /// When a norm falls below eps the forward pins that factor to the constant eps, so the
    /// self-normalising term must drop out. A backward derived from the idealised unclamped formula
    /// disagrees with finite differences here.
    /// </summary>
    [Fact]
    public void NormBelowEps_ClampBranch_GradientMatchesFiniteDifferences()
    {
        // ‖x1‖ ≈ 1e-3, comfortably below eps = 1e-2, so the clamp is active for x1 but not x2.
        var x1 = Rand([4], 9, -1e-3, 1e-3);
        var x2 = Rand([4], 10);
        Check(x1, x2, dim: -1, eps: 1e-2, "eps-clamped");
    }

    /// <summary>The eps argument must actually be honoured, not silently replaced by 1e-8.</summary>
    [Fact]
    public void EpsArgument_ChangesTheResult()
    {
        var x1 = Rand([4], 11, -1e-4, 1e-4);
        var x2 = Rand([4], 12);
        double tight = _engine.TensorCosineSimilarity(x1, x2, -1, 1e-8)[0];
        double loose = _engine.TensorCosineSimilarity(x1, x2, -1, 1e-1)[0];
        _out.WriteLine($"eps=1e-8 -> {tight:G10}, eps=1e-1 -> {loose:G10}");
        Assert.NotEqual(tight, loose);
    }
}
