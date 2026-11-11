using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The native jagged-rows grouped GEMM (2-D a x 3-D b, torch grouped_mm) against an explicit double-precision loop:
/// unbalanced mixture-of-experts loads with empty groups (first, middle, last), rows past the last offset, the
/// transposed-b form, and both gradients of a weighted loss.
/// </summary>
public class GroupedRowsMatMulTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    internal static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    internal static int GroupOf(int[] ends, int row)
    {
        for (int g = 0; g < ends.Length; g++) if (row < ends[g]) return g;
        return -1;   // past the last offset
    }

    internal static double[] Reference(Tensor<float> a, Tensor<float> b, int[] ends, bool transposeB)
    {
        int m = a.Shape[0], k = a.Shape[1], n = transposeB ? b.Shape[1] : b.Shape[2];
        var y = new double[m * n];
        for (int r = 0; r < m; r++)
        {
            int g = GroupOf(ends, r);
            if (g < 0) continue;
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int q = 0; q < k; q++) s += (double)a[r, q] * (transposeB ? b[g, j, q] : b[g, q, j]);
                y[r * n + j] = s;
            }
        }
        return y;
    }

    // d/da and d/db of sum(out * w) for out = grouped(a, b).
    internal static (double[] Da, double[] Db) ReferenceGrads(Tensor<float> a, Tensor<float> b, Tensor<float> w, int[] ends, bool transposeB)
    {
        int m = a.Shape[0], k = a.Shape[1], n = w.Shape[1];
        var da = new double[a.Length]; var db = new double[b.Length];
        for (int r = 0; r < m; r++)
        {
            int g = GroupOf(ends, r);
            if (g < 0) continue;
            for (int q = 0; q < k; q++)
                for (int j = 0; j < n; j++)
                {
                    int bi = transposeB ? (g * n + j) * k + q : (g * k + q) * n + j;
                    da[r * k + q] += (double)w[r, j] * b.GetDataArray()[bi];
                    db[bi] += (double)w[r, j] * a[r, q];
                }
        }
        return (da, db);
    }

    internal static void AssertClose(double[] expected, Tensor<float> actual, double tolerance, string what)
    {
        var a = actual.ToArray();
        Assert.Equal(expected.Length, a.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(expected[i] - a[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: expected {expected[i]:R}, got {a[i]:R}");
    }

    // A routed load like an unbalanced top-k router: 7 groups, empty first/middle/last, one heavy group, tail rows.
    public static readonly int[] UnbalancedEnds = { 0, 3, 3, 40, 41, 52, 52 };
    public const int UnbalancedRows = 57;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Unbalanced_WithEmptyGroupsAndTail_MatchesReference(bool transposeB)
    {
        int k = 24, n = 20, g = UnbalancedEnds.Length;
        var a = Random(1, UnbalancedRows, k);
        var b = transposeB ? Random(2, g, n, k) : Random(2, g, k, n);
        var y = _engine.GroupedRowsMatMul(a, b, UnbalancedEnds, transposeB);
        Assert.Equal(new[] { UnbalancedRows, n }, y.Shape.ToArray());
        AssertClose(Reference(a, b, UnbalancedEnds, transposeB), y, 1e-5, "y");
    }

    [Fact]
    public void PublicGroupedMatMul_TakesTheNativePath_AndMatchesReference()
    {
        int k = 16, n = 12, g = UnbalancedEnds.Length;
        var a = Random(3, UnbalancedRows, k); var b = Random(4, g, k, n);
        var y = _engine.TensorGroupedMatMul(a, b, new Tensor<int>(UnbalancedEnds, new[] { g }));
        AssertClose(Reference(a, b, UnbalancedEnds, false), y, 1e-5, "y");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Gradients_MatchReference(bool transposeB)
    {
        int k = 10, n = 9, g = UnbalancedEnds.Length;
        var a = Random(5, UnbalancedRows, k);
        var b = transposeB ? Random(6, g, n, k) : Random(6, g, k, n);
        var w = Random(7, UnbalancedRows, n);
        Tensor<float> ga, gb;
        using (var tape = new GradientTape<float>())
        {
            var loss = _engine.ReduceSum(_engine.TensorMultiply(_engine.GroupedRowsMatMul(a, b, UnbalancedEnds, transposeB), w), null);
            var grads = tape.ComputeGradients(loss, new[] { a, b });
            ga = grads[a]; gb = grads[b];
        }
        var (da, db) = ReferenceGrads(a, b, w, UnbalancedEnds, transposeB);
        AssertClose(da, ga, 1e-4, "da");   // rows past the last offset: zero
        AssertClose(db, gb, 1e-4, "db");   // empty groups: zero
    }

    [Fact]
    public void Double_MatchesReference()
    {
        int k = 6, n = 5, g = UnbalancedEnds.Length;
        var af = Random(8, UnbalancedRows, k); var bf = Random(9, g, k, n);
        var a = new Tensor<double>(Array.ConvertAll(af.ToArray(), v => (double)v), af.Shape.ToArray());
        var b = new Tensor<double>(Array.ConvertAll(bf.ToArray(), v => (double)v), bf.Shape.ToArray());
        var y = _engine.GroupedRowsMatMul(a, b, UnbalancedEnds, false).ToArray();
        var expected = Reference(af, bf, UnbalancedEnds, false);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], y[i], 10);
    }

    [Fact]
    public void Rejects_DecreasingOffsets_AndOffsetsPastRows()
    {
        var a = Random(10, 6, 4); var b = Random(11, 2, 4, 3);
        Assert.Throws<ArgumentException>(() => _engine.GroupedRowsMatMul(a, b, new[] { 4, 2 }, false));
        Assert.Throws<ArgumentException>(() => _engine.GroupedRowsMatMul(a, b, new[] { 2, 7 }, false));
        Assert.Throws<ArgumentException>(() => _engine.GroupedRowsMatMul(a, b, new[] { 6 }, false));
    }
}
