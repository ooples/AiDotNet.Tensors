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

    internal static double[] Reference(Tensor<float> a, Tensor<float> b, int[] ends, bool transposeB, Tensor<float>? bias = null)
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
                if (bias is not null) s += bias.Rank == 2 ? bias[g, j] : bias[j];
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

    // d/dbias of sum(out * w): the per-group column sums of w over the group's rows ([G, N]), or their total ([N]).
    internal static double[] ReferenceBiasGrad(Tensor<float> w, int[] ends, bool perGroup)
    {
        int n = w.Shape[1];
        var d = new double[(perGroup ? ends.Length : 1) * n];
        for (int r = 0; r < w.Shape[0]; r++)
        {
            int g = GroupOf(ends, r);
            if (g < 0) continue;
            for (int j = 0; j < n; j++) d[(perGroup ? g * n : 0) + j] += w[r, j];
        }
        return d;
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

    // Counts the native op's calls: the public grouped_mm must reach it exactly once (no per-group composition).
    private sealed class CountingEngine : CpuEngine
    {
        internal int Calls;

        internal override Tensor<T> GroupedRowsMatMul<T>(Tensor<T> a, Tensor<T> b, int[] ends, bool transposeB, Tensor<T>? bias = null)
        {
            Calls++;
            return base.GroupedRowsMatMul(a, b, ends, transposeB, bias);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void PublicGroupedMatMul_DispatchesToTheNativeOpOnce_AsOneTapeNode(int biasRank)
    {
        var engine = new CountingEngine();
        int k = 16, n = 12, g = UnbalancedEnds.Length;
        var a = Random(3, UnbalancedRows, k); var b = Random(4, g, k, n);
        var bias = biasRank == 0 ? null : biasRank == 1 ? Random(12, n) : Random(12, g, n);
        using var tape = new GradientTape<float>();
        var y = engine.TensorGroupedMatMul(a, b, new Tensor<int>(UnbalancedEnds, new[] { g }), bias);
        Assert.Equal(1, engine.Calls);
        Assert.Equal(1, tape.EntryCount);   // one native node, not a narrow + matmul (+ bias add) per group and a concat
        AssertClose(Reference(a, b, UnbalancedEnds, false, bias), y, 1e-5, "y");
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    public void Bias_ForwardAndAllGradients_MatchReference(int biasRank, bool transposeB)
    {
        int k = 10, n = 9, g = UnbalancedEnds.Length;
        var a = Random(5, UnbalancedRows, k);
        var b = transposeB ? Random(6, g, n, k) : Random(6, g, k, n);
        var bias = biasRank == 1 ? Random(13, n) : Random(13, g, n);
        var w = Random(7, UnbalancedRows, n);
        Tensor<float> y, ga, gb, gbias;
        using (var tape = new GradientTape<float>())
        {
            y = _engine.GroupedRowsMatMul(a, b, UnbalancedEnds, transposeB, bias);
            var loss = _engine.ReduceSum(_engine.TensorMultiply(y, w), null);
            var grads = tape.ComputeGradients(loss, new[] { a, b, bias });
            ga = grads[a]; gb = grads[b]; gbias = grads[bias];
        }
        AssertClose(Reference(a, b, UnbalancedEnds, transposeB, bias), y, 1e-5, "y");   // tail rows: zero, no bias
        var (da, db) = ReferenceGrads(a, b, w, UnbalancedEnds, transposeB);
        AssertClose(da, ga, 1e-4, "da");
        AssertClose(db, gb, 1e-4, "db");
        AssertClose(ReferenceBiasGrad(w, UnbalancedEnds, biasRank == 2), gbias, 1e-4, "dbias");
    }

    // Double backward through the bias: L = sum(y^2) / 2 gives dL/dbias = per-group column sums of y; differentiating
    // s = sum(dL/dbias * v) again gives d s / d bias[g, j] = rows(g) * v[g, j] ([G, N]) or sum_g rows(g) * v[j] ([N]).
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void BiasGradient_IsDifferentiable_UnderCreateGraph(int biasRank)
    {
        int k = 5, n = 4, g = UnbalancedEnds.Length;
        var a = Random(21, UnbalancedRows, k); var b = Random(22, g, k, n);
        var bias = biasRank == 1 ? Random(23, n) : Random(23, g, n);
        var v = biasRank == 1 ? Random(24, n) : Random(24, g, n);
        using var tape = new GradientTape<float>();
        var y = _engine.GroupedRowsMatMul(a, b, UnbalancedEnds, false, bias);
        var loss = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(y, y), null), 0.5f);
        var gBias = tape.ComputeGradients(loss, new[] { bias }, createGraph: true)[bias];
        var s = _engine.ReduceSum(_engine.TensorMultiply(gBias, v), null);
        var second = tape.ComputeGradients(s, new[] { bias }, createGraph: false)[bias].Contiguous().ToArray();
        var expected = new double[second.Length];
        for (int grp = 0; grp < g; grp++)
        {
            int rows = UnbalancedEnds[grp] - (grp == 0 ? 0 : UnbalancedEnds[grp - 1]);
            for (int j = 0; j < n; j++)
                if (biasRank == 2) expected[grp * n + j] = rows * v[grp, j];
                else expected[j] += rows * v[j];
        }
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], second[i], 3);
    }

    // Double backward through the weight gradient: L = sum(y * w) gives dL/db_g = a_g^T w_g; differentiating
    // s = sum(dL/db * u) with respect to a gives da_r = u_g w_r^T for routed rows and zero for the tail.
    [Fact]
    public void WeightGradient_IsDifferentiable_UnderCreateGraph()
    {
        int k = 5, n = 4, g = UnbalancedEnds.Length;
        var a = Random(25, UnbalancedRows, k); var b = Random(26, g, k, n);
        var w = Random(27, UnbalancedRows, n); var u = Random(28, g, k, n);
        using var tape = new GradientTape<float>();
        var y = _engine.GroupedRowsMatMul(a, b, UnbalancedEnds, false);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(y, w), null);
        var gB = tape.ComputeGradients(loss, new[] { b }, createGraph: true)[b];
        var s = _engine.ReduceSum(_engine.TensorMultiply(gB, u), null);
        var da = tape.ComputeGradients(s, new[] { a }, createGraph: false)[a];
        var expected = new double[UnbalancedRows * k];
        for (int r = 0; r < UnbalancedRows; r++)
        {
            int grp = GroupOf(UnbalancedEnds, r);
            if (grp < 0) continue;
            for (int q = 0; q < k; q++)
            {
                double acc = 0;
                for (int j = 0; j < n; j++) acc += (double)u[grp, q, j] * w[r, j];
                expected[r * k + q] = acc;
            }
        }
        AssertClose(expected, da, 1e-4, "d2/da");
    }

    [Fact]
    public void OffsetAndStridedViews_MatchReference()
    {
        int k = 8, n = 6, g = UnbalancedEnds.Length;
        // a as rows 3.. of a larger tensor (a storage offset), then as a column slice of that (not contiguous).
        var big = Random(14, UnbalancedRows + 5, k + 4);
        var rowsView = _engine.TensorNarrow(big, 0, 3, UnbalancedRows);
        var colsView = _engine.TensorNarrow(rowsView, 1, 2, k);
        Assert.False(colsView.IsContiguous);
        var b = Random(15, g, k, n);
        var dense = new Tensor<float>(colsView.ToArray(), new[] { UnbalancedRows, k });
        AssertClose(Reference(dense, b, UnbalancedEnds, false), _engine.GroupedRowsMatMul(colsView, b, UnbalancedEnds, false), 1e-5, "strided a");
        var rowsDense = new Tensor<float>(rowsView.ToArray(), new[] { UnbalancedRows, k + 4 });
        var b2 = Random(17, g, k + 4, n);
        AssertClose(Reference(rowsDense, b2, UnbalancedEnds, false), _engine.GroupedRowsMatMul(rowsView, b2, UnbalancedEnds, false), 1e-5, "offset a");
        // b as groups 1.. of a larger tensor (a storage offset).
        var bigB = Random(16, g + 1, k, n);
        var bView = _engine.TensorNarrow(bigB, 0, 1, g);
        var bDense = new Tensor<float>(bView.ToArray(), new[] { g, k, n });
        AssertClose(Reference(dense, bDense, UnbalancedEnds, false), _engine.GroupedRowsMatMul(dense, bView, UnbalancedEnds, false), 1e-5, "offset b");
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
    }

    [Fact]
    public void Rejects_AnOffsetCountThatIsNotTheGroupCount()
    {
        var a = Random(10, 6, 4); var b = Random(11, 2, 4, 3);
        Assert.Throws<ArgumentException>(() => _engine.GroupedRowsMatMul(a, b, new[] { 6 }, false));
    }

    [Fact]
    public void Accepts_ALastOffsetEqualToTheRowCount()
    {
        var a = Random(10, 6, 4); var b = Random(11, 2, 4, 3);
        var ends = new[] { 2, 6 };
        AssertClose(Reference(a, b, ends, false), _engine.GroupedRowsMatMul(a, b, ends, false), 1e-5, "y");
    }
}
