using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// torch.cdist takes batched inputs [..., P, M] / [..., R, M] with broadcast batch dims, and torch.searchsorted takes
/// an N-D sorted sequence whose rows are searched independently. Both CPU kernels used to accept only the 2-D / 1-D
/// case (the GPU engine falls back to them for every other rank).
/// </summary>
public sealed class BatchedCDistSearchSortedTests
{
    private static Tensor<double> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        return new Tensor<double>(shape, new Vector<double>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => rng.NextDouble() * 2 - 1).ToArray()));
    }

    private static Tensor<double> Slice(Tensor<double> t, int b)
    {
        int m = t.Shape[t.Rank - 2], d = t.Shape[t.Rank - 1];
        return new Tensor<double>(new[] { m, d }, new Vector<double>(t.ToArray().Skip(b * m * d).Take(m * d).ToArray()));
    }

    [Theory]
    [InlineData(2.0)]
    [InlineData(1.0)]
    [InlineData(3.0)]
    public void Batched_cdist_matches_per_batch_cdist_with_gradients(double p)
    {
        var eng = new CpuEngine();
        var x1 = Rand(new[] { 2, 3, 3, 4 }, 1);
        var x2 = Rand(new[] { 3, 5, 4 }, 2); // broadcasts over the leading 2
        var w = Rand(new[] { 2, 3, 3, 5 }, 3);
        Tensor<double> y;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> g;
        using (var tape = new GradientTape<double>())
        {
            y = eng.TensorCDist(x1, x2, p);
            g = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false), new[] { x1, x2 });
        }
        Assert.Equal(new[] { 2, 3, 3, 5 }, y.Shape.ToArray());

        var expected = new double[y.Length];
        var g1 = new double[x1.Length];
        var g2 = new double[x2.Length];
        for (int b = 0; b < 6; b++)
        {
            var a = Slice(x1, b);
            var c = Slice(x2, b % 3);
            var wb = new Tensor<double>(new[] { 3, 5 }, new Vector<double>(w.ToArray().Skip(b * 15).Take(15).ToArray()));
            using var tape = new GradientTape<double>();
            var yb = eng.TensorCDist(a, c, p);
            var gb = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(yb, wb), null, keepDims: false), new[] { a, c });
            Array.Copy(yb.ToArray(), 0, expected, b * 15, 15);
            var ga = gb[a].ToArray();
            var gc = gb[c].ToArray();
            for (int i = 0; i < 12; i++) g1[b * 12 + i] += ga[i];
            for (int i = 0; i < 20; i++) g2[(b % 3) * 20 + i] += gc[i];
        }
        AssertClose(expected, y.ToArray());
        AssertClose(g1, g[x1].ToArray());
        AssertClose(g2, g[x2].ToArray());
    }

    [Fact]
    public void Batched_cdist_broadcasts_singleton_batch_dims_on_both_sides()
    {
        var eng = new CpuEngine();
        var x1 = Rand(new[] { 2, 1, 3, 4 }, 4);
        var x2 = Rand(new[] { 1, 3, 5, 4 }, 5);
        var w = Rand(new[] { 2, 3, 3, 5 }, 6);
        Tensor<double> y;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> g;
        using (var tape = new GradientTape<double>())
        {
            y = eng.TensorCDist(x1, x2, 2.0);
            g = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false), new[] { x1, x2 });
        }
        Assert.Equal(new[] { 2, 3, 3, 5 }, y.Shape.ToArray());

        // Output batch (i, j) pairs x1's batch i with x2's batch j; each input's gradient sums over the other's batch.
        var expected = new double[y.Length];
        var g1 = new double[x1.Length];
        var g2 = new double[x2.Length];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
            {
                int b = i * 3 + j;
                var a = Slice(x1, i);
                var c = Slice(x2, j);
                var wb = new Tensor<double>(new[] { 3, 5 }, new Vector<double>(w.ToArray().Skip(b * 15).Take(15).ToArray()));
                using var tape = new GradientTape<double>();
                var yb = eng.TensorCDist(a, c, 2.0);
                var gb = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(yb, wb), null, keepDims: false), new[] { a, c });
                Array.Copy(yb.ToArray(), 0, expected, b * 15, 15);
                var ga = gb[a].ToArray();
                var gc = gb[c].ToArray();
                for (int k = 0; k < 12; k++) g1[i * 12 + k] += ga[k];
                for (int k = 0; k < 20; k++) g2[j * 20 + k] += gc[k];
            }
        AssertClose(expected, y.ToArray());
        AssertClose(g1, g[x1].ToArray());
        AssertClose(g2, g[x2].ToArray());
    }

    [Fact]
    public void Batched_cdist_broadcasts_a_zero_batch_against_a_singleton()
    {
        var eng = new CpuEngine();
        var y = eng.TensorCDist(new Tensor<double>(new[] { 0, 3, 4 }), Rand(new[] { 1, 5, 4 }, 7), 2.0);
        Assert.Equal(new[] { 0, 3, 5 }, y.Shape.ToArray());
        Assert.Throws<ArgumentException>(() => eng.TensorCDist(new Tensor<double>(new[] { 0, 3, 4 }), Rand(new[] { 2, 5, 4 }, 8)));
    }

    [Fact]
    public void Searchsorted_treats_nan_like_torch_on_every_path()
    {
        var eng = new CpuEngine();
        // torch.searchsorted moves right whenever !(seq[mid] >= x) (or !(seq[mid] > x)), which every comparison with
        // NaN satisfies: a NaN query lands past the end, and a query meeting a NaN in the sequence passes it. The
        // rank-one float and double paths and the batched path must all agree.
        foreach (bool right in new[] { false, true })
        {
            var f1 = eng.TensorSearchSorted(new Tensor<float>(new[] { 1f, 2f, float.NaN }, new[] { 3 }),
                new Tensor<float>(new[] { float.NaN, 2f, 5f }, new[] { 3 }), right).ToArray();
            var d1 = eng.TensorSearchSorted(new Tensor<double>(new[] { 1.0, 2.0, double.NaN }, new[] { 3 }),
                new Tensor<double>(new[] { double.NaN, 2.0, 5.0 }, new[] { 3 }), right).ToArray();
            var f2 = eng.TensorSearchSorted(new Tensor<float>(new[] { 1f, 2f, float.NaN, 0f, 3f, 4f }, new[] { 2, 3 }),
                new Tensor<float>(new[] { float.NaN, 2f, 5f, float.NaN, 3f, 5f }, new[] { 2, 3 }), right).ToArray();
            var d2 = eng.TensorSearchSorted(new Tensor<double>(new[] { 1.0, 2.0, double.NaN, 0.0, 3.0, 4.0 }, new[] { 2, 3 }),
                new Tensor<double>(new[] { double.NaN, 2.0, 5.0, double.NaN, 3.0, 5.0 }, new[] { 2, 3 }), right).ToArray();
            // [1, 2, NaN] with 2: the lower bound stops at 1; the upper bound passes 2 and then the NaN, reaching 3.
            // [0, 3, 4] with 3 (no NaN): 1 and 2.
            var row = new[] { 3, right ? 3 : 1, 3 };
            Assert.Equal(row, f1.Select(v => (int)v).ToArray());
            Assert.Equal(row, d1.Select(v => (int)v).ToArray());
            var rows = new[] { 3, right ? 3 : 1, 3, 3, right ? 2 : 1, 3 };
            Assert.Equal(rows, f2.Select(v => (int)v).ToArray());
            Assert.Equal(rows, d2.Select(v => (int)v).ToArray());
        }
    }

    [Fact]
    public void Cdist_keeps_a_zero_sized_batch_dim()
    {
        var eng = new CpuEngine();
        var y = eng.TensorCDist(new Tensor<float>(new[] { 0, 3, 4 }), new Tensor<float>(new[] { 1, 5, 4 }));
        Assert.Equal(new[] { 0, 3, 5 }, y.Shape.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Nd_searchsorted_places_nan_where_rank_1_does(bool right)
    {
        var eng = new CpuEngine();
        var sd = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 3.0, 5.0 }));
        var vd = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { double.NaN, 3.0 }));
        Assert.Equal(eng.TensorSearchSorted(sd, vd, right).ToArray(),
            eng.TensorSearchSorted(eng.Reshape(sd, new[] { 1, 3 }), eng.Reshape(vd, new[] { 1, 2 }), right).ToArray());
        var sf = new Tensor<float>(new[] { 3 }, new Vector<float>(new[] { 1f, 3f, 5f }));
        var vf = new Tensor<float>(new[] { 2 }, new Vector<float>(new[] { float.NaN, 3f }));
        Assert.Equal(eng.TensorSearchSorted(sf, vf, right).ToArray(),
            eng.TensorSearchSorted(eng.Reshape(sf, new[] { 1, 3 }), eng.Reshape(vf, new[] { 1, 2 }), right).ToArray());
    }

    [Fact]
    public void Cdist_rejects_batch_dims_that_do_not_broadcast()
    {
        var eng = new CpuEngine();
        Assert.Throws<ArgumentException>(() => eng.TensorCDist(Rand(new[] { 2, 3, 4 }, 1), Rand(new[] { 3, 5, 4 }, 2)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Nd_searchsorted_searches_each_row_in_its_own_sequence(bool right)
    {
        var eng = new CpuEngine();
        var seq = new Tensor<float>(new[] { 2, 2, 4 }, new Vector<float>(new float[]
        {
            1, 3, 5, 7, 0, 0, 2, 2,
            -4, -1, 0, 9, 10, 20, 30, 40,
        }));
        var vals = new Tensor<float>(new[] { 2, 2, 3 }, new Vector<float>(new float[]
        {
            3, 0, 8, 0, 2, 3,
            -1, 9, -9, 25, 10, 41,
        }));
        var r = eng.TensorSearchSorted(seq, vals, right);
        Assert.Equal(new[] { 2, 2, 3 }, r.Shape.ToArray());
        var actual = r.ToArray();
        for (int row = 0; row < 4; row++)
        {
            var s1 = new Tensor<float>(new[] { 4 }, new Vector<float>(seq.ToArray().Skip(row * 4).Take(4).ToArray()));
            var v1 = new Tensor<float>(new[] { 3 }, new Vector<float>(vals.ToArray().Skip(row * 3).Take(3).ToArray()));
            Assert.Equal(eng.TensorSearchSorted(s1, v1, right).ToArray(), actual.Skip(row * 3).Take(3).ToArray());
        }
    }

    [Fact]
    public void Nd_searchsorted_rejects_mismatched_leading_dims()
    {
        var eng = new CpuEngine();
        Assert.Throws<ArgumentException>(() => eng.TensorSearchSorted(new Tensor<float>(new[] { 2, 4 }), new Tensor<float>(new[] { 3, 1 })));
        Assert.Throws<ArgumentException>(() => eng.TensorSearchSorted(new Tensor<float>(new[] { 2, 4 }), new Tensor<float>(new[] { 2 })));
    }

    private static void AssertClose(double[] e, double[] a)
    {
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++) Assert.True(Math.Abs(e[i] - a[i]) <= 1e-9 * (1 + Math.Abs(e[i])), $"element {i}: {a[i]} vs {e[i]}");
    }
}
