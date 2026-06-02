// Copyright (c) AiDotNet. All rights reserved.
// Pins the allocation-free contiguous fast path added to Tensor<T>.Sum(int[] axes)
// (Tensor.cs). Profiling found the old SumRecursive — a `new int[]` heap alloc +
// multi-dim strided indexer + axes.Contains() scan PER ELEMENT — was ~40% of a
// ResNet CPU train step (every conv bias / BatchNorm / broadcast gradient reduces
// [B,C,H,W] over axes). The fast path walks the flat row-major backing once with a
// reused odometer. These tests assert it is bit-identical to an independent scalar
// reference across shapes / axis combinations / dtypes, AND that the non-contiguous
// fallback (SumRecursive) still agrees (transposed view).

using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class TensorSumAxesFastPathTests
{
    public static TheoryData<int[], int[]> Cases => new()
    {
        { new[] { 1, 64, 16, 16 }, new[] { 0, 2, 3 } },  // conv bias-grad shape (batch+spatial)
        { new[] { 2, 8, 5, 7 },    new[] { 0, 2, 3 } },  // batch>1 bias-grad
        { new[] { 4, 6, 9 },       new[] { 1 } },        // single interior axis
        { new[] { 4, 6, 9 },       new[] { 0 } },        // single leading axis
        { new[] { 4, 6, 9 },       new[] { 2 } },        // single trailing axis
        { new[] { 3, 5, 7, 2 },    new[] { 1, 3 } },     // non-adjacent axes
        { new[] { 5, 4 },          new[] { 0 } },        // 2D
        { new[] { 5, 4 },          new[] { 1 } },        // 2D other axis
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void SumAxes_Double_MatchesScalarReference(int[] shape, int[] axes)
        => AssertMatchesReference<double>(shape, axes, seed: 7,
            gen: rng => rng.NextDouble() - 0.5, tol: 1e-12);

    [Theory]
    [MemberData(nameof(Cases))]
    public void SumAxes_Float_MatchesScalarReference(int[] shape, int[] axes)
        => AssertMatchesReference<float>(shape, axes, seed: 9,
            gen: rng => (float)(rng.NextDouble() - 0.5), tol: 1e-3);

    private static void AssertMatchesReference<T>(int[] shape, int[] axes, int seed,
        Func<Random, T> gen, double tol)
    {
        var rng = new Random(seed);
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = gen(rng);

        var actual = t.Sum(axes);

        // Independent reference: brute-force reduce in row-major order into a
        // dense [non-reduced dims] buffer using the public indexer + odometer.
        int rank = shape.Length;
        var reduce = new bool[rank];
        foreach (int a in axes) reduce[a] = true;
        var outShape = new System.Collections.Generic.List<int>();
        for (int d = 0; d < rank; d++) if (!reduce[d]) outShape.Add(shape[d]);
        int outLen = 1; foreach (int s in outShape) outLen *= s;
        var outStrides = new int[outShape.Count];
        int acc = 1;
        for (int i = outShape.Count - 1; i >= 0; i--) { outStrides[i] = acc; acc *= outShape[i]; }

        var expected = new double[outLen];
        var coord = new int[rank];
        int total = t.Length;
        for (int flat = 0; flat < total; flat++)
        {
            // decode flat -> coord (row-major over shape)
            int rem = flat;
            for (int d = rank - 1; d >= 0; d--) { coord[d] = rem % shape[d]; rem /= shape[d]; }
            int outFlat = 0, oi = 0;
            for (int d = 0; d < rank; d++) if (!reduce[d]) outFlat += coord[d] * outStrides[oi++];
            expected[outFlat] += Convert.ToDouble(t[coord]);
        }

        Assert.Equal(outLen, actual.Length);
        for (int i = 0; i < outLen; i++)
            Assert.True(Math.Abs(Convert.ToDouble(actual[i]) - expected[i]) <= tol,
                $"[{i}] actual={Convert.ToDouble(actual[i])} expected={expected[i]}");
    }

    [Fact]
    public void SumAxes_NonContiguousView_MatchesReference()
    {
        // Transposed view is non-contiguous → exercises the SumRecursive fallback.
        var rng = new Random(13);
        var t = new Tensor<double>(new[] { 3, 4, 5 });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() - 0.5;

        var view = t.Transpose(new[] { 2, 0, 1 }); // shape [5,3,4], non-contiguous
        Assert.False(view.IsContiguous);

        var actual = view.Sum(new[] { 0, 2 }); // reduce over axes 0 and 2 → [3]

        var expected = new double[3];
        for (int a = 0; a < 5; a++)
            for (int b = 0; b < 3; b++)
                for (int c = 0; c < 4; c++)
                    expected[b] += view[a, b, c];

        Assert.Equal(3, actual.Length);
        for (int i = 0; i < 3; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-12,
                $"[{i}] actual={actual[i]} expected={expected[i]}");
    }
}
