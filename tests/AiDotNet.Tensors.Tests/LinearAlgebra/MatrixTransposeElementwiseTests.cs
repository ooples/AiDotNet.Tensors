using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Exact-value coverage for <see cref="MatrixBase{T}.Transpose"/> (AVX 4×4 / 8×8 register-transpose
/// kernel inside 64-element tiles, parallel over tile rows) and for the chunked-parallel allocating
/// elementwise operations. Shapes straddle every edge: the register-block size, the tile size, the
/// parallel threshold (65,536 elements) and the ArrayPool rent threshold (262,144 elements, where
/// the result's backing array is longer than rows * cols).
/// </summary>
public class MatrixTransposeElementwiseTests
{
    public static TheoryData<int, int> Shapes => new()
    {
        { 1, 1 }, { 1, 9 }, { 9, 1 }, { 3, 5 }, { 4, 4 }, { 7, 9 }, { 8, 8 }, { 17, 13 },
        { 63, 65 }, { 64, 64 }, { 100, 100 }, { 129, 257 }, { 255, 257 }, { 300, 300 },
        { 513, 517 }, { 1000, 1000 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Transpose_Double_MatchesReference(int rows, int cols)
    {
        var m = Filled<double>(rows, cols, (i, j) => i * 10007.0 + j);
        var t = m.Transpose();

        Assert.Equal(cols, t.Rows);
        Assert.Equal(rows, t.Columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                Assert.Equal(m[i, j], t[j, i]);
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Transpose_Float_MatchesReference(int rows, int cols)
    {
        var m = Filled<float>(rows, cols, (i, j) => i * 1009f + j);
        var t = m.Transpose();

        Assert.Equal(cols, t.Rows);
        Assert.Equal(rows, t.Columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                Assert.Equal(m[i, j], t[j, i]);
    }

    [Theory]
    [InlineData(3, 5)]
    [InlineData(513, 517)]
    public void Transpose_Int_UsesGenericPathAndMatchesReference(int rows, int cols)
    {
        var m = Filled<int>(rows, cols, (i, j) => i * 1000 + j);
        var t = m.Transpose();

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                Assert.Equal(m[i, j], t[j, i]);
    }

    [Theory]
    [InlineData(255, 257)]
    [InlineData(513, 517)]
    public void Transpose_Twice_RoundTrips(int rows, int cols)
    {
        var m = Filled<double>(rows, cols, (i, j) => Math.Sin(i * 0.37 + j * 1.91));
        var back = m.Transpose().Transpose();

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                Assert.Equal(m[i, j], back[i, j]);
    }

    [Theory]
    [InlineData(100, 100)]   // below the parallel threshold
    [InlineData(255, 257)]   // just under it
    [InlineData(300, 300)]   // parallel, two chunks
    [InlineData(513, 517)]   // parallel, pooled result
    [InlineData(1000, 1000)] // parallel, 128K-element chunks
    public void AllocatingElementwise_Double_MatchesScalarReference(int rows, int cols)
    {
        var a = Filled<double>(rows, cols, (i, j) => Math.Sin(i * 0.11 + j * 0.07));
        var b = Filled<double>(rows, cols, (i, j) => Math.Cos(i * 0.05 - j * 0.13));

        var sum = a.Add(b);
        var diff = a.Subtract(b);
        var scaled = a.Multiply(2.5);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                Assert.Equal(a[i, j] + b[i, j], sum[i, j]);
                Assert.Equal(a[i, j] - b[i, j], diff[i, j]);
                Assert.Equal(a[i, j] * 2.5, scaled[i, j]);
            }
    }

    [Theory]
    [InlineData(300, 300)]
    [InlineData(513, 517)]
    public void AllocatingElementwise_Float_MatchesScalarReference(int rows, int cols)
    {
        var a = Filled<float>(rows, cols, (i, j) => (float)Math.Sin(i * 0.11 + j * 0.07));
        var b = Filled<float>(rows, cols, (i, j) => (float)Math.Cos(i * 0.05 - j * 0.13));

        var sum = a.Add(b);
        var diff = a.Subtract(b);
        var scaled = a.Multiply(2.5f);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                Assert.Equal(a[i, j] + b[i, j], sum[i, j]);
                Assert.Equal(a[i, j] - b[i, j], diff[i, j]);
                Assert.Equal(a[i, j] * 2.5f, scaled[i, j]);
            }
    }

    [Fact]
    public void AllocatingElementwise_DoesNotMutateOperands()
    {
        var a = Filled<double>(300, 300, (i, j) => i + j * 0.5);
        var b = Filled<double>(300, 300, (i, j) => i * 0.25 - j);
        var aCopy = a.Clone();
        var bCopy = b.Clone();

        _ = a.Add(b);
        _ = a.Subtract(b);
        _ = a.Multiply(3.0);

        for (int i = 0; i < 300; i++)
            for (int j = 0; j < 300; j++)
            {
                Assert.Equal(aCopy[i, j], a[i, j]);
                Assert.Equal(bCopy[i, j], b[i, j]);
            }
    }

    private static Matrix<T> Filled<T>(int rows, int cols, Func<int, int, T> value)
    {
        var data = new T[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = value(i, j);
        return new Matrix<T>(data);
    }
}
