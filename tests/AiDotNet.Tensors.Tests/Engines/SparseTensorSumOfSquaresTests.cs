using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class SparseTensorSumOfSquaresTests
{
    private static SparseTensor<T> WithFormat<T>(SparseTensor<T> tensor, string format)
        => format switch
        {
            "coo" => tensor,
            "csr" => tensor.ToCsr(),
            "csc" => tensor.ToCsc(),
            "bsr" => tensor.ToBsr(2, 2),
            "bsc" => tensor.ToBsc(2, 2),
            _ => throw new ArgumentException(nameof(format)),
        };

    private static SparseTensor<T> OddColumnDuplicates<T>(string format, Func<double, T> fromDouble)
    {
        var rows = new List<int>();
        var columns = new List<int>();
        var values = new List<T>();
        for (int row = 0; row < 4; row++)
        {
            for (int column = 1; column < 8; column += 2)
            {
                rows.Add(row); columns.Add(column); values.Add(fromDouble(row * 4 + (column + 1) / 2));
                if (row == 0 && column == 1)
                {
                    rows.Add(row); columns.Add(column); values.Add(fromDouble(2));
                }
                if (row == 3 && column == 7)
                {
                    rows.Add(row); columns.Add(column); values.Add(fromDouble(-16));
                }
            }
        }

        // Preserve duplicate entries in compressed formats too, rather than coalescing in a conversion first.
        if (format == "csr")
            return SparseTensor<T>.FromCsr(4, 8, new[] { 0, 5, 9, 13, 18 }, columns.ToArray(), values.ToArray());
        if (format == "csc")
        {
            var order = Enumerable.Range(0, values.Count).OrderBy(i => columns[i]).ToArray();
            return SparseTensor<T>.FromCsc(4, 8, new[] { 0, 0, 5, 5, 9, 9, 13, 13, 18 },
                order.Select(i => rows[i]).ToArray(), order.Select(i => values[i]).ToArray());
        }
        return WithFormat(new SparseTensor<T>(4, 8, rows.ToArray(), columns.ToArray(), values.ToArray()), format);
    }

    private static T ReduceWithoutChangingSource<T>(SparseTensor<T> sparse)
    {
        var values = sparse.Values.ToArray();
        var rows = sparse.RowIndices.ToArray();
        var columns = sparse.ColumnIndices.ToArray();
        var rowPointers = sparse.RowPointers.ToArray();
        var columnPointers = sparse.ColumnPointers.ToArray();
        var shape = sparse.Shape.ToArray();
        var format = sparse.Format;
        int blockRows = sparse.BlockRowSize, blockColumns = sparse.BlockColSize;

        T result = new CpuEngine().TensorSumOfSquares(sparse);

        Assert.Equal(values, sparse.Values);
        Assert.Equal(rows, sparse.RowIndices);
        Assert.Equal(columns, sparse.ColumnIndices);
        Assert.Equal(rowPointers, sparse.RowPointers);
        Assert.Equal(columnPointers, sparse.ColumnPointers);
        Assert.Equal(shape, sparse.Shape.ToArray());
        Assert.Equal(format, sparse.Format);
        Assert.Equal(blockRows, sparse.BlockRowSize);
        Assert.Equal(blockColumns, sparse.BlockColSize);
        return result;
    }

    [Theory]
    [InlineData("coo")]
    [InlineData("csr")]
    [InlineData("csc")]
    [InlineData("bsr")]
    [InlineData("bsc")]
    public void Double_AllFormats_SquareCoalescedLogicalValues(string format)
        // 1^2 + ... + 16^2, with the first value changed from 1 to 3 and the last cancelled to zero.
        => Assert.Equal(1496.0 + 8.0 - 256.0, ReduceWithoutChangingSource(OddColumnDuplicates(format, value => value)));

    [Theory]
    [InlineData("coo")]
    [InlineData("csr")]
    [InlineData("csc")]
    [InlineData("bsr")]
    [InlineData("bsc")]
    public void Float_AllFormats_SquareCoalescedLogicalValues(string format)
        => Assert.Equal(1248f, ReduceWithoutChangingSource(OddColumnDuplicates(format, value => (float)value)));

    [Theory]
    [InlineData("coo")]
    [InlineData("csr")]
    [InlineData("csc")]
    [InlineData("bsr")]
    [InlineData("bsc")]
    public void Decimal_AllFormats_SquareCoalescedLogicalValues(string format)
        => Assert.Equal(1248m, ReduceWithoutChangingSource(OddColumnDuplicates(format, value => (decimal)value)));

    [Fact]
    public void Double_SixteenStoredValues_UseTheirLogicalSum()
    {
        var rows = Enumerable.Range(0, 16).Select(i => i / 4).ToArray();
        var columns = Enumerable.Range(0, 16).Select(i => 2 * (i % 4) + 1).ToArray();
        var values = Enumerable.Range(1, 16).Select(i => (double)i).ToArray();
        var sparse = new SparseTensor<double>(4, 8, rows, columns, values);

        Assert.Equal(1496.0, ReduceWithoutChangingSource(sparse));
    }

#if NET5_0_OR_GREATER
    [Theory]
    [InlineData("coo")]
    [InlineData("csr")]
    [InlineData("csc")]
    [InlineData("bsr")]
    [InlineData("bsc")]
    public void Half_AllFormats_SquareCoalescedLogicalValues(string format)
        => Assert.Equal((Half)1248, ReduceWithoutChangingSource(OddColumnDuplicates(format, value => (Half)value)));
#endif

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DuplicateBlocks_AreAddedBeforeSquaring(bool columnCompressed)
    {
        var values = new[] { 3.0, 4.0, 0.0, 0.0, -3.0, 1.0, 0.0, 0.0 };
        var sparse = columnCompressed
            ? SparseTensor<double>.FromBsc(4, 4, 2, 2, new[] { 0, 0, 2 }, new[] { 0, 0 }, values)
            : SparseTensor<double>.FromBsr(4, 4, 2, 2, new[] { 0, 2, 2 }, new[] { 1, 1 }, values);

        Assert.Equal(25.0, ReduceWithoutChangingSource(sparse));
    }

    [Theory]
    [InlineData("coo", false)]
    [InlineData("csr", false)]
    [InlineData("csc", false)]
    [InlineData("bsr", false)]
    [InlineData("bsc", false)]
    [InlineData("coo", true)]
    [InlineData("csr", true)]
    [InlineData("csc", true)]
    [InlineData("bsr", true)]
    [InlineData("bsc", true)]
    public void EmptyOrCancelledStorage_ReturnsZero(string format, bool cancelled)
    {
        var sparse = cancelled
            ? new SparseTensor<double>(4, 8, new[] { 0, 0 }, new[] { 1, 1 }, new[] { 3.0, -3.0 })
            : new SparseTensor<double>(4, 8, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<double>());

        Assert.Equal(0.0, ReduceWithoutChangingSource(WithFormat(sparse, format)));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void NonFiniteStoredValues_RetainTheirClassification(double value)
    {
        var sparse = new SparseTensor<double>(4, 8, new[] { 1 }, new[] { 3 }, new[] { value });
        double result = ReduceWithoutChangingSource(sparse);

        if (double.IsNaN(value)) Assert.True(double.IsNaN(result));
        else Assert.True(double.IsPositiveInfinity(result));
    }

    [Fact]
    public void OppositeInfiniteDuplicateValues_ProduceNaN()
    {
        var sparse = new SparseTensor<double>(4, 8, new[] { 1, 1 }, new[] { 3, 3 },
            new[] { double.PositiveInfinity, double.NegativeInfinity });

        Assert.True(double.IsNaN(ReduceWithoutChangingSource(sparse)));
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void LargeLogicalShape_DoesNotAllocateItsDensePayload()
    {
        var engine = new CpuEngine();
        engine.TensorSumOfSquares(new Tensor<double>(new[] { 3.0, 4.0 }, new[] { 2 }));
        var sparse = new SparseTensor<double>(1000, 1000, new[] { 0, 999 }, new[] { 999, 0 }, new[] { 3.0, 4.0 });
        long before = GC.GetAllocatedBytesForCurrentThread();

        double result = engine.TensorSumOfSquares(sparse);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(25.0, result);
        Assert.True(allocated < (long)sparse.Length * sizeof(double),
            $"A two-entry sparse reduction allocated {allocated} bytes, enough to materialize its dense payload.");
    }
#endif
}
