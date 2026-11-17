using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class TensorFinitePredicateTests
{
    private static readonly bool[] FloatingFinite = { false, true, true, true, true, true, false, false };
    private static readonly bool[] FloatingInfinite = { true, false, false, false, false, false, true, false };

    private static void AssertPredicates<T>(T[] values, bool[] expectedFinite, bool[] expectedInfinite, bool transpose)
    {
        var engine = new CpuEngine();
        var tensor = new Tensor<T>(values, new[] { 2, 4 });
        if (transpose)
        {
            tensor = tensor.Transpose(new[] { 1, 0 });
            Assert.False(tensor.IsContiguous);
        }

        var finite = engine.TensorIsFinite(tensor);
        var infinite = engine.TensorIsInf(tensor);

        Assert.Equal(tensor.Shape.ToArray(), finite.Shape.ToArray());
        Assert.Equal(tensor.Shape.ToArray(), infinite.Shape.ToArray());
        for (int i = 0; i < values.Length; i++)
        {
            int sourceIndex = transpose ? (i % 2) * 4 + i / 2 : i;
            Assert.Equal(expectedFinite[sourceIndex], (bool)finite[i]);
            Assert.Equal(expectedInfinite[sourceIndex], (bool)infinite[i]);
        }
    }

#if NET5_0_OR_GREATER
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Half_DoesNotRequireIConvertible(bool transpose)
    {
        AssertPredicates(new[]
        {
            Half.NegativeInfinity, Half.MinValue, (Half)(-0.0f), Half.Epsilon,
            (Half)0, Half.MaxValue, Half.PositiveInfinity, Half.NaN,
        }, FloatingFinite, FloatingInfinite, transpose);
    }
#endif

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Double_MaxValueRemainsFinite(bool transpose)
    {
        AssertPredicates(new[]
        {
            double.NegativeInfinity, double.MinValue, -0.0, double.Epsilon,
            0.0, double.MaxValue, double.PositiveInfinity, double.NaN,
        }, FloatingFinite, FloatingInfinite, transpose);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Float_MaxValueRemainsFinite(bool transpose)
    {
        AssertPredicates(new[]
        {
            float.NegativeInfinity, float.MinValue, -0.0f, float.Epsilon,
            0.0f, float.MaxValue, float.PositiveInfinity, float.NaN,
        }, FloatingFinite, FloatingInfinite, transpose);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decimal_ValuesAreFinite(bool transpose)
    {
        AssertPredicates(new[] { decimal.MinValue, -1m, 0m, 1m, 0.25m, -0.25m, 1000m, decimal.MaxValue },
            Enumerable.Repeat(true, 8).ToArray(), new bool[8], transpose);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Int64_ValuesAreFinite(bool transpose)
    {
        AssertPredicates(new[] { long.MinValue, -1L, 0L, 1L, 2L, 1L << 54, -(1L << 54), long.MaxValue },
            Enumerable.Repeat(true, 8).ToArray(), new bool[8], transpose);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BFloat16_ClassifiesItsExactEncodings(bool transpose)
    {
        AssertPredicates(new[]
        {
            BFloat16.NegativeInfinity, BFloat16.MinValue, BFloat16.FromRawBits(0x8000), BFloat16.FromRawBits(0x0001),
            BFloat16.Zero, BFloat16.MaxValue, BFloat16.PositiveInfinity, BFloat16.NaN,
        }, FloatingFinite, FloatingInfinite, transpose);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Float8E4M3_ExponentMaxValuesCanBeFiniteAndNeverInfinite(bool transpose)
    {
        AssertPredicates(new[]
        {
            Float8E4M3.NaN, Float8E4M3.MinFinite, Float8E4M3.FromRawValue(0x80), Float8E4M3.FromRawValue(0x01),
            Float8E4M3.Zero, Float8E4M3.MaxFinite, Float8E4M3.FromRawValue(0x78), Float8E4M3.FromRawValue(0xff),
        }, new[] { false, true, true, true, true, true, true, false }, new bool[8], transpose);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Float8E5M2_DistinguishesInfinityFromNaN(bool transpose)
    {
        AssertPredicates(new[]
        {
            Float8E5M2.NegativeInfinity, Float8E5M2.MinFinite, Float8E5M2.FromFloat(-0.0f), Float8E5M2.FromFloat(1.0f),
            Float8E5M2.Zero, Float8E5M2.MaxFinite, Float8E5M2.PositiveInfinity, Float8E5M2.NaN,
        }, FloatingFinite, FloatingInfinite, transpose);
    }

    [Fact]
    public void AlgebraicElements_AreExplicitlyUnsupportedByRealPredicates()
    {
        var tensor = new Tensor<Complex<double>>(new[] { new Complex<double>(1.0, 2.0) }, new[] { 1 });
        var engine = new CpuEngine();

        Assert.Throws<NotSupportedException>(() => engine.TensorIsFinite(tensor));
        Assert.Throws<NotSupportedException>(() => engine.TensorIsInf(tensor));
        var sparse = new SparseTensor<Complex<double>>(2, 2, new[] { 0 }, new[] { 1 },
            new[] { new Complex<double>(1.0, 2.0) });
        Assert.Throws<NotSupportedException>(() => engine.TensorIsFinite(sparse));
        Assert.Throws<NotSupportedException>(() => engine.TensorIsInf(sparse));
    }

    [Theory]
    [InlineData("coo")]
    [InlineData("csr")]
    [InlineData("csc")]
    [InlineData("bsr")]
    [InlineData("bsc")]
    public void SparseMasks_ClassifyCoalescedCoordinatesAndImplicitZeros(string format)
    {
        var coo = new SparseTensor<double>(4, 4,
            new[] { 0, 0, 1, 1, 1, 2, 3, 3 }, new[] { 1, 1, 0, 0, 3, 2, 0, 3 },
            new[] { double.PositiveInfinity, double.NegativeInfinity, 3.0, -3.0,
                double.PositiveInfinity, double.NaN, double.MaxValue, double.NegativeInfinity });
        var sparse = format switch
        {
            "coo" => coo,
            "csr" => coo.ToCsr(),
            "csc" => coo.ToCsc(),
            "bsr" => coo.ToBsr(2, 2),
            "bsc" => coo.ToBsc(2, 2),
            _ => throw new ArgumentException(nameof(format)),
        };
        var originalValues = sparse.Values.ToArray();
        var originalRows = sparse.RowIndices.ToArray();
        var originalColumns = sparse.ColumnIndices.ToArray();
        var engine = new CpuEngine();

        var finite = engine.TensorIsFinite(sparse);
        var infinite = engine.TensorIsInf(sparse);

        var expectedFinite = Enumerable.Repeat(true, 16).ToArray();
        foreach (int index in new[] { 1, 7, 10, 15 }) expectedFinite[index] = false;
        var expectedInfinite = new bool[16];
        expectedInfinite[7] = expectedInfinite[15] = true;
        Assert.Equal(new[] { 4, 4 }, finite.Shape.ToArray());
        Assert.Equal(new[] { 4, 4 }, infinite.Shape.ToArray());
        Assert.Equal(expectedFinite, finite.ToArray().Select(bit => (bool)bit).ToArray());
        Assert.Equal(expectedInfinite, infinite.ToArray().Select(bit => (bool)bit).ToArray());
        Assert.Equal(originalValues, sparse.Values);
        Assert.Equal(originalRows, sparse.RowIndices);
        Assert.Equal(originalColumns, sparse.ColumnIndices);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SparseEmptyOrCancelledValues_LeaveEveryLogicalZeroFinite(bool cancelled)
    {
        var sparse = cancelled
            ? new SparseTensor<double>(2, 3, new[] { 0, 0 }, new[] { 1, 1 }, new[] { 3.0, -3.0 })
            : new SparseTensor<double>(2, 3, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<double>());
        var engine = new CpuEngine();

        Assert.All(engine.TensorIsFinite(sparse).ToArray(), bit => Assert.True((bool)bit));
        Assert.All(engine.TensorIsInf(sparse).ToArray(), bit => Assert.False((bool)bit));
    }

    [Fact]
    public void SparseNonFloatingTypes_SkipCoalescingAndDoNotOverflowOnDuplicateMaxValue()
    {
        var engine = new CpuEngine();
        var decSparse = new SparseTensor<decimal>(
            2, 2,
            new[] { 0, 0, 1 },
            new[] { 1, 1, 0 },
            new[] { decimal.MaxValue, decimal.MaxValue, decimal.MinValue });
        Assert.All(engine.TensorIsFinite(decSparse).ToArray(), bit => Assert.True((bool)bit));
        Assert.All(engine.TensorIsInf(decSparse).ToArray(), bit => Assert.False((bool)bit));

        var intSparse = new SparseTensor<int>(
            2, 2,
            new[] { 1, 1 },
            new[] { 1, 1 },
            new[] { int.MaxValue, int.MaxValue });
        Assert.All(engine.TensorIsFinite(intSparse).ToArray(), bit => Assert.True((bool)bit));
        Assert.All(engine.TensorIsInf(intSparse).ToArray(), bit => Assert.False((bool)bit));
    }
}
