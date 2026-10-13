using System;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// The value-level contract of the Vector and Matrix readers and arithmetic that #1187 changed to hold their
/// instance alive while its storage is in use. That change added a liveness scope to each of these members, and
/// most of them had no test in this repository at all (Codecov: 71 changed lines uncovered), so a member that
/// returned the wrong thing would not have been noticed. The base-class versions are reached through a
/// <see cref="VectorBase{T}"/> or <see cref="MatrixBase{T}"/> reference, because the derived classes hide them.
/// </summary>
public class VectorMatrixReaderTests
{
    private static Vector<double> V(params double[] values) => new Vector<double>(values);

    private static Matrix<double> M(double[,] values) => new Matrix<double>(values);

    private static double[] Flat(MatrixBase<double> m)
    {
        var flat = new double[m.Rows * m.Columns];
        for (int r = 0; r < m.Rows; r++)
            for (int c = 0; c < m.Columns; c++) flat[r * m.Columns + c] = m[r, c];
        return flat;
    }

    // ---------------------------------------------------------------- Vector<T>

    [Fact]
    public void Vector_ElementwiseAndScalarArithmetic()
    {
        var a = V(2, 4, -6, 8);
        var b = V(1, 2, 3, -4);
        Assert.Equal(new double[] { 2, 2, -2, -2 }, a.ElementwiseDivide(b).ToArray());
        Assert.Equal(new double[] { 2, 8, -18, -32 }, a.ElementwiseMultiply(b).ToArray());
        Assert.Equal(new double[] { 3, 6, -3, 4 }, a.Add(b).ToArray());
        Assert.Equal(new double[] { 1, 2, -9, 12 }, a.Subtract(b).ToArray());
        Assert.Equal(new double[] { 5, 10, -15, 20 }, a.Multiply(2.5).ToArray());
        Assert.Equal(new double[] { 1, 2, -3, 4 }, a.Divide(2).ToArray());
        Assert.Equal(2 * 1 + 4 * 2 + -6 * 3 + 8 * -4, a.DotProduct(b));
        Assert.Equal(Math.Sqrt(4 + 16 + 36 + 64), a.Norm(), 12);
        // The operands are untouched.
        Assert.Equal(new double[] { 2, 4, -6, 8 }, a.ToArray());
        Assert.Equal(new double[] { 1, 2, 3, -4 }, b.ToArray());
    }

    [Fact]
    public void Vector_ProjectionsAndReshapes()
    {
        var a = V(5, -1, 7, 0, 3);
        Assert.Equal(new double[] { 5, 7, 3 }, a.Where(x => x > 0).ToArray());
        Assert.Equal(new[] { 10, -2, 14, 0, 6 }, a.Select(x => (int)(2 * x)).ToArray());
        Assert.Equal(new[] { "5", "-1", "7", "0", "3" }, a.Transform(x => x.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToArray());
        Assert.Equal(new double[] { 5, 0, 9, 3, 7 }, a.Transform((x, i) => x + i).ToArray());
        Assert.Equal(new double[] { -1, 7, 0 }, a.GetSubVector(1, 3).ToArray());
        Assert.Equal(new double[] { 5, -1, 0, 3 }, a.RemoveAt(2).ToArray());

        var transposed = a.Transpose();
        Assert.Equal(a.Length, transposed.Rows * transposed.Columns);
        Assert.Equal(a.ToArray(), Flat(transposed));

        byte[] serialized = a.Serialize();
        Assert.True(serialized.Length >= a.Length * sizeof(double), $"{serialized.Length} bytes for {a.Length} doubles");
    }

    // ---------------------------------------------------------------- VectorBase<T>

    [Fact]
    public void VectorBase_Members_ThroughABaseReference()
    {
        VectorBase<double> a = V(2, 4, -6, 8);
        VectorBase<double> b = V(1, 2, 3, -4);

        Assert.Equal(new double[] { 2, 4, -6, 8 }, a.ToArray());
        Assert.Equal(8.0, a.Sum());
        Assert.Equal(Math.Sqrt(120), a.L2Norm(), 12);
        Assert.Equal(new double[] { 4, -6 }, a.GetSubVector(1, 2).ToArray());
        Assert.Equal(new[] { 4, 8, -12, 16 }, a.Transform(x => (int)(2 * x)).ToArray());
        Assert.Equal(new double[] { 2, 5, -4, 11 }, a.Transform((x, i) => x + i).ToArray());

        Assert.Equal(new double[] { 3, 6, -3, 4 }, a.Add(b).ToArray());
        Assert.Equal(new double[] { 1, 2, -9, 12 }, a.Subtract(b).ToArray());
        Assert.Equal(new double[] { 6, 12, -18, 24 }, a.Multiply(3).ToArray());
        Assert.Equal(new double[] { 1, 2, -3, 4 }, a.Divide(2).ToArray());

        var destination = new double[4];
        a.Add(b, destination);
        Assert.Equal(new double[] { 3, 6, -3, 4 }, destination);
        a.Subtract(b, destination);
        Assert.Equal(new double[] { 1, 2, -9, 12 }, destination);
        a.Multiply(-1, destination);
        Assert.Equal(new double[] { -2, -4, 6, -8 }, destination);
        a.Divide(4, destination);
        Assert.Equal(new double[] { 0.5, 1, -1.5, 2 }, destination);

        // A clone is a separate vector: the in-place operations below change it and not the original.
        var clone = a.Clone();
        clone.AddInPlace(b);
        Assert.Equal(new double[] { 3, 6, -3, 4 }, clone.ToArray());
        clone.SubtractInPlace(b);
        Assert.Equal(new double[] { 2, 4, -6, 8 }, clone.ToArray());
        clone.MultiplyInPlace(10);
        Assert.Equal(new double[] { 20, 40, -60, 80 }, clone.ToArray());
        clone.DivideInPlace(20);
        Assert.Equal(new double[] { 1, 2, -3, 4 }, clone.ToArray());
        Assert.Equal(new double[] { 2, 4, -6, 8 }, a.ToArray());
    }

    // ---------------------------------------------------------------- Matrix<T>

    [Fact]
    public void Matrix_SegmentsAndArithmetic()
    {
        var m = M(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 } });
        Assert.Equal(new double[] { 5, 8 }, m.GetColumnSegment(1, 1, 2).ToArray());
        Assert.Equal(new double[] { 5, 6 }, m.GetRowSegment(1, 1, 2).ToArray());
        Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 10 }, m.ToRowVector().ToArray());

        Assert.Equal(new double[] { 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5 }, Flat(m.Divide(2)));
        var other = M(new double[,] { { 1, 2, 3 }, { 2, 5, 3 }, { 7, 4, 5 } });
        Assert.Equal(new double[] { 1, 1, 1, 2, 1, 2, 1, 2, 2 }, Flat(m.Divide(other)));

        var tensor = new Tensor<double>(new double[] { 10, 20, 30, 40, 50, 60, 70, 80, 90 }, new[] { 3, 3 });
        Assert.Equal(new double[] { 11, 22, 33, 44, 55, 66, 77, 88, 100 }, Flat(m.Add(tensor)));

        // The operand is untouched by the out-of-place forms, and changed by the in-place one.
        Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 10 }, Flat(m));
        m.DivideInPlace(0.5);
        Assert.Equal(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 20 }, Flat(m));
    }

    // ---------------------------------------------------------------- MatrixBase<T>

    [Fact]
    public void MatrixBase_Members_ThroughABaseReference()
    {
        MatrixBase<double> a = M(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        MatrixBase<double> b = M(new double[,] { { 6, 5 }, { 4, 3 }, { 2, 1 } });

        Assert.Equal(new double[] { 3, 4 }, a.GetRow(1).ToArray());
        Assert.Equal(new double[] { 2, 4, 6 }, a.GetColumn(1).ToArray());
        var slice = a.Slice(1, 2);
        Assert.Equal(2, slice.Rows);
        Assert.Equal(new double[] { 3, 4, 5, 6 }, Flat(slice));
        Assert.Equal(1 * 6 + 2 * 5 + 3 * 4 + 4 * 3 + 5 * 2 + 6 * 1, a.ElementWiseMultiplyAndSum(b));

        Assert.Equal(new double[] { 7, 7, 7, 7, 7, 7 }, Flat(a.Add(b)));
        Assert.Equal(new double[] { -5, -3, -1, 1, 3, 5 }, Flat(a.Subtract(b)));
        Assert.Equal(new double[] { 3, 6, 9, 12, 15, 18 }, Flat(a.Multiply(3)));

        var destination = new double[6];
        a.Add(b, destination);
        Assert.Equal(new double[] { 7, 7, 7, 7, 7, 7 }, destination);
        a.Subtract(b, destination);
        Assert.Equal(new double[] { -5, -3, -1, 1, 3, 5 }, destination);
        a.Multiply(-2, destination);
        Assert.Equal(new double[] { -2, -4, -6, -8, -10, -12 }, destination);

        // [3x2] . [2x3]
        MatrixBase<double> c = M(new double[,] { { 1, 0, 2 }, { 0, 1, 3 } });
        var product = a.Multiply(c);
        Assert.Equal(3, product.Rows);
        Assert.Equal(3, product.Columns);
        Assert.Equal(new double[] { 1, 2, 8, 3, 4, 18, 5, 6, 28 }, Flat(product));
        Assert.Equal(new double[] { 5, 11, 17 }, a.Multiply(new Vector<double>(new double[] { 1, 2 })).ToArray());

        var transposed = a.Transpose();
        Assert.Equal(2, transposed.Rows);
        Assert.Equal(3, transposed.Columns);
        Assert.Equal(new double[] { 1, 3, 5, 2, 4, 6 }, Flat(transposed));

        // A clone is a separate matrix: everything in place happens to it and not to the original.
        var clone = a.Clone();
        clone.AddInPlace(b);
        Assert.Equal(new double[] { 7, 7, 7, 7, 7, 7 }, Flat(clone));
        clone.SubtractInPlace(b);
        Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6 }, Flat(clone));
        clone.MultiplyInPlace(2);
        Assert.Equal(new double[] { 2, 4, 6, 8, 10, 12 }, Flat(clone));
        clone.SetRow(0, new Vector<double>(new double[] { -1, -2 }));
        clone.SetColumn(1, new Vector<double>(new double[] { 9, 8, 7 }));
        Assert.Equal(new double[] { -1, 9, 6, 8, 10, 7 }, Flat(clone));
        Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6 }, Flat(a));

        MatrixBase<double> square = M(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        square.TransposeInPlace();
        Assert.Equal(new double[] { 1, 4, 7, 2, 5, 8, 3, 6, 9 }, Flat(square));
    }
}
