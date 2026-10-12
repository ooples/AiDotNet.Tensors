using System;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Value checks for Vector / Matrix / Tensor members that read their own storage (the members #1187 keeps <c>this</c>
/// alive in) and that no other test in this repository calls. Each result is compared with a plain loop over the inputs.
/// </summary>
public class StorageReaderMemberTests
{
    private static double[] Seq(int n, double scale = 1.0, double shift = 0.0) =>
        Enumerable.Range(0, n).Select(i => (i * 0.37 + 1.0) * scale + shift).ToArray();

    private static Matrix<double> Mat(int rows, int cols, double shift = 0.0)
    {
        var m = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = i * 10.0 + j + shift;
        return m;
    }

    // ---- Vector ----

    [Fact]
    public void Vector_InPlaceArithmetic_MatchesElementwiseLoop()
    {
        double[] a = Seq(37), b = Seq(37, 0.5, 2.0);
        var v = new Vector<double>(a);
        v.AddInPlace(new Vector<double>(b));
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i] + b[i], v[i], 12);

        v = new Vector<double>(a);
        v.SubtractInPlace(new Vector<double>(b));
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i] - b[i], v[i], 12);

        v = new Vector<double>(a);
        v.MultiplyInPlace(3.0);
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i] * 3.0, v[i], 12);

        v = new Vector<double>(a);
        v.DivideInPlace(4.0);
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i] / 4.0, v[i], 12);
    }

    [Fact]
    public void Vector_ElementwiseOps_SubVector_AndNorms_MatchLoops()
    {
        double[] a = Seq(41), b = Seq(41, 0.25, 1.0);
        var va = new Vector<double>(a);
        var vb = new Vector<double>(b);

        var prod = va.ElementwiseMultiply(vb);
        var quot = va.ElementwiseDivide(vb);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(a[i] * b[i], prod[i], 12);
            Assert.Equal(a[i] / b[i], quot[i], 12);
        }

        var sub = va.GetSubVector(5, 9);
        Assert.Equal(9, sub.Length);
        for (int i = 0; i < 9; i++) Assert.Equal(a[5 + i], sub[i]);

        double l2 = Math.Sqrt(a.Sum(x => x * x));
        Assert.Equal(l2, va.Norm(), 10);
        Assert.Equal(l2, va.L2Norm(), 10);
    }

    [Fact]
    public void Vector_RawAccessors_ExposeTheSameElements()
    {
        double[] a = Seq(16);
        var v = new Vector<double>(a);

        Assert.Equal(a, v.AsMemoryUnmarked().ToArray());
        Assert.Equal(a, v.AsWritableMemoryUnmarked().ToArray());
        Assert.Equal(a, v.GetDataArrayUnmarked().Take(a.Length).ToArray());
        Assert.True(v.TryGetBackingArrayIdentity(out var array, out int offset));
        Assert.NotNull(array);
        Assert.Equal(a, array!.Skip(offset).Take(a.Length).ToArray());

        var writable = v.AsWritableMemory();
        writable.Span[3] = -1.0;
        Assert.Equal(-1.0, v[3]);
    }

    // ---- Matrix ----

    [Fact]
    public void Matrix_InPlaceArithmetic_MatchesElementwiseLoop()
    {
        var a = Mat(5, 7);
        var b = Mat(5, 7, 0.5);

        var m = Mat(5, 7);
        m.AddInPlace(b);
        for (int i = 0; i < 5; i++) for (int j = 0; j < 7; j++) Assert.Equal(a[i, j] + b[i, j], m[i, j], 12);

        m = Mat(5, 7);
        m.SubtractInPlace(b);
        for (int i = 0; i < 5; i++) for (int j = 0; j < 7; j++) Assert.Equal(a[i, j] - b[i, j], m[i, j], 12);

        m = Mat(5, 7);
        m.MultiplyInPlace(2.5);
        for (int i = 0; i < 5; i++) for (int j = 0; j < 7; j++) Assert.Equal(a[i, j] * 2.5, m[i, j], 12);

        m = Mat(5, 7);
        m.DivideInPlace(2.0);
        for (int i = 0; i < 5; i++) for (int j = 0; j < 7; j++) Assert.Equal(a[i, j] / 2.0, m[i, j], 12);

        double expected = 0;
        for (int i = 0; i < 5; i++) for (int j = 0; j < 7; j++) expected += a[i, j] * b[i, j];
        Assert.Equal(expected, a.ElementWiseMultiplyAndSum(b), 8);
    }

    [Fact]
    public void Matrix_RowAndColumnAccessors_MatchIndexer()
    {
        var m = Mat(6, 4);

        var row = m.GetRow(2);
        var col = m.GetColumn(3);
        Assert.Equal(Enumerable.Range(0, 4).Select(j => m[2, j]), row.ToArray());
        Assert.Equal(Enumerable.Range(0, 6).Select(i => m[i, 3]), col.ToArray());

        var colSeg = m.GetColumnSegment(1, 2, 3);
        var rowSeg = m.GetRowSegment(4, 1, 2);
        Assert.Equal(new[] { m[2, 1], m[3, 1], m[4, 1] }, colSeg.ToArray());
        Assert.Equal(new[] { m[4, 1], m[4, 2] }, rowSeg.ToArray());

        Assert.Equal(row.ToArray(), m.GetRowReadOnlySpan(2).ToArray());
        m.GetRowSpan(5)[0] = -7.0;
        Assert.Equal(-7.0, m[5, 0]);

        m.SetRow(0, new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 }));
        m.SetColumn(1, new Vector<double>(new[] { 9.0, 8.0, 7.0, 6.0, 5.0, 4.0 }));
        Assert.Equal(new[] { 1.0, 9.0, 3.0, 4.0 }, m.GetRow(0).ToArray());
        Assert.Equal(new[] { 9.0, 8.0, 7.0, 6.0, 5.0, 4.0 }, m.GetColumn(1).ToArray());

        var single = Mat(1, 5);
        Assert.Equal(Enumerable.Range(0, 5).Select(j => single[0, j]), single.ToRowVector().ToArray());
    }

    [Theory]
    [InlineData(7)]   // element-swap path
    [InlineData(80)]  // n*n >= 4096: cache-blocked path
    public void Matrix_TransposeInPlace_MatchesTranspose(int n)
    {
        var m = Mat(n, n);
        var original = Mat(n, n);
        m.TransposeInPlace();
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Assert.Equal(original[j, i], m[i, j]);
    }

    [Fact]
    public void Matrix_RawAccessors_ExposeTheSameElements()
    {
        var m = Mat(3, 4);
        Assert.True(m.TryGetBackingArray(out var array));
        Assert.Equal(m[1, 2], array[1 * 4 + 2]);
        Assert.Equal(m[2, 3], m.GetDataArrayUnmarked()[2 * 4 + 3]);
        m.AsWritableMemory().Span[0] = 42.0;
        Assert.Equal(42.0, m[0, 0]);
    }

    // ---- Tensor ----

    [Fact]
    public void Tensor_InPlaceArithmetic_MatchesElementwiseLoop()
    {
        double[] a = Seq(24), b = Seq(24, 0.5, 1.0);
        int[] shape = { 2, 3, 4 };

        var t = new Tensor<double>((double[])a.Clone(), shape);
        t.AddInPlace(new Tensor<double>(b, shape));
        Assert.Equal(a.Zip(b, (x, y) => x + y), t.ToArray());

        t = new Tensor<double>((double[])a.Clone(), shape);
        t.SubtractInPlace(new Tensor<double>(b, shape));
        Assert.Equal(a.Zip(b, (x, y) => x - y), t.ToArray());

        t = new Tensor<double>((double[])a.Clone(), shape);
        t.DivideInPlace(2.0);
        Assert.Equal(a.Select(x => x / 2.0), t.ToArray());
    }

    [Fact]
    public void Tensor_RowsSlicesAndAxisReductions_MatchLoops()
    {
        double[] a = Seq(24);
        var t = new Tensor<double>((double[])a.Clone(), new[] { 2, 3, 4 });

        Assert.Equal(a.Skip(12).Take(12), t.GetRow(1).ToArray());
        Assert.Equal(a.Skip(5).Take(6), t.GetSlice(5, 6).ToArray());
        Assert.Equal(a.Skip(12).Take(12), t.GetSlice(1).ToArray());

        // Reductions over the last axis of [2,3,4] -> [2,3]. (Other axes: #1211.)
        var max = t.MaxOverAxis(2);
        var sum = t.SumOverAxisDirect(2);
        var mean = t.MeanOverAxisDirect(2);
        Assert.Equal(new[] { 2, 3 }, max.Shape.ToArray());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
            {
                double[] run = Enumerable.Range(0, 4).Select(k => a[i * 12 + j * 4 + k]).ToArray();
                Assert.Equal(run.Max(), max[i, j], 12);
                Assert.Equal(run.Sum(), sum[i, j], 12);
                Assert.Equal(run.Average(), mean[i, j], 12);
            }
    }

    [Fact]
    public void Tensor_SetSliceAndSetVectorAlongAxis_WriteTheExpectedElements()
    {
        var t = new Tensor<double>(new double[24], new[] { 2, 3, 4 });

        t.SetSlice(1, new Tensor<double>(Seq(12), new[] { 3, 4 }));
        Assert.Equal(Seq(12), t.ToArray().Skip(12));

        t.SetSlice(2, new Vector<double>(new[] { 5.0, 6.0 }));
        Assert.Equal(5.0, t.ToArray()[2]);
        Assert.Equal(6.0, t.ToArray()[3]);

        // dimension 2, index 3: the [2,3] plane at k = 3
        t.SetSlice(2, 3, new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, new[] { 2, 3 }));
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(i * 3 + j + 1.0, t[i, j, 3]);

        // the vector along axis 1 at (i = 0, k = 1)
        t.SetVectorAlongAxis(new Vector<double>(new[] { -1.0, -2.0, -3.0 }), 1, 0, 1);
        for (int j = 0; j < 3; j++) Assert.Equal(-(j + 1.0), t[0, j, 1]);
    }

    [Fact]
    public void Tensor_TryGetContiguousSpan_ExposesTheElements()
    {
        double[] a = Seq(12);
        var t = new Tensor<double>((double[])a.Clone(), new[] { 3, 4 });
        Assert.True(t.TryGetContiguousSpan(out var span));
        Assert.Equal(a, span.ToArray());
    }
}
