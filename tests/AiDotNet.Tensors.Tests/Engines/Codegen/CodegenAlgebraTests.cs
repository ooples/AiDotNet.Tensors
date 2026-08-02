// Copyright (c) AiDotNet. All rights reserved.
// Complex and quaternion arithmetic in the code generator.
//
// A wrong sign in the multiplication table produces a kernel that assembles, runs, and
// returns values of the right magnitude that are simply not the product -- a quaternion
// rotation that is subtly not a rotation. The table therefore is NOT checked against itself
// or against a second copy of the same formula. It is checked against the defining relations
// (i^2 = j^2 = k^2 = ijk = -1, ij = k) and against norm multiplicativity |ab| = |a||b|, both
// of which fail loudly on any sign error.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenAlgebraTests
{
    private static double[] Multiply(CodegenAlgebra algebra, double[] a, double[] b)
    {
        var result = new double[algebra.Components()];
        algebra.Multiply(a, b, result);
        return result;
    }

    // ---- The tables, against their defining relations -----------------------------------

    /// <summary>i² = -1, the relation that defines the complex numbers.</summary>
    [Fact]
    public void Complex_ISquaredIsMinusOne()
    {
        double[] i = { 0, 1 };
        Assert.Equal(new double[] { -1, 0 }, Multiply(CodegenAlgebra.Complex, i, i));
    }

    /// <summary>The complex product, against the formula written out separately.</summary>
    [Theory]
    [InlineData(1, 2, 3, 4)]
    [InlineData(-1, 0.5, 2, -3)]
    [InlineData(0, 0, 7, -7)]
    public void Complex_MatchesTheClosedForm(double ar, double ai, double br, double bi)
    {
        double[] got = Multiply(CodegenAlgebra.Complex, new[] { ar, ai }, new[] { br, bi });

        Assert.Equal(ar * br - ai * bi, got[0], 12);
        Assert.Equal(ar * bi + ai * br, got[1], 12);
    }

    /// <summary>i² = j² = k² = -1.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Quaternion_EachImaginaryUnitSquaresToMinusOne(int unit)
    {
        var u = new double[4];
        u[unit] = 1;

        double[] got = Multiply(CodegenAlgebra.Quaternion, u, u);
        Assert.Equal(new double[] { -1, 0, 0, 0 }, got);
    }

    /// <summary>ij = k, jk = i, ki = j -- and each reversal gives the negative.</summary>
    [Theory]
    [InlineData(1, 2, 3)]        // ij = k
    [InlineData(2, 3, 1)]        // jk = i
    [InlineData(3, 1, 2)]        // ki = j
    public void Quaternion_CyclicRelationsHold(int left, int right, int result)
    {
        var a = new double[4]; a[left] = 1;
        var b = new double[4]; b[right] = 1;

        var expected = new double[4]; expected[result] = 1;
        Assert.Equal(expected, Multiply(CodegenAlgebra.Quaternion, a, b));

        // NOT COMMUTATIVE. If the table were symmetric this would silently pass everywhere
        // else and only show up as a rotation composed in the wrong order.
        var reversed = new double[4]; reversed[result] = -1;
        Assert.Equal(reversed, Multiply(CodegenAlgebra.Quaternion, b, a));
    }

    /// <summary>ijk = -1, the relation that pins the whole table down.</summary>
    [Fact]
    public void Quaternion_IjkIsMinusOne()
    {
        double[] i = { 0, 1, 0, 0 }, j = { 0, 0, 1, 0 }, k = { 0, 0, 0, 1 };

        double[] ij = Multiply(CodegenAlgebra.Quaternion, i, j);
        double[] ijk = Multiply(CodegenAlgebra.Quaternion, ij, k);

        Assert.Equal(new double[] { -1, 0, 0, 0 }, ijk);
    }

    /// <summary>
    /// |ab| = |a||b|. An independent check that catches ANY single sign error, including
    /// ones the basis relations above would miss.
    /// </summary>
    [Theory]
    [InlineData(1, 2, 3, 4, 5, -6, 7, 8)]
    [InlineData(0.5, -1.5, 2.25, 0, -3, 1, 0, 4)]
    [InlineData(1, 1, 1, 1, 1, -1, 1, -1)]
    public void Quaternion_NormIsMultiplicative(
        double a0, double a1, double a2, double a3,
        double b0, double b1, double b2, double b3)
    {
        var a = new[] { a0, a1, a2, a3 };
        var b = new[] { b0, b1, b2, b3 };
        double[] product = Multiply(CodegenAlgebra.Quaternion, a, b);

        Assert.Equal(Norm(a) * Norm(b), Norm(product), 9);
    }

    /// <summary>Quaternion multiplication is associative, unlike the octonions.</summary>
    [Fact]
    public void Quaternion_IsAssociative()
    {
        var a = new double[] { 1, 2, -3, 4 };
        var b = new double[] { -2, 0.5, 1, 3 };
        var c = new double[] { 0, -1, 2, 0.25 };

        double[] left = Multiply(CodegenAlgebra.Quaternion,
            Multiply(CodegenAlgebra.Quaternion, a, b), c);
        double[] right = Multiply(CodegenAlgebra.Quaternion, a,
            Multiply(CodegenAlgebra.Quaternion, b, c));

        for (int i = 0; i < 4; i++) Assert.Equal(left[i], right[i], 9);
    }

    private static double Norm(double[] q)
    {
        double sum = 0;
        foreach (double v in q) sum += v * v;
        return Math.Sqrt(sum);
    }

    // ---- The kernels ---------------------------------------------------------------------

    /// <summary>An elementwise product over an algebra.</summary>
    private static CodegenKernelSpec ElementwiseProduct(CodegenAlgebra algebra, int count)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));

        var a = new CodegenTensorBinding(0, "a", new[] { count },
            new[] { CodegenAffineExpr.Axis(0) });
        var b = new CodegenTensorBinding(1, "b", new[] { count },
            new[] { CodegenAffineExpr.Axis(0) });
        var output = new CodegenTensorBinding(2, "out", new[] { count },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        return new CodegenKernelSpec("algebra_mul", space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.None, algebra: algebra);
    }

    /// <summary>A contraction over an algebra -- a complex matrix-vector product.</summary>
    private static CodegenKernelSpec Contraction(CodegenAlgebra algebra, int rows, int inner)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("k", inner));

        var m = new CodegenTensorBinding(0, "m", new[] { rows, inner },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var v = new CodegenTensorBinding(1, "v", new[] { inner },
            new[] { CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(2, "out", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        return new CodegenKernelSpec("algebra_contract", space, new[] { m, v }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum, algebra: algebra);
    }

    private static double[] Ramp(int elements, int components, int salt)
    {
        var data = new double[elements * components];
        for (int i = 0; i < data.Length; i++) data[i] = (((i * 7 + salt) % 13) - 6) / 4.0;
        return data;
    }

    /// <summary>The interpreter's elementwise product must match the table applied by hand.</summary>
    [Theory]
    [InlineData(CodegenAlgebra.Complex)]
    [InlineData(CodegenAlgebra.Quaternion)]
    public void ElementwiseProduct_MatchesTheTableAppliedByHand(CodegenAlgebra algebra)
    {
        int components = algebra.Components();
        var spec = ElementwiseProduct(algebra, 5);
        double[] a = Ramp(5, components, 0), b = Ramp(5, components, 3);

        double[] got = spec.Interpret(new[] { a, b });

        for (int e = 0; e < 5; e++)
        {
            var left = new double[components];
            var right = new double[components];
            Array.Copy(a, e * components, left, 0, components);
            Array.Copy(b, e * components, right, 0, components);

            double[] want = Multiply(algebra, left, right);
            for (int c = 0; c < components; c++)
                Assert.Equal(want[c], got[e * components + c], 9);
        }
    }

    /// <summary>A complex contraction, against a hand-written sum of complex products.</summary>
    [Fact]
    public void ComplexContraction_MatchesAHandWrittenSum()
    {
        const int Rows = 3, Inner = 4;
        var spec = Contraction(CodegenAlgebra.Complex, Rows, Inner);
        double[] m = Ramp(Rows * Inner, 2, 0), v = Ramp(Inner, 2, 5);

        double[] got = spec.Interpret(new[] { m, v });

        for (int i = 0; i < Rows; i++)
        {
            double real = 0, imaginary = 0;
            for (int k = 0; k < Inner; k++)
            {
                double mr = m[(i * Inner + k) * 2], mi = m[(i * Inner + k) * 2 + 1];
                double vr = v[k * 2], vi = v[k * 2 + 1];
                real += mr * vr - mi * vi;
                imaginary += mr * vi + mi * vr;
            }
            Assert.Equal(real, got[i * 2], 9);
            Assert.Equal(imaginary, got[i * 2 + 1], 9);
        }
    }

    /// <summary>The emitted kernel must carry one accumulator per component.</summary>
    [Theory]
    [InlineData(CodegenAlgebra.Complex, 4)]
    [InlineData(CodegenAlgebra.Quaternion, 16)]
    public void EmittedKernel_UsesTheProductTable(CodegenAlgebra algebra, int expectedTerms)
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(ElementwiseProduct(algebra, 256), 8, 6);

        Assert.Equal(expectedTerms, emitter.AlgebraProductTerms);
        Assert.Contains("fma.rn.f32", ptx, StringComparison.Ordinal);
        Assert.Contains(algebra.ToString(), ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// Components are ADJACENT, so an element is addressed at 4*components bytes and the
    /// components at fixed offsets inside it. The index maps do not change at all.
    /// </summary>
    [Fact]
    public void ComponentsAreAdjacentInMemory()
    {
        string ptx = new PtxAffineEmitter().Emit(
            ElementwiseProduct(CodegenAlgebra.Quaternion, 256), 8, 6);

        Assert.Contains(", 16;", ptx, StringComparison.Ordinal);     // 4 components x 4 bytes
        Assert.Contains("+4]", ptx, StringComparison.Ordinal);
        Assert.Contains("+8]", ptx, StringComparison.Ordinal);
        Assert.Contains("+12]", ptx, StringComparison.Ordinal);
    }

    // ---- What must be refused ------------------------------------------------------------

    /// <summary>There is no order on the complex numbers, so Max is undefined.</summary>
    [Fact]
    public void MaxReduction_IsRefused()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", 8), CodegenAxis.Reduce("k", 4));
        var a = new CodegenTensorBinding(0, "a", new[] { 8, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(1, "out", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad", space, new[] { a }, output, new[] { 0 }, CodegenReduceKind.Max,
            algebra: CodegenAlgebra.Complex));

        Assert.Contains("no order", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// A real activation applied component-wise is a DIFFERENT operator, not the same one
    /// generalised, so it has to be written as one rather than silently allowed.
    /// </summary>
    [Fact]
    public void RealActivation_IsRefused()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 8));
        var a = new CodegenTensorBinding(0, "a", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) });
        var output = new CodegenTensorBinding(1, "out", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad", space, new[] { a }, output, new[] { 0 }, CodegenReduceKind.None,
            activation: CodegenActivationKind.ReLU, algebra: CodegenAlgebra.Complex));

        Assert.Contains("DIFFERENT operator", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>Narrow storage is refused: the components are fp32.</summary>
    [Fact]
    public void NarrowStorage_IsRefused()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 8));
        var a = new CodegenTensorBinding(0, "a", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: CodegenElementType.Float16);
        var output = new CodegenTensorBinding(1, "out", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad", space, new[] { a }, output, new[] { 0 }, CodegenReduceKind.None,
            algebra: CodegenAlgebra.Complex));

        Assert.Contains("fp32", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>Real kernels must take the original path, unchanged.</summary>
    [Fact]
    public void RealKernels_AreUnchanged()
    {
        var emitter = new PtxAffineEmitter();
        emitter.Emit(ElementwiseProduct(CodegenAlgebra.Real, 256), 8, 6);

        Assert.Equal(0, emitter.AlgebraProductTerms);
    }
}
