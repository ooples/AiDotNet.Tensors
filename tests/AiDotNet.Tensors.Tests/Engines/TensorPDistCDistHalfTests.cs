using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class TensorPDistCDistHalfTests
{
    public static IEnumerable<object[]> NormOrders()
    {
        yield return new object[] { 0.0 };
        yield return new object[] { 1.0 };
        yield return new object[] { 2.0 };
        yield return new object[] { 3.0 };
        yield return new object[] { double.PositiveInfinity };
    }

    [Theory]
    [MemberData(nameof(NormOrders))]
    public void TensorCDist_HalfAndBFloat16_ContiguousAndTransposed_MatchDouble(double p)
    {
        var x1Vals = new double[] { -1.5, 0.25, 2.0, 0.5, -0.75, 1.25 };
        var x2Vals = new double[] { 0.5, -1.0, 2.0, -2.0, 0.25, 0.75, 1.0, 1.5, -0.5 };
        var x1Shape = new[] { 2, 3 };
        var x2Shape = new[] { 3, 3 };

        AssertCDistMatchesDouble<Half>(x1Vals, x1Shape, x2Vals, x2Shape, p, transposeInputs: false, tol: 1e-2);
        AssertCDistMatchesDouble<Half>(x1Vals, new[] { 3, 2 }, x2Vals, new[] { 3, 3 }, p, transposeInputs: true, tol: 1e-2);
        AssertCDistMatchesDouble<BFloat16>(x1Vals, x1Shape, x2Vals, x2Shape, p, transposeInputs: false, tol: 5e-2);
        AssertCDistMatchesDouble<BFloat16>(x1Vals, new[] { 3, 2 }, x2Vals, new[] { 3, 3 }, p, transposeInputs: true, tol: 5e-2);
    }

    [Fact]
    public void TensorCDist_HalfAndBFloat16_BroadcastBatchAndTapeBackward_MatchDouble()
    {
        var x1Vals = new double[]
        {
            -1.25, 0.5, 1.75, 0.25, -0.5, 1.0,
            0.75, -1.5, 0.25, -0.25, 1.25, -0.75,
        };
        var x2Vals = new double[]
        {
            0.5, -0.75, 1.0,
            -1.0, 0.25, -0.5,
        };

        AssertCDistBackwardMatchesDouble<Half>(x1Vals, new[] { 2, 2, 3 }, x2Vals, new[] { 1, 2, 3 }, p: 2.0, tol: 2e-2);
        AssertCDistBackwardMatchesDouble<BFloat16>(x1Vals, new[] { 2, 2, 3 }, x2Vals, new[] { 1, 2, 3 }, p: 2.0, tol: 8e-2);
    }

    [Theory]
    [MemberData(nameof(NormOrders))]
    public void TensorPDist_HalfAndBFloat16_ContiguousAndTransposed_MatchDouble(double p)
    {
        var vals = new double[]
        {
            -1.5, 0.5, 2.0,
            0.5, -1.25, 2.0,
            1.75, 0.25, -0.75,
            -0.25, 1.5, 0.5,
        };

        AssertPDistMatchesDouble<Half>(vals, new[] { 4, 3 }, p, transposeInput: false, tol: 1e-2);
        AssertPDistMatchesDouble<Half>(vals, new[] { 3, 4 }, p, transposeInput: true, tol: 1e-2);
        AssertPDistMatchesDouble<BFloat16>(vals, new[] { 4, 3 }, p, transposeInput: false, tol: 5e-2);
        AssertPDistMatchesDouble<BFloat16>(vals, new[] { 3, 4 }, p, transposeInput: true, tol: 5e-2);
    }

    [Fact]
    public void TensorPDist_HalfAndBFloat16_TapeBackward_MatchDouble()
    {
        var vals = new double[]
        {
            -1.25, 0.5, 1.5,
            0.75, -1.0, 0.25,
            -0.5, 1.25, -0.75,
        };

        AssertPDistBackwardMatchesDouble<Half>(vals, new[] { 3, 3 }, p: 2.0, tol: 2e-2);
        AssertPDistBackwardMatchesDouble<BFloat16>(vals, new[] { 3, 3 }, p: 2.0, tol: 8e-2);
    }

    [Fact]
    public void SpecialFnAndNanToNumBackward_HalfAndBFloat16_DoNotRequireIConvertible()
    {
        AssertSpecialBackwardMatchesDouble<Half>(tol: 2e-2);
        AssertSpecialBackwardMatchesDouble<BFloat16>(tol: 8e-2);
    }

    private static void AssertCDistMatchesDouble<T>(
        double[] x1Vals, int[] x1Shape, double[] x2Vals, int[] x2Shape,
        double p, bool transposeInputs, double tol)
    {
        var engine = new CpuEngine();
        var ops = MathHelper.GetNumericOperations<T>();

        var x1T = CreateTypedTensor<T>(x1Vals, x1Shape);
        var x2T = CreateTypedTensor<T>(x2Vals, x2Shape);
        var x1Ref = ToDoubleTensor(x1T);
        var x2Ref = ToDoubleTensor(x2T);

        if (transposeInputs)
        {
            x1T = x1T.Transpose(new[] { 1, 0 });
            x2T = x2T.Transpose(new[] { 1, 0 });
            x1Ref = x1Ref.Transpose(new[] { 1, 0 });
            x2Ref = x2Ref.Transpose(new[] { 1, 0 });
            Assert.False(x1T.IsContiguous);
            Assert.False(x2T.IsContiguous);
        }

        var actual = engine.TensorCDist(x1T, x2T, p);
        var expected = engine.TensorCDist(x1Ref, x2Ref, p);

        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.InRange(Math.Abs(ops.ToDouble(actual[i]) - expected[i]), 0.0, tol);
        }
    }

    private static void AssertPDistMatchesDouble<T>(
        double[] vals, int[] shape, double p, bool transposeInput, double tol)
    {
        var engine = new CpuEngine();
        var ops = MathHelper.GetNumericOperations<T>();

        var xT = CreateTypedTensor<T>(vals, shape);
        var xRef = ToDoubleTensor(xT);

        if (transposeInput)
        {
            xT = xT.Transpose(new[] { 1, 0 });
            xRef = xRef.Transpose(new[] { 1, 0 });
            Assert.False(xT.IsContiguous);
        }

        var actual = engine.TensorPDist(xT, p);
        var expected = engine.TensorPDist(xRef, p);

        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.InRange(Math.Abs(ops.ToDouble(actual[i]) - expected[i]), 0.0, tol);
        }
    }

    private static void AssertCDistBackwardMatchesDouble<T>(
        double[] x1Vals, int[] x1Shape, double[] x2Vals, int[] x2Shape, double p, double tol)
    {
        var engine = new CpuEngine();
        var ops = MathHelper.GetNumericOperations<T>();

        var x1T = CreateTypedTensor<T>(x1Vals, x1Shape);
        var x2T = CreateTypedTensor<T>(x2Vals, x2Shape);
        var x1Ref = ToDoubleTensor(x1T);
        var x2Ref = ToDoubleTensor(x2T);

        Dictionary<Tensor<T>, Tensor<T>> gradsT;
        using (var tape = new GradientTape<T>())
        {
            var dist = engine.TensorCDist(x1T, x2T, p);
            var loss = engine.ReduceSum(dist, null);
            gradsT = tape.ComputeGradients(loss, new[] { x1T, x2T });
        }

        Dictionary<Tensor<double>, Tensor<double>> gradsRef;
        using (var tape = new GradientTape<double>())
        {
            var dist = engine.TensorCDist(x1Ref, x2Ref, p);
            var loss = engine.ReduceSum(dist, null);
            gradsRef = tape.ComputeGradients(loss, new[] { x1Ref, x2Ref });
        }

        Assert.True(gradsT.TryGetValue(x1T, out var gx1) && gx1 is not null);
        Assert.True(gradsT.TryGetValue(x2T, out var gx2) && gx2 is not null);
        Assert.True(gradsRef.TryGetValue(x1Ref, out var gx1Ref) && gx1Ref is not null);
        Assert.True(gradsRef.TryGetValue(x2Ref, out var gx2Ref) && gx2Ref is not null);

        for (int i = 0; i < gx1Ref.Length; i++)
            Assert.InRange(Math.Abs(ops.ToDouble(gx1[i]) - gx1Ref[i]), 0.0, tol);
        for (int i = 0; i < gx2Ref.Length; i++)
            Assert.InRange(Math.Abs(ops.ToDouble(gx2[i]) - gx2Ref[i]), 0.0, tol);
    }

    private static void AssertPDistBackwardMatchesDouble<T>(
        double[] vals, int[] shape, double p, double tol)
    {
        var engine = new CpuEngine();
        var ops = MathHelper.GetNumericOperations<T>();

        var xT = CreateTypedTensor<T>(vals, shape);
        var xRef = ToDoubleTensor(xT);

        Dictionary<Tensor<T>, Tensor<T>> gradsT;
        using (var tape = new GradientTape<T>())
        {
            var dist = engine.TensorPDist(xT, p);
            var loss = engine.ReduceSum(dist, null);
            gradsT = tape.ComputeGradients(loss, new[] { xT });
        }

        Dictionary<Tensor<double>, Tensor<double>> gradsRef;
        using (var tape = new GradientTape<double>())
        {
            var dist = engine.TensorPDist(xRef, p);
            var loss = engine.ReduceSum(dist, null);
            gradsRef = tape.ComputeGradients(loss, new[] { xRef });
        }

        Assert.True(gradsT.TryGetValue(xT, out var gx) && gx is not null);
        Assert.True(gradsRef.TryGetValue(xRef, out var gxRef) && gxRef is not null);
        for (int i = 0; i < gxRef.Length; i++)
            Assert.InRange(Math.Abs(ops.ToDouble(gx[i]) - gxRef[i]), 0.0, tol);
    }

    private static void AssertSpecialBackwardMatchesDouble<T>(double tol)
    {
        var engine = new CpuEngine();
        var ops = MathHelper.GetNumericOperations<T>();

        var posVals = new double[] { 1.5, 2.25, 3.5, 4.75 };
        var xPosT = CreateTypedTensor<T>(posVals, new[] { 2, 2 });
        var xPosRef = ToDoubleTensor(xPosT);
        var goPosT = CreateTypedTensor<T>(new[] { 1.0, -0.5, 0.75, 1.25 }, new[] { 2, 2 });
        var goPosRef = ToDoubleTensor(goPosT);
        var dummyOutT = new Tensor<T>(new[] { 2, 2 });
        var dummyOutRef = new Tensor<double>(new[] { 2, 2 });

        var lgammaGradsT = new Dictionary<Tensor<T>, Tensor<T>>();
        BackwardFunctions<T>.LgammaBackward(goPosT, new[] { xPosT }, dummyOutT, Array.Empty<object>(), engine, lgammaGradsT);
        var lgammaGradsRef = new Dictionary<Tensor<double>, Tensor<double>>();
        BackwardFunctions<double>.LgammaBackward(goPosRef, new[] { xPosRef }, dummyOutRef, Array.Empty<object>(), engine, lgammaGradsRef);
        Assert.True(lgammaGradsT.TryGetValue(xPosT, out var gLgamma) && gLgamma is not null);
        Assert.True(lgammaGradsRef.TryGetValue(xPosRef, out var gLgammaRef) && gLgammaRef is not null);
        for (int i = 0; i < gLgammaRef.Length; i++)
            Assert.InRange(Math.Abs(ops.ToDouble(gLgamma[i]) - gLgammaRef[i]), 0.0, tol);

        var digammaGradsT = new Dictionary<Tensor<T>, Tensor<T>>();
        BackwardFunctions<T>.DigammaBackward(goPosT, new[] { xPosT }, dummyOutT, Array.Empty<object>(), engine, digammaGradsT);
        var digammaGradsRef = new Dictionary<Tensor<double>, Tensor<double>>();
        BackwardFunctions<double>.DigammaBackward(goPosRef, new[] { xPosRef }, dummyOutRef, Array.Empty<object>(), engine, digammaGradsRef);
        Assert.True(digammaGradsT.TryGetValue(xPosT, out var gDigamma) && gDigamma is not null);
        Assert.True(digammaGradsRef.TryGetValue(xPosRef, out var gDigammaRef) && gDigammaRef is not null);
        for (int i = 0; i < gDigammaRef.Length; i++)
            Assert.InRange(Math.Abs(ops.ToDouble(gDigamma[i]) - gDigammaRef[i]), 0.0, tol);

        var nanInfVals = new double[] { -1.25, double.NaN, double.PositiveInfinity, 2.5 };
        var xNanT = CreateTypedTensor<T>(nanInfVals, new[] { 4 });
        var goNanT = CreateTypedTensor<T>(new[] { 1.0, 1.0, 1.0, 1.0 }, new[] { 4 });
        var nanGradsT = new Dictionary<Tensor<T>, Tensor<T>>();
        BackwardFunctions<T>.NanToNumBackward(goNanT, new[] { xNanT }, new Tensor<T>(new[] { 4 }), Array.Empty<object>(), engine, nanGradsT);
        Assert.True(nanGradsT.TryGetValue(xNanT, out var gx) && gx is not null);
        Assert.Equal(1.0, ops.ToDouble(gx[0]), 3);
        Assert.Equal(0.0, ops.ToDouble(gx[1]), 3);
        Assert.Equal(0.0, ops.ToDouble(gx[2]), 3);
        Assert.Equal(1.0, ops.ToDouble(gx[3]), 3);
    }

    private static Tensor<T> CreateTypedTensor<T>(double[] values, int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var typed = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
            typed[i] = ops.FromDouble(values[i]);
        return new Tensor<T>(typed, shape);
    }

    private static Tensor<double> ToDoubleTensor<T>(Tensor<T> source)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var data = new double[source.Length];
        for (int i = 0; i < source.Length; i++)
            data[i] = ops.ToDouble(source[i]);
        return new Tensor<double>(data, source.Shape.ToArray());
    }
}
