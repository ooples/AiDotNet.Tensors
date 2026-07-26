// Copyright (c) AiDotNet. All rights reserved.
// The adjoint test that matters: <forward(x), y> == <x, backward(y)>.
//
// Checking a derived backward kernel against its own interpreter proves only that the
// derivation is self-consistent. The dot-product identity is independent of how the
// adjoint was constructed -- it holds for the true adjoint of a linear operator and
// for nothing else -- so it catches a wrong index map, a wrong reduction set, and a
// missing exactness predicate alike.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenAdjointTests
{
    private static double[] Fill(long count, int salt)
    {
        var v = new double[count];
        for (long i = 0; i < count; i++) v[i] = (((i * 37 + salt * 101) % 97) - 48) / 64.0;
        return v;
    }

    private static long Elements(IReadOnlyList<int> shape)
    {
        long total = 1;
        foreach (int d in shape) total *= d;
        return total;
    }

    private static double Dot(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>Depthwise 3x3: the window inverts to an exact-division transposed window.</summary>
    [Fact]
    public void DepthwiseConv_AdjointSatisfiesDotProductIdentity() =>
        AssertAdjoint(Forward.Depthwise(2, 4, 8, 8), dataInput: 0);

    /// <summary>
    /// Dense 1x1: the contracted axis is a forward PARALLEL axis (output channels),
    /// so the adjoint has to move it into the reduction set.
    /// </summary>
    [Fact]
    public void Conv1x1_AdjointSatisfiesDotProductIdentity() =>
        AssertAdjoint(Forward.Conv1x1(2, 6, 5, 8, 8), dataInput: 0);

    /// <summary>Dense 3x3: both effects at once -- a moved parallel axis AND two windows.</summary>
    [Fact]
    public void Conv3x3_AdjointSatisfiesDotProductIdentity() =>
        AssertAdjoint(Forward.Conv3x3(2, 3, 4, 8, 8), dataInput: 0);

    /// <summary>
    /// Stride 2 is where the exactness predicate is load-bearing: only one in four
    /// output positions contributes to a given input position.
    /// </summary>
    [Fact]
    public void StridedConv_AdjointSatisfiesDotProductIdentity() =>
        AssertAdjoint(Forward.DepthwiseStrided(2, 4, 8, 8, stride: 2), dataInput: 0);

    private static void AssertAdjoint(CodegenKernelSpec forward, int dataInput)
    {
        var backward = CodegenAdjoint.BackwardData(forward, dataInput);

        int weightInput = dataInput == 0 ? 1 : 0;
        var weight = forward.Inputs[weightInput];
        double[] x = Fill(Elements(forward.Inputs[dataInput].Shape), 1);
        double[] w = Fill(Elements(weight.Shape), 2);
        double[] y = Fill(Elements(forward.Output.Shape), 3);

        var forwardInputs = new double[2][];
        forwardInputs[dataInput] = x;
        forwardInputs[weightInput] = w;
        double lhs = Dot(forward.Interpret(forwardInputs), y);
        double rhs = Dot(x, backward.Interpret(new[] { y, w }));

        Assert.True(Math.Abs(lhs - rhs) <= 1e-9 * Math.Max(1.0, Math.Abs(lhs)),
            forward.Name + ": inner product via forward is " + lhs.ToString("R") +
            " but via the derived backward is " + rhs.ToString("R") +
            "; the derived map is not the adjoint.");
        Assert.NotEqual(0.0, lhs);
    }

    /// <summary>Derived backward kernels must also be emittable, not merely expressible.</summary>
    [Fact]
    public void DerivedBackwardKernels_Emit()
    {
        foreach (var forward in new[]
                 {
                     Forward.Depthwise(2, 4, 8, 8),
                     Forward.Conv1x1(2, 6, 5, 8, 8),
                     Forward.Conv3x3(2, 3, 4, 8, 8),
                     Forward.DepthwiseStrided(2, 4, 8, 8, 2)
                 })
        {
            var backward = CodegenAdjoint.BackwardData(forward, 0);
            string ptx = new PtxAffineEmitter().Emit(backward, 8, 6);
            Assert.Contains(".visible .entry", ptx, StringComparison.Ordinal);
            Assert.Contains(backward.Name, ptx, StringComparison.Ordinal);
        }
    }

    /// <summary>Operators whose adjoint is not an index-map transform must be refused.</summary>
    [Fact]
    public void NonLinearForward_IsRefused()
    {
        var withRelu = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(2, 4, 8, 8);
        Assert.Throws<NotSupportedException>(() => CodegenAdjoint.BackwardData(withRelu, 0));
    }

    /// <summary>
    /// The WEIGHT gradient, tested by the same identity with the roles swapped. The
    /// forward operator is bilinear in (data, weights), so
    ///
    ///     &lt;fwd(x, w), y&gt;  ==  &lt;w, dW(x, y)&gt;
    ///
    /// holds for the true weight gradient and for nothing else. Without this kernel the
    /// generated set cannot train -- only the data gradient existed.
    /// </summary>
    [Theory]
    [InlineData("depthwise")]
    [InlineData("conv1x1")]
    [InlineData("conv3x3")]
    [InlineData("strided")]
    public void WeightGradient_SatisfiesTheDotProductIdentity(string which)
    {
        CodegenKernelSpec forward = which switch
        {
            "depthwise" => Forward.Depthwise(2, 4, 8, 8),
            "conv1x1" => Forward.Conv1x1(2, 6, 5, 8, 8),
            "conv3x3" => Forward.Conv3x3(2, 3, 4, 8, 8),
            _ => Forward.DepthwiseStrided(2, 4, 8, 8, 2),
        };

        // Operand 1 is the weights in every one of these specs.
        var backward = CodegenAdjoint.BackwardWeights(forward, 1);

        double[] x = Fill(Elements(forward.Inputs[0].Shape), 1);
        double[] w = Fill(Elements(forward.Inputs[1].Shape), 2);
        double[] y = Fill(Elements(forward.Output.Shape), 3);

        double lhs = Dot(forward.Interpret(new[] { x, w }), y);

        // The derived kernel takes (dOut, data) and produces dWeights.
        double rhs = Dot(w, backward.Interpret(new[] { y, x }));

        Assert.True(Math.Abs(lhs - rhs) <= 1e-9 * Math.Max(1.0, Math.Abs(lhs)),
            which + ": inner product via forward is " + lhs.ToString("R") +
            " but via the derived weight gradient is " + rhs.ToString("R") +
            "; the derived map is not the adjoint in the weight argument.");
        Assert.NotEqual(0.0, lhs);
    }

    /// <summary>Derived weight-gradient kernels must also be emittable.</summary>
    [Fact]
    public void WeightGradientKernels_Emit()
    {
        foreach (var forward in new[]
                 {
                     Forward.Depthwise(2, 4, 8, 8),
                     Forward.Conv1x1(2, 6, 5, 8, 8),
                     Forward.Conv3x3(2, 3, 4, 8, 8),
                 })
        {
            var backward = CodegenAdjoint.BackwardWeights(forward, 1);
            string ptx = new PtxAffineEmitter().Emit(backward, 8, 6);
            Assert.Contains(".visible .entry", ptx, StringComparison.Ordinal);
        }
    }

    private static class Forward
    {
        internal static CodegenKernelSpec Depthwise(int n, int c, int h, int w) =>
            DepthwiseStrided(n, c, h, w, 1);

        internal static CodegenKernelSpec DepthwiseStrided(int n, int c, int h, int w, int stride)
        {
            int oh = h / stride, ow = w / stride;
            var space = new CodegenIterationSpace(
                CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("c", c),
                CodegenAxis.Parallel("oh", oh), CodegenAxis.Parallel("ow", ow),
                CodegenAxis.Reduce("kh", 3), CodegenAxis.Reduce("kw", 3));
            const int N = 0, C = 1, OH = 2, OW = 3, KH = 4, KW = 5;

            var input = new CodegenTensorBinding(0, "input", new[] { n, c, h, w },
                new[]
                {
                    CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                    CodegenAffineExpr.Window(OH, KH, stride, 1),
                    CodegenAffineExpr.Window(OW, KW, stride, 1)
                });
            var weights = new CodegenTensorBinding(1, "weights", new[] { c, 3, 3 },
                new[] { CodegenAffineExpr.Axis(C), CodegenAffineExpr.Axis(KH), CodegenAffineExpr.Axis(KW) });
            var output = new CodegenTensorBinding(2, "output", new[] { n, c, oh, ow },
                new[]
                {
                    CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                    CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
                }, isOutput: true);

            return new CodegenKernelSpec("dw3x3s" + stride, space, new[] { input, weights }, output,
                new[] { 0, 1 }, CodegenReduceKind.Sum);
        }

        internal static CodegenKernelSpec Conv1x1(int n, int c, int k, int h, int w)
        {
            var space = new CodegenIterationSpace(
                CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("k", k),
                CodegenAxis.Parallel("oh", h), CodegenAxis.Parallel("ow", w),
                CodegenAxis.Reduce("c", c));
            const int N = 0, K = 1, OH = 2, OW = 3, C = 4;

            var input = new CodegenTensorBinding(0, "input", new[] { n, c, h, w },
                new[]
                {
                    CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                    CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
                });
            var weights = new CodegenTensorBinding(1, "weights", new[] { k, c },
                new[] { CodegenAffineExpr.Axis(K), CodegenAffineExpr.Axis(C) });
            var output = new CodegenTensorBinding(2, "output", new[] { n, k, h, w },
                new[]
                {
                    CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(K),
                    CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
                }, isOutput: true);

            return new CodegenKernelSpec("conv1x1", space, new[] { input, weights }, output,
                new[] { 0, 1 }, CodegenReduceKind.Sum);
        }

        internal static CodegenKernelSpec Conv3x3(int n, int c, int k, int h, int w)
        {
            var space = new CodegenIterationSpace(
                CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("k", k),
                CodegenAxis.Parallel("oh", h), CodegenAxis.Parallel("ow", w),
                CodegenAxis.Reduce("c", c), CodegenAxis.Reduce("kh", 3), CodegenAxis.Reduce("kw", 3));
            const int N = 0, K = 1, OH = 2, OW = 3, C = 4, KH = 5, KW = 6;

            var input = new CodegenTensorBinding(0, "input", new[] { n, c, h, w },
                new[]
                {
                    CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                    CodegenAffineExpr.Window(OH, KH, 1, 1), CodegenAffineExpr.Window(OW, KW, 1, 1)
                });
            var weights = new CodegenTensorBinding(1, "weights", new[] { k, c, 3, 3 },
                new[]
                {
                    CodegenAffineExpr.Axis(K), CodegenAffineExpr.Axis(C),
                    CodegenAffineExpr.Axis(KH), CodegenAffineExpr.Axis(KW)
                });
            var output = new CodegenTensorBinding(2, "output", new[] { n, k, h, w },
                new[]
                {
                    CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(K),
                    CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
                }, isOutput: true);

            return new CodegenKernelSpec("conv3x3", space, new[] { input, weights }, output,
                new[] { 0, 1 }, CodegenReduceKind.Sum);
        }
    }
}
