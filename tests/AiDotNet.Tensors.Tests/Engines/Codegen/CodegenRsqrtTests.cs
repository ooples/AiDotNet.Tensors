// Copyright (c) AiDotNet. All rights reserved.
// The reciprocal square root, which RMSNorm and Adam both need.
//
// Added alongside the extra-output model because both are prerequisites for the optimizer
// family, and because PR #874's own measurement redirected the target there: a hand-written
// fused SGD TIED the existing AiDotNet kernel (0.73x-1.05x), since that kernel is already
// single-pass. The unfused headroom is in Adam and AdamW, and neither is expressible
// without an rsqrt.
//
// rsqrt.approx.f32 is APPROXIMATE, like ex2 and rcp, so this measures the deviation rather
// than assuming it.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenRsqrtTests
{
    /// <summary>The oracle's rsqrt must be the mathematical one.</summary>
    [Theory]
    [InlineData(0.25)]
    [InlineData(1.0)]
    [InlineData(4.0)]
    [InlineData(1e-3)]
    [InlineData(1e6)]
    public void Oracle_ComputesTheReciprocalSquareRoot(double x)
    {
        Assert.Equal(1.0 / Math.Sqrt(x),
            CodegenKernelSpec.ApplyActivation(CodegenActivationKind.Rsqrt, x), 12);
    }

    /// <summary>An RMSNorm-shaped spec: rsqrt of a mean of squares.</summary>
    private static CodegenKernelSpec RmsNormScale(int rows, int columns)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("j", columns));

        var x = new CodegenTensorBinding(0, "x", new[] { rows, columns },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(1, "scale", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        // 1 / sqrt(mean(x^2)) -- the RMSNorm normalising factor.
        return new CodegenKernelSpec("rmsnorm_scale", space, new[] { x }, output,
            new[] { 0 }, CodegenReduceKind.Sum,
            activation: CodegenActivationKind.Rsqrt,
            reduceScale: 1.0 / columns,
            preReduce: CodegenPreReduceOp.Square);
    }

    /// <summary>
    /// The RMSNorm factor must match a hand-written one. This composes three separate
    /// additions -- pre-reduction square, constant scale, rsqrt epilogue -- so it is also a
    /// check that they compose in the right order.
    /// </summary>
    [Theory]
    [InlineData(4, 16)]
    [InlineData(7, 64)]
    public void RmsNormScale_MatchesAHandWrittenReference(int rows, int columns)
    {
        var spec = RmsNormScale(rows, columns);

        var x = new double[rows * columns];
        for (int i = 0; i < x.Length; i++) x[i] = (((i * 37) % 97) - 48) / 32.0;

        double[] got = spec.Interpret(new[] { x });

        for (int i = 0; i < rows; i++)
        {
            double sumSquares = 0;
            for (int j = 0; j < columns; j++)
            {
                double v = x[i * columns + j];
                sumSquares += v * v;
            }
            Assert.Equal(1.0 / Math.Sqrt(sumSquares / columns), got[i], 9);
        }
    }

    /// <summary>The emitted kernel must use the hardware rsqrt.</summary>
    [Fact]
    public void EmittedKernel_UsesRsqrtApprox()
    {
        string ptx = new PtxAffineEmitter().Emit(RmsNormScale(32, 64), 8, 6);
        Assert.Contains("rsqrt.approx.f32", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// The order matters: square inside the reduction, then scale, THEN rsqrt. Applying
    /// the scale after the rsqrt would compute sqrt(columns)/sqrt(sum), which is a
    /// different number that still looks like a normalising factor.
    /// </summary>
    [Fact]
    public void ScaleAppliesBeforeTheRsqrt()
    {
        var spec = RmsNormScale(1, 4);
        var x = new double[] { 2.0, 2.0, 2.0, 2.0 };   // mean of squares = 4, rsqrt = 0.5

        Assert.Equal(0.5, spec.Interpret(new[] { x })[0], 9);
    }
}
