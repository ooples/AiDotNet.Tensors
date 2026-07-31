// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenKernelSpecValidationTests
{
    private static CodegenIterationSpace Space() =>
        new(CodegenAxis.Parallel("i", 8));

    [Fact]
    public void InputParameterIndex_MustMatchItsPosition()
    {
        var input = new CodegenTensorBinding(1, "input", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) });
        var output = new CodegenTensorBinding(1, "output", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad_input_abi", Space(), new[] { input }, output,
            new[] { 0 }, CodegenReduceKind.None));
        Assert.Contains("position 0", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void OutputParameterIndex_MustFollowTheInputs()
    {
        var input = new CodegenTensorBinding(0, "input", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) });
        var output = new CodegenTensorBinding(2, "output", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad_output_abi", Space(), new[] { input }, output,
            new[] { 0 }, CodegenReduceKind.None));
        Assert.Contains("immediately after", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void OutputAffineAxis_MustExistInTheIterationSpace()
    {
        var input = new CodegenTensorBinding(0, "input", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) });
        var output = new CodegenTensorBinding(1, "output", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(1) }, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad_output_axis", Space(), new[] { input }, output,
            new[] { 0 }, CodegenReduceKind.None));
        Assert.Contains("affine axis 1", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ExistingOpOrdinalsRemainStableWhenConvolutionKindsAreAdded()
    {
        Assert.Equal(47, (int)CodegenOpKind.Softmax);
        Assert.Equal(55, (int)CodegenOpKind.Opaque);
        Assert.Equal(56, (int)CodegenOpKind.Conv2D);
    }
}
