// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Semantic specs shared by the incumbent oracle and the generated-vs-incumbent harness.
/// </summary>
/// <remarks>
/// A ceiling and a head-to-head result must describe the same operator. Keeping the mapping
/// here prevents the two tools from independently drifting into similar-looking but different
/// formulas while retaining the same incumbent kernel name.
/// </remarks>
internal static class IncumbentSemanticSpecs
{
    internal static CodegenKernelSpec Gather(
        string name, int tokens, int vocabulary, int width)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", tokens), CodegenAxis.Parallel("e", width));
        var ids = new CodegenTensorBinding(0, "ids", new[] { tokens },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: CodegenElementType.Int32);
        var table = new CodegenTensorBinding(1, "table", new[] { vocabulary, width },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), vocabulary),
                null,
            });
        var output = new CodegenTensorBinding(2, "out", new[] { tokens, width },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);
        return new CodegenKernelSpec(name, space,
            new[] { ids, table }, output, new[] { 1 }, CodegenReduceKind.None);
    }

    internal static CodegenKernelSpec Scatter(
        string name, int tokens, int vocabulary, int width)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", tokens), CodegenAxis.Parallel("e", width));
        var ids = new CodegenTensorBinding(0, "ids", new[] { tokens },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: CodegenElementType.Int32);
        var grad = new CodegenTensorBinding(1, "grad", new[] { tokens, width },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var table = new CodegenTensorBinding(2, "grad_table", new[] { vocabulary, width },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            isOutput: true,
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), vocabulary),
                null,
            });
        return new CodegenKernelSpec(name, space,
            new[] { ids, grad }, table, new[] { 1 }, CodegenReduceKind.None);
    }

    internal static CodegenKernelSpec Momentum(
        string name, int count, double momentum, double learningRate)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));
        var map = new[] { CodegenAffineExpr.Axis(0) };
        var velocity = new CodegenTensorBinding(0, "v", new[] { count }, map);
        var gradient = new CodegenTensorBinding(1, "g", new[] { count }, map);
        var parameter = new CodegenTensorBinding(2, "p", new[] { count }, map);
        var velocityOut = new CodegenTensorBinding(
            3, "v_out", new[] { count }, map, isOutput: true);
        var parameterOut = new CodegenTensorBinding(
            4, "p_out", new[] { count }, map, isOutput: true);

        return new CodegenKernelSpec(name, space,
            new[] { velocity, gradient, parameter }, velocityOut,
            new[] { 0 }, CodegenReduceKind.None,
            biasInput: 1, reduceScale: momentum,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(parameterOut, CodegenExtraOutputKind.AffineOfPrimary,
                    Scale: -learningRate, BiasInput: 2),
            });
    }

    internal static CodegenKernelSpec RowReduction(
        string name, int rows, int inner, CodegenReduceKind reduce, double reduceScale)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("row", rows), CodegenAxis.Reduce("k", inner));
        var input = new CodegenTensorBinding(0, "x", new[] { rows, inner },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(1, "y", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        return new CodegenKernelSpec(name, space, new[] { input }, output,
            new[] { 0 }, reduce, reduceScale: reduceScale);
    }
}
