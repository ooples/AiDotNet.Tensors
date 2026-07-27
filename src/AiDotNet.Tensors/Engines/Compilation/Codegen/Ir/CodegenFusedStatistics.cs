// Copyright (c) AiDotNet. All rights reserved.
// Softmax and LayerNorm as multi-pass programs over the existing spec form.
//
// These were blocked on the BODY's shape, not on the number of passes. The spec applies
// its activation once, to the finished accumulator, but softmax's denominator needs
// exp applied to every TERM before summing, and LayerNorm's variance needs a square in the
// same position. Sequencing two kernels does not supply that; a pre-reduction slot does,
// and CodegenKernelSpec now has one (PreBiasInput + PreReduce).
//
// With that in place both operators fall out of pieces that already existed:
//
//   ReduceScale   (EXP-2)  negates a maximum in the same pass that computes it
//   PreBias       (here)   broadcasts the per-row statistic back over the row
//   PreReduce     (here)   exp or square, inside the reduction
//   Reciprocal    (here)   turns a summed denominator into a multiplier
//   ScaleInput    (existing) applies that multiplier per row
//
// Numerical stability is not optional here. exp(x) overflows fp32 above about 88, so the
// maximum is subtracted first -- that is why pass 1 exists at all, and why it produces the
// NEGATED maximum rather than the maximum: the spec can add a bias but cannot subtract one.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>A multi-kernel program: run the passes in order, then read the output.</summary>
/// <param name="Passes">Kernels to launch in order.</param>
/// <param name="TempElements">Elements each intermediate needs, parallel to the passes.</param>
/// <param name="Description">What the program computes, for reporting.</param>
public sealed record CodegenProgram(
    IReadOnlyList<CodegenKernelSpec> Passes,
    IReadOnlyList<long> TempElements,
    string Description);

/// <summary>Builds the fused-statistics operators.</summary>
public static class CodegenFusedStatistics
{
    /// <summary>
    /// Row-wise softmax over the last axis of a rank-2 tensor.
    /// </summary>
    /// <param name="rows">Number of independent rows.</param>
    /// <param name="columns">Length of the axis being normalised.</param>
    /// <remarks>
    /// Three passes:
    /// <list type="number">
    /// <item><c>m[i] = -max_j x[i,j]</c> — a maximum with a constant scale of −1, so the
    /// shift can be ADDED later. The spec has no subtract.</item>
    /// <item><c>r[i] = 1 / sum_j exp(x[i,j] + m[i])</c> — the pre-reduction slot doing the
    /// work, with a reciprocal epilogue so pass 3 multiplies instead of dividing.</item>
    /// <item><c>y[i,j] = exp(x[i,j] + m[i]) * r[i]</c> — no reduction axis at all.</item>
    /// </list>
    /// Pass 3 recomputes the exponential rather than reading pass 2's terms back. That is
    /// deliberate: materialising them would cost a full rows x columns intermediate and the
    /// bandwidth to write and re-read it, against one cheap transcendental per element.
    /// </remarks>
    public static CodegenProgram Softmax(int rows, int columns)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns <= 0) throw new ArgumentOutOfRangeException(nameof(columns));

        int[] matrix = { rows, columns };
        int[] vector = { rows };

        // ---- pass 1: m[i] = -max_j x[i,j]
        var maxSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("j", columns));
        var negMax = new CodegenKernelSpec(
            "softmax_negmax", maxSpace,
            new[] { Bind(0, "x", matrix, Axis(0), Axis(1)) },
            Bind(1, "negmax", vector, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Max, reduceScale: -1.0);

        // ---- pass 2: r[i] = 1 / sum_j exp(x[i,j] + m[i])
        var sumSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("j", columns));
        var recipSum = new CodegenKernelSpec(
            "softmax_recipsum", sumSpace,
            new[]
            {
                Bind(0, "x", matrix, Axis(0), Axis(1)),
                Bind(1, "negmax", vector, Axis(0)),
            },
            Bind(2, "recipsum", vector, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Sum,
            activation: CodegenActivationKind.Reciprocal,
            preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1);

        // ---- pass 3: y[i,j] = exp(x[i,j] + m[i]) * r[i]
        var outSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Parallel("j", columns));
        var normalise = new CodegenKernelSpec(
            "softmax_normalise", outSpace,
            new[]
            {
                Bind(0, "x", matrix, Axis(0), Axis(1)),
                Bind(1, "negmax", vector, Axis(0)),
                Bind(2, "recipsum", vector, Axis(0)),
            },
            Bind(3, "y", matrix, isOutput: true, Axis(0), Axis(1)),
            new[] { 0 }, CodegenReduceKind.None,
            scaleInput: 2,
            preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1);

        return new CodegenProgram(
            new[] { negMax, recipSum, normalise },
            new long[] { rows, rows, (long)rows * columns },
            "row softmax " + rows + "x" + columns);
    }

    /// <summary>
    /// Row-wise LayerNorm statistics over the last axis: mean, then reciprocal deviation.
    /// </summary>
    /// <param name="rows">Number of independent rows.</param>
    /// <param name="columns">Length of the axis being normalised.</param>
    /// <param name="epsilon">Added to the variance before the reciprocal square root.</param>
    /// <remarks>
    /// Two passes produce the statistics; the affine application is an ordinary pointwise
    /// kernel the front end already handles.
    /// <list type="number">
    /// <item><c>m[i] = -mean_j x[i,j]</c> — a sum with scale −1/columns, negated for the
    /// same reason softmax's is.</item>
    /// <item><c>v[i] = mean_j (x[i,j] + m[i])^2</c> — the variance, using the pre-reduction
    /// square.</item>
    /// </list>
    /// The reciprocal square root is NOT folded in, because the spec has no rsqrt epilogue
    /// and inventing one to hide an epsilon would be worse than leaving the caller to add
    /// it: epsilon placement is a documented source of disagreement between frameworks.
    /// </remarks>
    public static CodegenProgram LayerNormStatistics(int rows, int columns, double epsilon = 1e-5)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns <= 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (epsilon < 0) throw new ArgumentOutOfRangeException(nameof(epsilon));

        int[] matrix = { rows, columns };
        int[] vector = { rows };

        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("j", columns));

        var negMean = new CodegenKernelSpec(
            "layernorm_negmean", space,
            new[] { Bind(0, "x", matrix, Axis(0), Axis(1)) },
            Bind(1, "negmean", vector, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Sum, reduceScale: -1.0 / columns);

        var variance = new CodegenKernelSpec(
            "layernorm_variance",
            new CodegenIterationSpace(
                CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("j", columns)),
            new[]
            {
                Bind(0, "x", matrix, Axis(0), Axis(1)),
                Bind(1, "negmean", vector, Axis(0)),
            },
            Bind(2, "variance", vector, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Sum, reduceScale: 1.0 / columns,
            preReduce: CodegenPreReduceOp.Square, preBiasInput: 1);

        return new CodegenProgram(
            new[] { negMean, variance },
            new long[] { rows, rows },
            "layernorm statistics " + rows + "x" + columns);
    }

    private static CodegenAffineExpr Axis(int axis) => CodegenAffineExpr.Axis(axis);

    private static CodegenTensorBinding Bind(
        int parameterIndex, string name, int[] shape, params CodegenAffineExpr[] map) =>
        new(parameterIndex, name, (int[])shape.Clone(), map);

    private static CodegenTensorBinding Bind(
        int parameterIndex, string name, int[] shape, bool isOutput, params CodegenAffineExpr[] map) =>
        new(parameterIndex, name, (int[])shape.Clone(), map, isOutput);
}
