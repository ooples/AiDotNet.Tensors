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
    string Description)
{
    // Passes are chained BY NAME: an input named like an earlier pass's output reads that
    // intermediate, and any other input is a program source shared by every pass that
    // names it. Positional chaining (input i = output of pass i-1) worked for softmax by
    // coincidence and cannot express a normalisation, whose affine pass reads two
    // statistics AND two parameters.

    /// <summary>
    /// The program's external inputs: every input binding no earlier pass produces, once per
    /// name, in first-use order.
    /// </summary>
    public IReadOnlyList<CodegenTensorBinding> Sources()
    {
        var produced = new Dictionary<string, CodegenTensorBinding>(StringComparer.Ordinal);
        var seen = new Dictionary<string, CodegenTensorBinding>(StringComparer.Ordinal);
        var sources = new List<CodegenTensorBinding>();
        foreach (var pass in Passes)
        {
            foreach (var input in pass.Inputs)
            {
                if (produced.TryGetValue(input.Name, out var producer))
                {
                    RequireSameStorage(producer, input);
                    continue;
                }
                if (seen.TryGetValue(input.Name, out var first))
                {
                    RequireSameStorage(first, input);
                    continue;
                }
                seen.Add(input.Name, input);
                sources.Add(input);
            }
            if (produced.ContainsKey(pass.Output.Name) || seen.ContainsKey(pass.Output.Name))
                throw new InvalidOperationException(
                    "Pass '" + pass.Name + "' writes '" + pass.Output.Name + "', which the " +
                    "program already has. Every intermediate must be written exactly once.");
            produced.Add(pass.Output.Name, pass.Output);
        }
        return sources;
    }

    /// <summary>
    /// Runs every pass on the fp64 interpreter, feeding each pass the intermediates the
    /// earlier passes produced. Returns every pass output, keyed by name.
    /// </summary>
    public IReadOnlyDictionary<string, double[]> Interpret(IReadOnlyDictionary<string, double[]> sources)
    {
        if (sources is null) throw new ArgumentNullException(nameof(sources));
        foreach (var source in Sources())
        {
            if (!sources.TryGetValue(source.Name, out var data))
                throw new ArgumentException("Missing program source '" + source.Name + "'.", nameof(sources));
            if (data.LongLength != source.ElementCount)
                throw new ArgumentException(
                    "Source '" + source.Name + "' has " + data.LongLength + " elements; the " +
                    "binding needs " + source.ElementCount + ".", nameof(sources));
        }

        var values = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var pair in sources) values[pair.Key] = pair.Value;
        var outputs = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var pass in Passes)
        {
            var operands = new double[pass.Inputs.Count][];
            for (int i = 0; i < operands.Length; i++) operands[i] = values[pass.Inputs[i].Name];
            double[] result = pass.Interpret(operands);
            values[pass.Output.Name] = result;
            outputs[pass.Output.Name] = result;
        }
        return outputs;
    }

    private static void RequireSameStorage(CodegenTensorBinding first, CodegenTensorBinding again)
    {
        if (first.ElementCount != again.ElementCount || first.ElementType != again.ElementType)
            throw new InvalidOperationException(
                "Program tensor '" + first.Name + "' is bound with " + first.ElementCount + " " +
                first.ElementType + " elements in one pass and " + again.ElementCount + " " +
                again.ElementType + " in another; one name must mean one buffer.");
    }
}

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
    public static CodegenProgram LayerNormStatistics(int rows, int columns)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns <= 0) throw new ArgumentOutOfRangeException(nameof(columns));

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

    /// <summary>
    /// Per-sample mean squared error: <c>mse[n] = mean_j (a[n,j] - b[n,j])^2</c>.
    /// </summary>
    /// <param name="samples">Number of independent samples (rows).</param>
    /// <param name="features">Length of the axis being averaged.</param>
    /// <remarks>
    /// One kernel, not a program. The body multiplies its operands, so a DIFFERENCE of two
    /// tensors had no expression at all and this was the last thing in the audit still
    /// blocked on the spec's shape. A signed pre-bias supplies it: the product is
    /// <c>a</c>, the pre-bias is <c>b</c> scaled by −1, the pre-reduction op squares the
    /// result, and the constant scale divides by the count.
    /// </remarks>
    public static CodegenKernelSpec MeanSquaredError(int samples, int features)
    {
        if (samples <= 0) throw new ArgumentOutOfRangeException(nameof(samples));
        if (features <= 0) throw new ArgumentOutOfRangeException(nameof(features));

        int[] matrix = { samples, features };
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", samples), CodegenAxis.Reduce("j", features));

        return new CodegenKernelSpec(
            "mse_per_sample", space,
            new[]
            {
                Bind(0, "a", matrix, Axis(0), Axis(1)),
                Bind(1, "b", matrix, Axis(0), Axis(1)),
            },
            Bind(2, "mse", new[] { samples }, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Sum,
            reduceScale: 1.0 / features,
            preReduce: CodegenPreReduceOp.Square,
            preBiasInput: 1, preBiasScale: -1.0);
    }

    /// <summary>
    /// Full LayerNorm over the last axis: <c>y = (x - mean) * rstd * gamma + beta</c>, with
    /// <c>rstd = 1/sqrt(var + eps)</c>. Also produces the saved statistics a backward pass
    /// reads (<c>negmean</c> and <c>rstd</c>), as PyTorch and the hand-written kernel do.
    /// </summary>
    public static CodegenProgram LayerNorm(int rows, int columns, double epsilon = 1e-5)
    {
        RequireRowShape(rows, columns, epsilon);
        int[] matrix = { rows, columns };
        int[] vector = { rows };
        int[] feature = { columns };
        var reduceRows = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("j", columns));

        var negMean = new CodegenKernelSpec(
            "layernorm_negmean", reduceRows,
            new[] { Bind(0, "x", matrix, Axis(0), Axis(1)) },
            Bind(1, "negmean", vector, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Sum, reduceScale: -1.0 / columns);

        var rstd = new CodegenKernelSpec(
            "layernorm_rstd", reduceRows,
            new[]
            {
                Bind(0, "x", matrix, Axis(0), Axis(1)),
                Bind(1, "negmean", vector, Axis(0)),
            },
            Bind(2, "rstd", vector, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Sum, reduceScale: 1.0 / columns,
            activation: CodegenActivationKind.Rsqrt,
            preReduce: CodegenPreReduceOp.Square, preBiasInput: 1,
            reduceOffset: epsilon);

        var affine = new CodegenKernelSpec(
            "layernorm_affine",
            new CodegenIterationSpace(CodegenAxis.Parallel("i", rows), CodegenAxis.Parallel("j", columns)),
            new[]
            {
                Bind(0, "x", matrix, Axis(0), Axis(1)),
                Bind(1, "negmean", vector, Axis(0)),
                Bind(2, "rstd", vector, Axis(0)),
                Bind(3, "gamma", feature, Axis(1)),
                Bind(4, "beta", feature, Axis(1)),
            },
            Bind(5, "y", matrix, isOutput: true, Axis(0), Axis(1)),
            new[] { 0 }, CodegenReduceKind.None,
            scaleInput: 2, preBiasInput: 1, postScaleInput: 3, postBiasInput: 4);

        return new CodegenProgram(
            new[] { negMean, rstd, affine },
            new long[] { rows, rows, (long)rows * columns },
            "layernorm " + rows + "x" + columns);
    }

    /// <summary>
    /// RMSNorm over the last axis: <c>y = x * rstd * gamma</c> with
    /// <c>rstd = 1/sqrt(mean(x^2) + eps)</c>; <c>rstd</c> is kept for the backward pass.
    /// </summary>
    public static CodegenProgram RmsNorm(int rows, int columns, double epsilon = 1e-6)
    {
        RequireRowShape(rows, columns, epsilon);
        int[] matrix = { rows, columns };
        int[] vector = { rows };

        var rstd = new CodegenKernelSpec(
            "rmsnorm_rstd",
            new CodegenIterationSpace(CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("j", columns)),
            new[] { Bind(0, "x", matrix, Axis(0), Axis(1)) },
            Bind(1, "rstd", vector, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Sum, reduceScale: 1.0 / columns,
            activation: CodegenActivationKind.Rsqrt,
            preReduce: CodegenPreReduceOp.Square, reduceOffset: epsilon);

        var scale = new CodegenKernelSpec(
            "rmsnorm_scale",
            new CodegenIterationSpace(CodegenAxis.Parallel("i", rows), CodegenAxis.Parallel("j", columns)),
            new[]
            {
                Bind(0, "x", matrix, Axis(0), Axis(1)),
                Bind(1, "rstd", vector, Axis(0)),
                Bind(2, "gamma", new[] { columns }, Axis(1)),
            },
            Bind(3, "y", matrix, isOutput: true, Axis(0), Axis(1)),
            new[] { 0 }, CodegenReduceKind.None,
            scaleInput: 1, postScaleInput: 2);

        return new CodegenProgram(
            new[] { rstd, scale },
            new long[] { rows, (long)rows * columns },
            "rmsnorm " + rows + "x" + columns);
    }

    /// <summary>
    /// Training-mode BatchNorm over NCHW: per-channel statistics over (n, h, w), then the
    /// channel affine. The reduction axes are NOT contiguous -- each channel's values are
    /// <c>n</c> separate planes -- which is exactly what an index map states and a
    /// hand-written kernel has to re-derive.
    /// </summary>
    public static CodegenProgram BatchNorm2D(int batch, int channels, int height, int width, double epsilon = 1e-5)
    {
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        RequireEpsilon(epsilon);
        int[] nchw = { batch, channels, height, width };
        int[] channel = { channels };
        double count = (double)batch * height * width;

        // Axis order (c | n, h, w): the channel is the only parallel axis.
        CodegenIterationSpace StatisticsSpace() => new(
            CodegenAxis.Parallel("c", channels), CodegenAxis.Reduce("n", batch),
            CodegenAxis.Reduce("h", height), CodegenAxis.Reduce("w", width));
        CodegenTensorBinding StatisticsInput() =>
            Bind(0, "x", nchw, Axis(1), Axis(0), Axis(2), Axis(3));

        var negMean = new CodegenKernelSpec(
            "batchnorm_negmean", StatisticsSpace(),
            new[] { StatisticsInput() },
            Bind(1, "negmean", channel, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Sum, reduceScale: -1.0 / count);

        var rstd = new CodegenKernelSpec(
            "batchnorm_rstd", StatisticsSpace(),
            new[] { StatisticsInput(), Bind(1, "negmean", channel, Axis(0)) },
            Bind(2, "rstd", channel, isOutput: true, Axis(0)),
            new[] { 0 }, CodegenReduceKind.Sum, reduceScale: 1.0 / count,
            activation: CodegenActivationKind.Rsqrt,
            preReduce: CodegenPreReduceOp.Square, preBiasInput: 1, reduceOffset: epsilon);

        var affine = new CodegenKernelSpec(
            "batchnorm_affine",
            new CodegenIterationSpace(
                CodegenAxis.Parallel("n", batch), CodegenAxis.Parallel("c", channels),
                CodegenAxis.Parallel("h", height), CodegenAxis.Parallel("w", width)),
            new[]
            {
                Bind(0, "x", nchw, Axis(0), Axis(1), Axis(2), Axis(3)),
                Bind(1, "negmean", channel, Axis(1)),
                Bind(2, "rstd", channel, Axis(1)),
                Bind(3, "gamma", channel, Axis(1)),
                Bind(4, "beta", channel, Axis(1)),
            },
            Bind(5, "y", nchw, isOutput: true, Axis(0), Axis(1), Axis(2), Axis(3)),
            new[] { 0 }, CodegenReduceKind.None,
            scaleInput: 2, preBiasInput: 1, postScaleInput: 3, postBiasInput: 4);

        return new CodegenProgram(
            new[] { negMean, rstd, affine },
            new long[] { channels, channels, (long)batch * channels * height * width },
            "batchnorm2d " + batch + "x" + channels + "x" + height + "x" + width);
    }

    /// <summary>
    /// GroupNorm over NC(spatial): statistics per (sample, group) over the group's channels
    /// and every spatial position, then a per-channel affine. The input is bound as
    /// <c>[n, groups, channels/groups, spatial]</c> -- the same bytes as <c>[n, c, spatial]</c>
    /// -- so a group is a plain axis rather than a division. InstanceNorm is the
    /// <c>groups == channels</c> case.
    /// </summary>
    public static CodegenProgram GroupNorm(int batch, int channels, int groups, int spatial, double epsilon = 1e-5)
    {
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (groups <= 0 || channels % groups != 0)
            throw new ArgumentOutOfRangeException(nameof(groups),
                "Groups must divide the channel count; " + groups + " does not divide " + channels + ".");
        if (spatial <= 0) throw new ArgumentOutOfRangeException(nameof(spatial));
        RequireEpsilon(epsilon);
        int perGroup = channels / groups;
        int[] grouped = { batch, groups, perGroup, spatial };
        int[] statistic = { batch, groups };
        int[] channel = { groups, perGroup };
        double count = (double)perGroup * spatial;

        CodegenIterationSpace StatisticsSpace() => new(
            CodegenAxis.Parallel("n", batch), CodegenAxis.Parallel("g", groups),
            CodegenAxis.Reduce("cg", perGroup), CodegenAxis.Reduce("s", spatial));
        CodegenTensorBinding StatisticsInput() =>
            Bind(0, "x", grouped, Axis(0), Axis(1), Axis(2), Axis(3));

        var negMean = new CodegenKernelSpec(
            "groupnorm_negmean", StatisticsSpace(),
            new[] { StatisticsInput() },
            Bind(1, "negmean", statistic, isOutput: true, Axis(0), Axis(1)),
            new[] { 0 }, CodegenReduceKind.Sum, reduceScale: -1.0 / count);

        var rstd = new CodegenKernelSpec(
            "groupnorm_rstd", StatisticsSpace(),
            new[] { StatisticsInput(), Bind(1, "negmean", statistic, Axis(0), Axis(1)) },
            Bind(2, "rstd", statistic, isOutput: true, Axis(0), Axis(1)),
            new[] { 0 }, CodegenReduceKind.Sum, reduceScale: 1.0 / count,
            activation: CodegenActivationKind.Rsqrt,
            preReduce: CodegenPreReduceOp.Square, preBiasInput: 1, reduceOffset: epsilon);

        var affine = new CodegenKernelSpec(
            "groupnorm_affine",
            new CodegenIterationSpace(
                CodegenAxis.Parallel("n", batch), CodegenAxis.Parallel("g", groups),
                CodegenAxis.Parallel("cg", perGroup), CodegenAxis.Parallel("s", spatial)),
            new[]
            {
                Bind(0, "x", grouped, Axis(0), Axis(1), Axis(2), Axis(3)),
                Bind(1, "negmean", statistic, Axis(0), Axis(1)),
                Bind(2, "rstd", statistic, Axis(0), Axis(1)),
                Bind(3, "gamma", channel, Axis(1), Axis(2)),
                Bind(4, "beta", channel, Axis(1), Axis(2)),
            },
            Bind(5, "y", grouped, isOutput: true, Axis(0), Axis(1), Axis(2), Axis(3)),
            new[] { 0 }, CodegenReduceKind.None,
            scaleInput: 2, preBiasInput: 1, postScaleInput: 3, postBiasInput: 4);

        return new CodegenProgram(
            new[] { negMean, rstd, affine },
            new long[] { (long)batch * groups, (long)batch * groups, (long)batch * channels * spatial },
            "groupnorm " + batch + "x" + channels + "/" + groups + "x" + spatial);
    }

    private static void RequireRowShape(int rows, int columns, double epsilon)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns <= 0) throw new ArgumentOutOfRangeException(nameof(columns));
        RequireEpsilon(epsilon);
    }

    private static void RequireEpsilon(double epsilon)
    {
        if (!(epsilon >= 0) || double.IsInfinity(epsilon))
            throw new ArgumentOutOfRangeException(nameof(epsilon),
                "Epsilon must be finite and non-negative; got " + epsilon + ".");
    }

    private static CodegenAffineExpr Axis(int axis) => CodegenAffineExpr.Axis(axis);

    private static CodegenTensorBinding Bind(
        int parameterIndex, string name, int[] shape, params CodegenAffineExpr[] map) =>
        new(parameterIndex, name, (int[])shape.Clone(), map);

    private static CodegenTensorBinding Bind(
        int parameterIndex, string name, int[] shape, bool isOutput, params CodegenAffineExpr[] map) =>
        new(parameterIndex, name, (int[])shape.Clone(), map, isOutput);
}
