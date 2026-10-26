// Copyright (c) AiDotNet. All rights reserved.
// The rest of the row-wise scatter family as declared programs (#890 phase 4e, #1441), each in the exact contract of the
// hand-written kernel it replaces: the scatter maximum with its argmax, the backward passes of the add, mean, maximum
// and softmax, and the activation-layout ScatterMean with int32 counts. Like the scatter inversions before them, each
// reduces over the source rows (or the groups) in increasing order with no atomic, so every result is deterministic.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

public static partial class CodegenBackendPrograms
{
    /// <summary>
    /// The row-wise scatter maximum with its argmax (#1441), in CudaBackend.ScatterMaxWithArgmaxRows' contract:
    /// <c>out[g, j] = max of source[i, j] over the rows i with indices[i] == g</c> and <c>argmax[g, j]</c> the first
    /// such row to reach it, as a float; a group no row reaches holds -infinity and argmax -1, as the hand-written
    /// kernel leaves them. A row only replaces the running maximum when strictly greater, so a tie keeps the earlier
    /// row and a NaN never wins.
    /// </summary>
    public static CodegenProgram ScatterMaxArgmaxRows(int sourceRows, int innerSize, int outputRows)
    {
        var (space, indices, source) = ScatterInversion(sourceRows, outputRows, innerSize);
        int[] cells = { outputRows, innerSize };
        var output = new CodegenTensorBinding(2, "out", cells, new[] { A(0), A(1) }, isOutput: true);
        var argmax = new CodegenTensorBinding(3, "argmax", cells, new[] { A(0), A(1) }, isOutput: true);
        var pass = new CodegenKernelSpec("scatter_max_argmax_rows", space, new[] { indices, source }, output,
            new[] { 1 }, CodegenReduceKind.Max,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(argmax, CodegenExtraOutputKind.ArgMaxIndex, A(2), NoWinnerIndex: -1,
                    IndexAsFloat: true),
            });
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "row-wise scatter max with argmax (backend contract)");
    }

    /// <summary>
    /// The backward of the row-wise scatter maximum (#1441), in CudaBackend.ScatterMaxBackwardRows' contract:
    /// <c>gradSource[i, j] = gradOutput[g, j]</c> for the group g whose int32 <c>argmax[g, j]</c> is i, and 0 for a row
    /// no group chose. Written as an inversion over the groups, admitting group g where <c>argmax[g, j] - i</c> lands in
    /// [0, 1): a row belongs to one group, so at most one term is admitted and the sum is that term.
    /// </summary>
    public static CodegenProgram ScatterMaxBackwardRows(int sourceRows, int innerSize, int outputRows)
    {
        if (sourceRows <= 0 || innerSize <= 0 || outputRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceRows), "Scatter extents must be positive.");
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", sourceRows), CodegenAxis.Parallel("j", innerSize),
            CodegenAxis.Reduce("g", outputRows));
        var argmax = new CodegenTensorBinding(0, "argmax", new[] { outputRows * innerSize },
            new[] { new CodegenAffineExpr(new[] { new CodegenAffineTerm(2, innerSize), new CodegenAffineTerm(1, 1) }, 0) },
            elementType: CodegenElementType.Int32);
        var position = new CodegenAffineExpr(new[] { new CodegenAffineTerm(2, innerSize), new CodegenAffineTerm(1, 1) }, 0);
        var gradOutput = new CodegenTensorBinding(1, "gradOutput", new[] { 1, outputRows, innerSize },
            new[] { CodegenAffineExpr.Const(0), A(2), A(1) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, position, 1, CodegenIndexOutOfRange.Skip, floatIndices: false, offset: A(0)),
                null, null,
            });
        var output = new CodegenTensorBinding(2, "gradSource", new[] { sourceRows, innerSize }, new[] { A(0), A(1) },
            isOutput: true);
        var pass = new CodegenKernelSpec("scatter_max_backward_rows", space, new[] { argmax, gradOutput }, output,
            new[] { 1 }, CodegenReduceKind.Sum);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "row-wise scatter max backward (backend contract)");
    }

    /// <summary>
    /// The backward of the row-wise scatter mean (#1441), in CudaBackend.ScatterMeanBackwardRows' contract:
    /// <c>gradSource[i, j] = gradOutput[g, j] / max(counts[g], 1)</c> for g = indices[i], and 0 for a row outside
    /// [0, outputRows). The int32 counts are read as values; the divisor is <c>count - notEqual(count, 0) + 1</c>, the
    /// scatter mean's own clamp, and the division is <paramref name="divide"/> (the derived divide_vectors, so it
    /// rounds as the hand-written kernel's <c>/</c> does under the same compile flags).
    /// </summary>
    public static CodegenProgram ScatterMeanBackwardRows(int sourceRows, int innerSize, int outputRows,
        CodegenLibraryFunction notEqualZero, CodegenLibraryFunction divide)
    {
        if (sourceRows <= 0 || innerSize <= 0 || outputRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceRows), "Scatter extents must be positive.");
        if (notEqualZero is null) throw new ArgumentNullException(nameof(notEqualZero));
        if (divide is null) throw new ArgumentNullException(nameof(divide));

        // divisor[g] = count - notEqual(count, 0) + 1: the count, or 1 for an empty group.
        var counts = new CodegenTensorBinding(0, "counts", new[] { outputRows }, new[] { A(0) },
            elementType: CodegenElementType.Int32, integerValue: true);
        var clamp = CodegenExpr.Add(
            CodegenExpr.Subtract(Op(0), CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0))),
            CodegenExpr.Constant(1.0));
        var divisorPass = new CodegenKernelSpec("scatter_mean_backward_divisor",
            new CodegenIterationSpace(CodegenAxis.Parallel("g", outputRows)), new[] { counts },
            Bind(1, "divisor", new[] { outputRows }, isOutput: true), clamp.Operands(), CodegenReduceKind.None,
            body: clamp);

        // gradSource[i, j] = divide(gradOutput[indices[i], j], divisor[indices[i]]); a row outside the groups reads
        // neither, and the skipped term is the identity 0.
        var rowSpace = new CodegenIterationSpace(CodegenAxis.Parallel("i", sourceRows), CodegenAxis.Parallel("j", innerSize));
        var rowIndices = new CodegenTensorBinding(0, "indices", new[] { sourceRows }, new[] { A(0) },
            elementType: CodegenElementType.Int32);
        var gradOutput = new CodegenTensorBinding(1, "gradOutput", new[] { outputRows, innerSize },
            new[] { CodegenAffineExpr.Const(0), A(1) },
            indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(0), outputRows), null });
        var divisor = new CodegenTensorBinding(2, "divisor", new[] { outputRows }, new[] { CodegenAffineExpr.Const(0) },
            indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(0), outputRows) });
        var quotient = CodegenExpr.Call(divide, Op(1), Op(2));
        var outPass = new CodegenKernelSpec("scatter_mean_backward_rows", rowSpace,
            new[] { rowIndices, gradOutput, divisor },
            Bind(3, "gradSource", new[] { sourceRows, innerSize }, isOutput: true), quotient.Operands(),
            CodegenReduceKind.None, body: quotient);
        return new CodegenProgram(new[] { divisorPass, outPass }, new long[] { outputRows },
            "row-wise scatter mean backward (backend contract)");
    }

    /// <summary>
    /// The backward of the row-wise scatter softmax (#1441), in CudaBackend.ScatterSoftmaxBackwardRows' contract: for
    /// a row i in group g = indices[i], <c>gradSource[i, j] = output[i, j] * (gradOutput[i, j] - s[g, j])</c> with
    /// <c>s[g, j]</c> the sum of output * gradOutput over g's rows in increasing row order; a row outside [0, numGroups)
    /// writes exactly 0. The group sums are a scatter inversion of the products, gathered back per row; the final pass
    /// reads a per-group one through indices[i] (built with <paramref name="notEqualZero"/>, as the forward's are), so
    /// an out-of-group row's whole term is skipped rather than multiplied by a gathered zero.
    /// </summary>
    public static CodegenProgram ScatterSoftmaxBackwardRows(int sourceRows, int innerSize, int numGroups,
        CodegenLibraryFunction notEqualZero)
    {
        if (notEqualZero is null) throw new ArgumentNullException(nameof(notEqualZero));
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();
        int[] groupShape = { numGroups, innerSize }, rowShape = { sourceRows, innerSize };

        // s[g, j] = sum over g's rows of output * gradOutput.
        var (sumSpace, sumIndices, sumOutput) = ScatterInversion(sourceRows, numGroups, innerSize, "output");
        var sumGrad = new CodegenTensorBinding(2, "gradOutput", new[] { 1, sourceRows, innerSize },
            new[] { CodegenAffineExpr.Const(0), A(2), A(1) }, indirect: sumOutput.Indirect is { } ind
                ? new[] { ind[0], ind[1], ind[2] } : null);
        passes.Add(new CodegenKernelSpec("scatter_softmax_backward_groupsum", sumSpace,
            new[] { sumIndices, sumOutput, sumGrad },
            new CodegenTensorBinding(3, "groupsum", groupShape, new[] { A(0), A(1) }, isOutput: true),
            new[] { 1, 2 }, CodegenReduceKind.Sum));
        temps.Add((long)numGroups * innerSize);

        // row_sum[i, j] = s[indices[i], j] (0 outside the groups; the final pass skips those rows anyway).
        var rowSpace = new CodegenIterationSpace(CodegenAxis.Parallel("i", sourceRows), CodegenAxis.Parallel("j", innerSize));
        passes.Add(new CodegenKernelSpec("scatter_softmax_backward_gather", rowSpace,
            new[]
            {
                new CodegenTensorBinding(0, "indices", new[] { sourceRows }, new[] { A(0) },
                    elementType: CodegenElementType.Int32),
                new CodegenTensorBinding(1, "groupsum", groupShape, new[] { CodegenAffineExpr.Const(0), A(1) },
                    indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(0), numGroups), null }),
            },
            Bind(2, "row_sum", rowShape, isOutput: true), new[] { 1 }, CodegenReduceKind.None));
        temps.Add((long)sourceRows * innerSize);

        // group_ones[g] = a + (1 - a), a = notEqual(s[g, 0], 0): exactly 1 for every group, NaN included.
        var nonZero = CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0));
        var oneBody = CodegenExpr.Add(nonZero, CodegenExpr.Subtract(CodegenExpr.Constant(1.0), nonZero));
        passes.Add(new CodegenKernelSpec("scatter_softmax_backward_group_ones",
            new CodegenIterationSpace(CodegenAxis.Parallel("g", numGroups)),
            new[] { new CodegenTensorBinding(0, "groupsum", groupShape, new[] { A(0), CodegenAffineExpr.Const(0) }) },
            Bind(1, "group_ones", new[] { numGroups }, isOutput: true), oneBody.Operands(), CodegenReduceKind.None,
            body: oneBody));
        temps.Add(numGroups);

        // gradSource[i, j] = output * (gradOutput - row_sum) * group_ones[indices[i]].
        var body = CodegenExpr.Multiply(CodegenExpr.Multiply(Op(0), CodegenExpr.Subtract(Op(1), Op(2))), Op(4));
        passes.Add(new CodegenKernelSpec("scatter_softmax_backward_rows", rowSpace,
            new[]
            {
                Bind(0, "output", rowShape), Bind(1, "gradOutput", rowShape), Bind(2, "row_sum", rowShape),
                new CodegenTensorBinding(3, "indices", new[] { sourceRows }, new[] { A(0) },
                    elementType: CodegenElementType.Int32),
                new CodegenTensorBinding(4, "group_ones", new[] { numGroups }, new[] { CodegenAffineExpr.Const(0) },
                    indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(3, A(0), numGroups) }),
            },
            Bind(5, "gradSource", rowShape, isOutput: true), body.Operands(), CodegenReduceKind.None, body: body));
        return new CodegenProgram(passes, temps, "row-wise scatter softmax backward (backend contract)");
    }

    /// <summary>
    /// The activation-layout scatter mean (#1441), in CudaBackend.ScatterMean's contract: <c>counts[g]</c> (int32) is
    /// the number of source rows with indices[i] == g, and <c>output[g, f]</c> their sum divided by the count, or the
    /// (zero) sum for an empty group -- scatter_mean_divide's <c>if (cnt &gt; 0) output /= cnt</c>. The sums run in
    /// increasing row order, as scatter_mean_activation_deterministic does; the division is <paramref name="divide"/>
    /// by <c>count - notEqual(count, 0) + 1</c>, and dividing by 1 is exact.
    /// </summary>
    public static CodegenProgram ScatterMeanActivation(int sourceRows, int outputRows, int features,
        CodegenLibraryFunction notEqualZero, CodegenLibraryFunction divide)
    {
        if (notEqualZero is null) throw new ArgumentNullException(nameof(notEqualZero));
        if (divide is null) throw new ArgumentNullException(nameof(divide));
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();

        // ones[i] = a + (1 - a), a = notEqual(source[i, 0], 0).
        var nonZero = CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0));
        var oneBody = CodegenExpr.Add(nonZero, CodegenExpr.Subtract(CodegenExpr.Constant(1.0), nonZero));
        passes.Add(new CodegenKernelSpec("scatter_mean_activation_ones",
            new CodegenIterationSpace(CodegenAxis.Parallel("i", sourceRows)),
            new[] { new CodegenTensorBinding(0, "source", new[] { sourceRows, features }, new[] { A(0), CodegenAffineExpr.Const(0) }) },
            new CodegenTensorBinding(1, "ones", new[] { 1, sourceRows, 1 },
                new[] { CodegenAffineExpr.Const(0), A(0), CodegenAffineExpr.Const(0) }, isOutput: true),
            oneBody.Operands(), CodegenReduceKind.None, body: oneBody));
        temps.Add(sourceRows);

        // counts[g] (int32) = the scatter inversion of the ones.
        var (countSpace, countIndices, ones) = ScatterInversion(sourceRows, outputRows, 1, "ones");
        passes.Add(new CodegenKernelSpec("scatter_mean_activation_counts", countSpace, new[] { countIndices, ones },
            new CodegenTensorBinding(2, "counts", new[] { outputRows, 1 }, new[] { A(0), A(1) }, isOutput: true,
                elementType: CodegenElementType.Int32, integerValue: true),
            new[] { 1 }, CodegenReduceKind.Sum));

        // sums[g, f] = the scatter inversion of the source.
        var (sumSpace, sumIndices, source) = ScatterInversion(sourceRows, outputRows, features);
        passes.Add(new CodegenKernelSpec("scatter_mean_activation_sums", sumSpace, new[] { sumIndices, source },
            new CodegenTensorBinding(2, "sums", new[] { outputRows, features }, new[] { A(0), A(1) }, isOutput: true),
            new[] { 1 }, CodegenReduceKind.Sum));
        temps.Add((long)outputRows * features);

        // divisor[g] = count - notEqual(count, 0) + 1.
        var clamp = CodegenExpr.Add(
            CodegenExpr.Subtract(Op(0), CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0))),
            CodegenExpr.Constant(1.0));
        passes.Add(new CodegenKernelSpec("scatter_mean_activation_divisor",
            new CodegenIterationSpace(CodegenAxis.Parallel("g", outputRows)),
            new[]
            {
                new CodegenTensorBinding(0, "counts", new[] { outputRows }, new[] { A(0) },
                    elementType: CodegenElementType.Int32, integerValue: true),
            },
            Bind(1, "divisor", new[] { outputRows }, isOutput: true), clamp.Operands(), CodegenReduceKind.None,
            body: clamp));
        temps.Add(outputRows);

        // output[g, f] = divide(sums[g, f], divisor[g]).
        var quotient = CodegenExpr.Call(divide, Op(0), Op(1));
        passes.Add(new CodegenKernelSpec("scatter_mean_activation_out",
            new CodegenIterationSpace(CodegenAxis.Parallel("g", outputRows), CodegenAxis.Parallel("f", features)),
            new[]
            {
                Bind(0, "sums", new[] { outputRows, features }),
                new CodegenTensorBinding(1, "divisor", new[] { outputRows }, new[] { A(0) }),
            },
            Bind(2, "output", new[] { outputRows, features }, isOutput: true), quotient.Operands(),
            CodegenReduceKind.None, body: quotient));
        return new CodegenProgram(passes, temps, "activation-layout scatter mean with int32 counts (backend contract)");
    }
}