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
    /// Sampled dense-dense matrix multiplication (#1441), in CudaBackend.CsrSddmm's contract: for each stored entry p,
    /// <c>out[p] = sum over k of x[rowIndices[p], k] * y[colIndices[p], k]</c>, in increasing k. x has
    /// <paramref name="xRows"/> rows and y <paramref name="yRows"/>; an index outside them contributes nothing (the
    /// hand-written kernel read outside the matrix).
    /// </summary>
    public static CodegenProgram CsrSddmm(int nnz, int innerK, int xRows, int yRows)
    {
        if (nnz <= 0 || innerK <= 0 || xRows <= 0 || yRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(nnz), "SDDMM extents must be positive.");
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("p", nnz), CodegenAxis.Reduce("k", innerK));
        var rowIndices = new CodegenTensorBinding(0, "rowIndices", new[] { nnz }, new[] { A(0) },
            elementType: CodegenElementType.Int32);
        var colIndices = new CodegenTensorBinding(1, "colIndices", new[] { nnz }, new[] { A(0) },
            elementType: CodegenElementType.Int32);
        var x = new CodegenTensorBinding(2, "x", new[] { xRows, innerK }, new[] { CodegenAffineExpr.Const(0), A(1) },
            indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(0), xRows), null });
        var y = new CodegenTensorBinding(3, "y", new[] { yRows, innerK }, new[] { CodegenAffineExpr.Const(0), A(1) },
            indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(1, A(0), yRows), null });
        var output = new CodegenTensorBinding(4, "out", new[] { nnz }, new[] { A(0) }, isOutput: true);
        var pass = new CodegenKernelSpec("sddmm", space, new[] { rowIndices, colIndices, x, y }, output,
            new[] { 2, 3 }, CodegenReduceKind.Sum);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "sddmm (backend contract)");
    }

    /// <summary>
    /// The edge scatter-add of a graph layer (#1441), in CudaBackend.ScatterAddEdges' contract: <c>out[t, f] = sum over
    /// the edges e with target[e] == t of input[source[e], f] (times edgeValues[e] when weighted)</c>, in increasing e
    /// -- scatter_add_edges_deterministic's order. An inversion over the edges: the input is read through the source
    /// index and admitted only where <c>target[e] - t</c> lands in [0, 1), as the deterministic scatter-add admits.
    /// </summary>
    public static CodegenProgram ScatterAddEdges(int numNodes, int numEdges, int features, bool weighted)
    {
        if (numNodes <= 0 || numEdges <= 0 || features <= 0)
            throw new ArgumentOutOfRangeException(nameof(numNodes), "Edge scatter extents must be positive.");
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", numNodes), CodegenAxis.Parallel("f", features), CodegenAxis.Reduce("e", numEdges));
        var target = new CodegenTensorBinding(0, "targetIndices", new[] { numEdges }, new[] { A(2) },
            elementType: CodegenElementType.Int32);
        var source = new CodegenTensorBinding(1, "sourceIndices", new[] { numEdges }, new[] { A(2) },
            elementType: CodegenElementType.Int32);
        var admit = new CodegenIndirectIndex(0, A(2), 1, CodegenIndexOutOfRange.Skip, floatIndices: false, offset: A(0));
        var input = new CodegenTensorBinding(2, "input", new[] { 1, numNodes, features },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Const(0), A(1) },
            indirect: new CodegenIndirectIndex?[] { admit, new CodegenIndirectIndex(1, A(2), numNodes), null });
        var inputs = new List<CodegenTensorBinding> { target, source, input };
        var product = new List<int> { 2 };
        if (weighted)
        {
            inputs.Add(new CodegenTensorBinding(3, "edgeValues", new[] { 1, numEdges },
                new[] { CodegenAffineExpr.Const(0), A(2) }, indirect: new CodegenIndirectIndex?[] { admit, null }));
            product.Add(3);
        }
        var output = new CodegenTensorBinding(inputs.Count, "output", new[] { numNodes, features }, new[] { A(0), A(1) },
            isOutput: true);
        var pass = new CodegenKernelSpec(weighted ? "scatter_add_edges_weighted" : "scatter_add_edges", space,
            inputs.ToArray(), output, product.ToArray(), CodegenReduceKind.Sum);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "edge scatter-add (backend contract)");
    }

    /// <summary>
    /// The CSR segmented maximum (#1441), in CudaBackend.CsrSegmentedMax's contract: float-encoded column indices and
    /// row pointers (read as (int), truncating), <c>out[r, j] = max(-3.402823e38, max over row r's entries p of
    /// input[col[p], j])</c>, and 0 for an empty row. Each output walks its own row's entries (a data-dependent reduction
    /// extent), as the hand-written kernel loops; the accumulator starts where csr_segmented_max's does, and an empty
    /// row writes 0. A NaN never wins, as with fmaxf.
    /// </summary>
    public static CodegenProgram CsrSegmentedMax(int rows, int k, int n, int entries, CodegenLibraryFunction notEqualZero,
        CodegenLibraryFunction maximum) =>
        new(new[] { CsrSegmentPass("csr_segmented_max", rows, k, n, entries, "out", CodegenReduceKind.Max, negate: false,
            initial: (float)-3.402823e+38, emptyValue: 0.0) }, Array.Empty<long>(), "csr segmented max (backend contract)");

    /// <summary>
    /// The CSR segmented minimum (#1441), CudaBackend.CsrSegmentedMin's contract: <see cref="CsrSegmentedMax"/> with the
    /// minimum, as <c>-max(-x)</c> (exact), from csr_segmented_min's 3.402823e38 start.
    /// </summary>
    public static CodegenProgram CsrSegmentedMin(int rows, int k, int n, int entries, CodegenLibraryFunction notEqualZero,
        CodegenLibraryFunction minimum) =>
        new(new[] { CsrSegmentPass("csr_segmented_min", rows, k, n, entries, "out", CodegenReduceKind.Max, negate: true,
            initial: (float)-3.402823e+38, emptyValue: 0.0) }, Array.Empty<long>(), "csr segmented min (backend contract)");

    /// <summary>
    /// The CSR segmented standard deviation (#1441), CudaBackend.CsrSegmentedStdDev's contract: per row and column, the
    /// two-pass mean then <c>sqrt(sum (x - mean)^2 / count + epsilon)</c>, 0 for an empty row. The count is the row
    /// pointers' difference; the sums walk each row's own entries; the division, square root and the empty row's select
    /// are <paramref name="divide"/>, <paramref name="sqrt"/> and <paramref name="select"/> (the derived divide_vectors,
    /// sqrt_vector and where_select).
    /// </summary>
    public static CodegenProgram CsrSegmentedStdDev(int rows, int k, int n, int entries, float epsilon,
        CodegenLibraryFunction notEqualZero, CodegenLibraryFunction divide, CodegenLibraryFunction sqrt,
        CodegenLibraryFunction select)
    {
        if (divide is null) throw new ArgumentNullException(nameof(divide));
        if (sqrt is null) throw new ArgumentNullException(nameof(sqrt));
        if (select is null) throw new ArgumentNullException(nameof(select));
        var passes = new List<CodegenKernelSpec>();
        int[] cells = { rows, n };

        // count[r] = rowPointers[r + 1] - rowPointers[r] (exact: float-encoded integers).
        var countBody = CodegenExpr.Subtract(Op(1), Op(0));
        passes.Add(new CodegenKernelSpec("csr_segmented_count", new CodegenIterationSpace(CodegenAxis.Parallel("r", rows)),
            new[]
            {
                new CodegenTensorBinding(0, "rowPointers", new[] { rows + 1 }, new[] { A(0) }),
                new CodegenTensorBinding(1, "rowPointers", new[] { rows + 1 },
                    new[] { new CodegenAffineExpr(new[] { new CodegenAffineTerm(0, 1) }, 1) }),
            },
            Bind(2, "row_count", new[] { rows }, isOutput: true), countBody.Operands(), CodegenReduceKind.None, body: countBody));

        passes.Add(CsrSegmentPass("csr_segmented_sum", rows, k, n, entries, "row_sum", CodegenReduceKind.Sum));

        var rowSpace = new CodegenIterationSpace(CodegenAxis.Parallel("r", rows), CodegenAxis.Parallel("j", n));
        var meanBody = CodegenExpr.Call(divide, Op(0), Op(1));
        passes.Add(new CodegenKernelSpec("csr_segmented_mean", rowSpace,
            new[] { Bind(0, "row_sum", cells), new CodegenTensorBinding(1, "row_count", new[] { rows }, new[] { A(0) }) },
            Bind(2, "row_mean", cells, isOutput: true), meanBody.Operands(), CodegenReduceKind.None, body: meanBody));

        passes.Add(CsrSegmentPass("csr_segmented_squares", rows, k, n, entries, "row_squares", CodegenReduceKind.Sum,
            preBiasMean: true));

        // out = select(count, sqrt(divide(squares, count) + epsilon), 0): an empty row is exactly 0.
        var stdBody = CodegenExpr.Call(select, Op(1),
            CodegenExpr.Call(sqrt, CodegenExpr.Add(CodegenExpr.Call(divide, Op(0), Op(1)), CodegenExpr.Constant(epsilon))),
            CodegenExpr.Constant(0.0));
        passes.Add(new CodegenKernelSpec("csr_segmented_stddev", rowSpace,
            new[] { Bind(0, "row_squares", cells), new CodegenTensorBinding(1, "row_count", new[] { rows }, new[] { A(0) }) },
            Bind(2, "out", cells, isOutput: true), stdBody.Operands(), CodegenReduceKind.None, body: stdBody));
        return new CodegenProgram(passes, new long[] { rows, (long)rows * n, (long)rows * n, (long)rows * n },
            "csr segmented stddev (backend contract)");
    }

    // A reduction over row r's own entries [rowPointers[r], rowPointers[r + 1]), gathering input through the float column
    // index. With preBiasMean the term is (x - mean[r, j])^2; with negate the reduction is a maximum of -x.
    private static CodegenKernelSpec CsrSegmentPass(string name, int rows, int k, int n, int entries, string output,
        CodegenReduceKind reduce, bool negate = false, bool preBiasMean = false, double? initial = null,
        double? emptyValue = null)
    {
        if (rows <= 0 || k <= 0 || n <= 0 || entries <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "CSR extents must be positive.");
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("r", rows), CodegenAxis.Parallel("j", n), CodegenAxis.Reduce("p", entries));
        var inputs = new List<CodegenTensorBinding>
        {
            new CodegenTensorBinding(0, "rowPointers", new[] { rows + 1 }, new[] { A(0) }),
            new CodegenTensorBinding(1, "colIndices", new[] { entries }, new[] { A(2) }),
            new CodegenTensorBinding(2, "input", new[] { k, n }, new[] { CodegenAffineExpr.Const(0), A(1) },
                indirect: new CodegenIndirectIndex?[]
                {
                    new CodegenIndirectIndex(1, A(2), k, CodegenIndexOutOfRange.Skip, floatIndices: true), null,
                }),
        };
        if (preBiasMean) inputs.Add(Bind(3, "row_mean", new[] { rows, n }));
        var outBinding = Bind(inputs.Count, output, new[] { rows, n }, isOutput: true);
        var range = new CodegenReductionRange(2, 0, A(0), FloatIndices: true, Initial: initial, EmptyValue: emptyValue);
        if (negate)
        {
            var body = CodegenExpr.Negate(Op(2));
            return new CodegenKernelSpec(name, space, inputs.ToArray(), outBinding, body.Operands(), reduce,
                reduceScale: -1.0, body: body, reductionRange: range);
        }
        return new CodegenKernelSpec(name, space, inputs.ToArray(), outBinding, new[] { 2 }, reduce,
            preReduce: preBiasMean ? CodegenPreReduceOp.Square : CodegenPreReduceOp.None,
            preBiasInput: preBiasMean ? 3 : null, preBiasScale: preBiasMean ? -1.0 : 1.0, reductionRange: range);
    }
    /// <summary>
    /// The deterministic index write (#1441), in CudaBackend.IndexWrite's contract (parity210_index_write): for each
    /// destination slice t, the LAST j with <c>indices[j] == t</c> writes <c>out[o, t, i] = source[o, j, i]</c>
    /// (<paramref name="copy"/>) or <paramref name="fillValue"/>; a slice no index names keeps its value. The caller
    /// binds <c>seed</c> and <c>out</c> to the same buffer, as an in-place write means. The same program serves
    /// Parity210IndexCopy and Parity210IndexFill, whose duplicate indices race in the hand-written kernels; here the
    /// last one wins, as IndexWrite's does.
    /// </summary>
    /// <remarks>
    /// The last match is a maximum: <c>iota[j] = j</c> (a sum of ones over j' &lt; j, the condition an affine read that
    /// leaves the tensor otherwise), maximised over the matching j with its argmax, -1 when none. The final pass
    /// selects through <paramref name="select"/> (the derived where_select) on <c>argmax + 1</c>, which is 0 exactly
    /// when no index matched, so an untouched slice keeps its seed bit for bit.
    /// </remarks>
    public static CodegenProgram IndexWrite(int outer, int indexLength, int inner, int destinationAxis, bool copy,
        float fillValue, CodegenLibraryFunction notEqualZero, CodegenLibraryFunction select)
    {
        if (outer <= 0 || indexLength <= 0 || inner <= 0 || destinationAxis <= 0)
            throw new ArgumentOutOfRangeException(nameof(outer), "Index write extents must be positive.");
        if (notEqualZero is null) throw new ArgumentNullException(nameof(notEqualZero));
        if (select is null) throw new ArgumentNullException(nameof(select));
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();

        // one[0] = a + (1 - a), a = notEqual(seed[0], 0): exactly 1.
        var nonZero = CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0));
        var oneBody = CodegenExpr.Add(nonZero, CodegenExpr.Subtract(CodegenExpr.Constant(1.0), nonZero));
        long cells = (long)outer * destinationAxis * inner;
        passes.Add(new CodegenKernelSpec("index_write_one", new CodegenIterationSpace(CodegenAxis.Parallel("u", 1)),
            new[] { new CodegenTensorBinding(0, "seed", new[] { checked((int)cells) }, new[] { A(0) }) },
            Bind(1, "one", new[] { 1 }, isOutput: true), oneBody.Operands(), CodegenReduceKind.None, body: oneBody));
        temps.Add(1);

        // ones_j[j] = one[0].
        passes.Add(new CodegenKernelSpec("index_write_ones", new CodegenIterationSpace(CodegenAxis.Parallel("j", indexLength)),
            new[] { new CodegenTensorBinding(0, "one", new[] { 1 }, new[] { CodegenAffineExpr.Const(0) }) },
            Bind(1, "ones_j", new[] { indexLength }, isOutput: true), new[] { 0 }, CodegenReduceKind.None));
        temps.Add(indexLength);

        // iota[j] = sum over j' of ones_j[j - j' - 1], which is in range exactly for j' < j.
        passes.Add(new CodegenKernelSpec("index_write_iota",
            new CodegenIterationSpace(CodegenAxis.Parallel("j", indexLength), CodegenAxis.Reduce("k", indexLength)),
            new[]
            {
                new CodegenTensorBinding(0, "ones_j", new[] { indexLength },
                    new[] { new CodegenAffineExpr(new[] { new CodegenAffineTerm(0, 1), new CodegenAffineTerm(1, -1) }, -1) }),
            },
            Bind(1, "iota", new[] { indexLength }, isOutput: true), new[] { 0 }, CodegenReduceKind.Sum));
        temps.Add(indexLength);

        // last[t] = the largest j with indices[j] == t, -1 when none (an argmax over the matching iota).
        var lastSpace = new CodegenIterationSpace(CodegenAxis.Parallel("t", destinationAxis), CodegenAxis.Reduce("j", indexLength));
        var lastIota = new CodegenTensorBinding(1, "iota", new[] { 1, indexLength }, new[] { CodegenAffineExpr.Const(0), A(1) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, A(1), 1, CodegenIndexOutOfRange.Skip, floatIndices: false, offset: A(0)), null,
            });
        passes.Add(new CodegenKernelSpec("index_write_last", lastSpace,
            new[]
            {
                new CodegenTensorBinding(0, "indices", new[] { indexLength }, new[] { A(1) }, elementType: CodegenElementType.Int32),
                lastIota,
            },
            Bind(2, "last_value", new[] { destinationAxis }, isOutput: true), new[] { 1 }, CodegenReduceKind.Max,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(Bind(3, "last", new[] { destinationAxis }, isOutput: true),
                    CodegenExtraOutputKind.ArgMaxIndex, A(1), NoWinnerIndex: -1, IndexAsFloat: true),
            }));
        temps.Add(destinationAxis);
        temps.Add(destinationAxis);

        // out[o, t, i] = select(last[t] + 1, written, seed[o, t, i]).
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("o", outer), CodegenAxis.Parallel("t", destinationAxis),
            CodegenAxis.Parallel("i", inner));
        int[] shape = { outer, destinationAxis, inner };
        var inputs = new List<CodegenTensorBinding>
        {
            new CodegenTensorBinding(0, "seed", shape, new[] { A(0), A(1), A(2) }),
            new CodegenTensorBinding(1, "last", new[] { destinationAxis }, new[] { A(1) }),
        };
        CodegenExpr written;
        if (copy)
        {
            // source[o, last[t], i]; clamped rather than skipped, since an unmatched slice selects its seed anyway. last
            // is bound a second time as that index's source: one binding cannot be both an index and an operand.
            inputs.Add(new CodegenTensorBinding(2, "last", new[] { destinationAxis }, new[] { A(1) }));
            inputs.Add(new CodegenTensorBinding(3, "source", new[] { outer, indexLength, inner },
                new[] { A(0), CodegenAffineExpr.Const(0), A(2) },
                indirect: new CodegenIndirectIndex?[]
                {
                    null, new CodegenIndirectIndex(2, A(1), indexLength, CodegenIndexOutOfRange.Clamp, floatIndices: true), null,
                }));
            written = Op(3);
        }
        else
        {
            written = CodegenExpr.Constant(fillValue);
        }
        var body = CodegenExpr.Call(select, CodegenExpr.Add(Op(1), CodegenExpr.Constant(1.0)), written, Op(0));
        passes.Add(new CodegenKernelSpec(copy ? "index_write_copy" : "index_write_fill", space, inputs.ToArray(),
            new CodegenTensorBinding(inputs.Count, "out", shape, new[] { A(0), A(1), A(2) }, isOutput: true),
            body.Operands(), CodegenReduceKind.None, body: body));
        return new CodegenProgram(passes, temps, "deterministic index write (backend contract)");
    }

    /// <summary>
    /// CSR SpMM in double precision (#1441), in CudaBackend.CsrSpMMDouble's contract: values, dense and out are fp64,
    /// <c>out[r, j] = sum over row r's entries p of values[p] * dense[colIdx[p], j]</c> in increasing p, an empty row
    /// exactly 0. The same per-row walk over the stored entries as <see cref="CsrSpMM(int, int, int, int,
    /// CodegenLibraryFunction)"/>, computed in fp64 (PtxAffineEmitter.Plain).
    /// </summary>
    public static CodegenProgram CsrSpMMDouble(int rows, int cols, int n, int nnz)
    {
        if (rows <= 0 || cols <= 0 || n <= 0 || nnz <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "CSR SpMM extents must be positive.");
        return new CodegenProgram(new[] { CsrSpMMPass(rows, cols, n, nnz, CodegenElementType.Float64, bias: false) },
            Array.Empty<long>(), "fp64 csr spmm (backend contract)");
    }

    /// <summary>
    /// The masked scatter (#1441), in CudaBackend.Parity210MaskedScatter's contract: where the byte mask is set,
    /// <c>out[k] = source[prefix[k]]</c> when that position is inside the source; everywhere else out keeps its value.
    /// The mask is read as uint8 values; whether the prefix position is inside the source is its own pass (a ones read
    /// through it, 0 when skipped), and the write selects through <paramref name="select"/> (the derived where_select)
    /// on <c>mask * valid</c>, so an unwritten element keeps its seed bit for bit. The caller binds <c>seed</c> and
    /// <c>out</c> to the same buffer.
    /// </summary>
    public static CodegenProgram MaskedScatter(int total, int sourceLength, CodegenLibraryFunction notEqualZero,
        CodegenLibraryFunction select)
    {
        if (total <= 0 || sourceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(total), "Masked scatter extents must be positive.");
        if (notEqualZero is null) throw new ArgumentNullException(nameof(notEqualZero));
        if (select is null) throw new ArgumentNullException(nameof(select));
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();

        // ones[s] = a + (1 - a), a = notEqual(source[s], 0).
        var nonZero = CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0));
        var oneBody = CodegenExpr.Add(nonZero, CodegenExpr.Subtract(CodegenExpr.Constant(1.0), nonZero));
        passes.Add(new CodegenKernelSpec("masked_scatter_ones", new CodegenIterationSpace(CodegenAxis.Parallel("s", sourceLength)),
            new[] { Bind(0, "source", new[] { sourceLength }) }, Bind(1, "ones", new[] { sourceLength }, isOutput: true),
            oneBody.Operands(), CodegenReduceKind.None, body: oneBody));
        temps.Add(sourceLength);

        // valid[k] = ones[prefix[k]]: 1 inside the source, 0 (the skipped read) outside it.
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("k", total));
        var prefix = new CodegenTensorBinding(0, "prefix", new[] { total }, new[] { A(0) }, elementType: CodegenElementType.Int32);
        passes.Add(new CodegenKernelSpec("masked_scatter_valid", space,
            new[]
            {
                prefix,
                new CodegenTensorBinding(1, "ones", new[] { sourceLength }, new[] { CodegenAffineExpr.Const(0) },
                    indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(0), sourceLength) }),
            },
            Bind(2, "valid", new[] { total }, isOutput: true), new[] { 1 }, CodegenReduceKind.None));
        temps.Add(total);

        // out[k] = select(mask[k] * valid[k], source[prefix[k]], seed[k]); the source read is clamped, as an
        // unwritten element selects its seed anyway.
        var body = CodegenExpr.Call(select, CodegenExpr.Multiply(Op(1), Op(2)), Op(4), Op(0));
        passes.Add(new CodegenKernelSpec("masked_scatter", space,
            new[]
            {
                Bind(0, "seed", new[] { total }),
                new CodegenTensorBinding(1, "mask", new[] { total }, new[] { A(0) }, elementType: CodegenElementType.UInt8,
                    integerValue: true),
                Bind(2, "valid", new[] { total }),
                new CodegenTensorBinding(3, "prefix", new[] { total }, new[] { A(0) }, elementType: CodegenElementType.Int32),
                new CodegenTensorBinding(4, "source", new[] { sourceLength }, new[] { CodegenAffineExpr.Const(0) },
                    indirect: new CodegenIndirectIndex?[]
                    {
                        new CodegenIndirectIndex(3, A(0), sourceLength, CodegenIndexOutOfRange.Clamp),
                    }),
            },
            Bind(5, "out", new[] { total }, isOutput: true), body.Operands(), CodegenReduceKind.None, body: body));
        return new CodegenProgram(passes, temps, "masked scatter (backend contract)");
    }

    /// <summary>
    /// The deterministic index add (#1441), in CudaBackend.Parity210IndexAdd's contract: <c>out[o, t, i] +=
    /// sum over the j with indices[j] == t of source[o, j, i]</c>, in increasing j, with no atomic. The caller binds
    /// <c>seed</c> and <c>out</c> to the same buffer; an index outside [0, destinationAxis) adds nowhere.
    /// </summary>
    public static CodegenProgram IndexAdd(int outer, int destinationAxis, int inner, int indexLength)
    {
        if (outer <= 0 || destinationAxis <= 0 || inner <= 0 || indexLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(outer), "Index add extents must be positive.");
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("o", outer), CodegenAxis.Parallel("t", destinationAxis),
            CodegenAxis.Parallel("i", inner), CodegenAxis.Reduce("j", indexLength));
        int[] shape = { outer, destinationAxis, inner };
        var pass = new CodegenKernelSpec("index_add", space,
            new[]
            {
                new CodegenTensorBinding(0, "indices", new[] { indexLength }, new[] { A(3) }, elementType: CodegenElementType.Int32),
                new CodegenTensorBinding(1, "source", new[] { outer, 1, indexLength, inner },
                    new[] { A(0), CodegenAffineExpr.Const(0), A(3), A(2) },
                    indirect: new CodegenIndirectIndex?[]
                    {
                        null,
                        new CodegenIndirectIndex(0, A(3), 1, CodegenIndexOutOfRange.Skip, floatIndices: false, offset: A(1)),
                        null, null,
                    }),
                new CodegenTensorBinding(2, "seed", shape, new[] { A(0), A(1), A(2) }),
            },
            new CodegenTensorBinding(3, "out", shape, new[] { A(0), A(1), A(2) }, isOutput: true),
            new[] { 1 }, CodegenReduceKind.Sum, biasInput: 2);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "deterministic index add (backend contract)");
    }

    /// <summary>
    /// The deterministic scatter reduction (#1441), in CudaBackend.ScatterReduce's contract for every mode -- sum (0),
    /// product (1), maximum (2), minimum (3): <c>out[o, t, i] = seed op (every source[o, s, i] with index[o, s, i] ==
    /// t)</c>, in increasing s, an index outside [0, dstDim) contributing nowhere. The caller binds <c>seed</c> and
    /// <c>out</c> to the same buffer. The sum adds the seed after the matches and the product multiplies it in after
    /// them, where the hand-written compare-and-swap folds each match into the seed in arrival order, so the two can
    /// differ in rounding. The maximum and minimum reduce the matches first (a NaN never wins, as fmaxf) and fold the
    /// seed in through <paramref name="extreme"/> (the derived max_vectors or min_vectors).
    /// </summary>
    public static CodegenProgram ScatterReduce(int outer, int sourceDim, int destinationDim, int inner, int mode,
        CodegenLibraryFunction? extreme)
    {
        if (outer <= 0 || sourceDim <= 0 || destinationDim <= 0 || inner <= 0)
            throw new ArgumentOutOfRangeException(nameof(outer), "Scatter reduce extents must be positive.");
        if (mode is < 0 or > 3) throw new ArgumentOutOfRangeException(nameof(mode), "Scatter reduce modes are 0..3.");
        if (mode >= 2 && extreme is null) throw new ArgumentNullException(nameof(extreme));
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("o", outer), CodegenAxis.Parallel("t", destinationDim),
            CodegenAxis.Parallel("i", inner), CodegenAxis.Reduce("s", sourceDim));
        int[] shape = { outer, destinationDim, inner };
        // index[o, s, i] sits at (o * sourceDim + s) * inner + i.
        var position = new CodegenAffineExpr(new[]
        {
            new CodegenAffineTerm(0, sourceDim * inner), new CodegenAffineTerm(3, inner), new CodegenAffineTerm(2, 1),
        }, 0);
        var index = new CodegenTensorBinding(0, "index", new[] { outer * sourceDim * inner }, new[] { position },
            elementType: CodegenElementType.Int32);
        var source = new CodegenTensorBinding(1, "source", new[] { outer, 1, sourceDim, inner },
            new[] { A(0), CodegenAffineExpr.Const(0), A(3), A(2) },
            indirect: new CodegenIndirectIndex?[]
            {
                null,
                new CodegenIndirectIndex(0, position, 1, CodegenIndexOutOfRange.Skip, floatIndices: false, offset: A(1)),
                null, null,
            });
        var seed = new CodegenTensorBinding(2, "seed", shape, new[] { A(0), A(1), A(2) });
        if (mode == 0)
        {
            var sum = new CodegenKernelSpec("scatter_reduce_sum", space, new[] { index, source, seed },
                new CodegenTensorBinding(3, "out", shape, new[] { A(0), A(1), A(2) }, isOutput: true),
                new[] { 1 }, CodegenReduceKind.Sum, biasInput: 2);
            return new CodegenProgram(new[] { sum }, Array.Empty<long>(), "deterministic scatter reduce sum (backend contract)");
        }
        if (mode == 1)
        {
            var product = new CodegenKernelSpec("scatter_reduce_prod", space, new[] { index, source, seed },
                new CodegenTensorBinding(3, "out", shape, new[] { A(0), A(1), A(2) }, isOutput: true),
                new[] { 1 }, CodegenReduceKind.Product, scaleInput: 2);
            return new CodegenProgram(new[] { product }, Array.Empty<long>(), "deterministic scatter reduce product (backend contract)");
        }

        bool minimum = mode == 3;
        var matches = minimum
            ? new CodegenKernelSpec("scatter_reduce_min_matches", space, new[] { index, source },
                new CodegenTensorBinding(2, "matches", shape, new[] { A(0), A(1), A(2) }, isOutput: true),
                new[] { 1 }, CodegenReduceKind.Max, reduceScale: -1.0, body: CodegenExpr.Negate(Op(1)))
            : new CodegenKernelSpec("scatter_reduce_max_matches", space, new[] { index, source },
                new CodegenTensorBinding(2, "matches", shape, new[] { A(0), A(1), A(2) }, isOutput: true),
                new[] { 1 }, CodegenReduceKind.Max);
        var fold = CodegenExpr.Call(extreme!, Op(0), Op(1));
        var cellSpace = new CodegenIterationSpace(CodegenAxis.Parallel("o", outer), CodegenAxis.Parallel("t", destinationDim),
            CodegenAxis.Parallel("i", inner));
        var combine = new CodegenKernelSpec(minimum ? "scatter_reduce_min" : "scatter_reduce_max", cellSpace,
            new[] { Bind(0, "seed", shape), Bind(1, "matches", shape) }, Bind(2, "out", shape, isOutput: true),
            fold.Operands(), CodegenReduceKind.None, body: fold);
        return new CodegenProgram(new[] { matches, combine }, new long[] { outer * (long)destinationDim * inner },
            minimum ? "deterministic scatter reduce min (backend contract)" : "deterministic scatter reduce max (backend contract)");
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