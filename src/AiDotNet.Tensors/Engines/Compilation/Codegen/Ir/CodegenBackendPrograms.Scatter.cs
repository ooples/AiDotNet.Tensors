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
    /// input[col[p], j])</c>, and 0 for an empty row. <paramref name="entries"/> is the column-index buffer's length; an
    /// entry past the last row pointer belongs to no row.
    /// </summary>
    /// <remarks>
    /// A row's extent is data, so the entries are first assigned their rows -- <c>rowOf[p] = |{r : rowPtr[r + 1] &lt;=
    /// p}|</c>, an int32 sum over the row pointers -- and the maximum is then a segment maximum over the entries keyed by
    /// rowOf, gathering input through the column index. An empty row's maximum would be the reduction's identity, so
    /// the final pass reads it through a per-row gate that only a row with at least one entry passes, and the skipped
    /// term is exactly 0. The clamp is <paramref name="maximum"/>, the derived max_vectors.
    /// </remarks>
    public static CodegenProgram CsrSegmentedMax(int rows, int k, int n, int entries, CodegenLibraryFunction notEqualZero,
        CodegenLibraryFunction maximum) =>
        CsrSegmentedExtreme(rows, k, n, entries, notEqualZero, maximum, minimum: false);

    /// <summary>
    /// The CSR segmented minimum (#1441), CudaBackend.CsrSegmentedMin's contract: <see cref="CsrSegmentedMax"/> with the
    /// minimum (as <c>-max(-x)</c>, exact) and the clamp <c>min(x, 3.402823e38)</c> through <paramref name="minimum"/>,
    /// the derived min_vectors.
    /// </summary>
    public static CodegenProgram CsrSegmentedMin(int rows, int k, int n, int entries, CodegenLibraryFunction notEqualZero,
        CodegenLibraryFunction minimum) =>
        CsrSegmentedExtreme(rows, k, n, entries, notEqualZero, minimum, minimum: true);

    /// <summary>
    /// The CSR segmented standard deviation (#1441), CudaBackend.CsrSegmentedStdDev's contract: per row and column, the
    /// two-pass mean then <c>sqrt(sum (x - mean)^2 / count + epsilon)</c>, 0 for an empty row. The entries are keyed by
    /// their rows as in <see cref="CsrSegmentedMax"/>; the division and square root are <paramref name="divide"/> and
    /// <paramref name="sqrt"/>, the derived divide_vectors and sqrt_vector.
    /// </summary>
    public static CodegenProgram CsrSegmentedStdDev(int rows, int k, int n, int entries, float epsilon,
        CodegenLibraryFunction notEqualZero, CodegenLibraryFunction divide, CodegenLibraryFunction sqrt)
    {
        if (divide is null) throw new ArgumentNullException(nameof(divide));
        if (sqrt is null) throw new ArgumentNullException(nameof(sqrt));
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();
        CsrRowsOfEntries(rows, entries, notEqualZero, passes, temps);

        // count[r] as a float.
        passes.Add(new CodegenKernelSpec("csr_segmented_count_float",
            new CodegenIterationSpace(CodegenAxis.Parallel("r", rows)),
            new[]
            {
                new CodegenTensorBinding(0, "row_counts", new[] { rows }, new[] { A(0) },
                    elementType: CodegenElementType.Int32, integerValue: true),
            },
            Bind(1, "row_counts_f", new[] { rows }, isOutput: true), new[] { 0 }, CodegenReduceKind.None));
        temps.Add(rows);

        int[] cells = { rows, n };
        // sum[r, j] over row r's entries, in increasing entry order.
        passes.Add(CsrSegmentPass("csr_segmented_sum", rows, k, n, entries, "row_sum", CodegenReduceKind.Sum));
        temps.Add((long)rows * n);

        // mean[r, j] = divide(sum, count).
        var meanBody = CodegenExpr.Call(divide, Op(0), Op(1));
        var rowSpace = new CodegenIterationSpace(CodegenAxis.Parallel("r", rows), CodegenAxis.Parallel("j", n));
        passes.Add(new CodegenKernelSpec("csr_segmented_mean", rowSpace,
            new[] { Bind(0, "row_sum", cells), new CodegenTensorBinding(1, "row_counts_f", new[] { rows }, new[] { A(0) }) },
            Bind(2, "row_mean", cells, isOutput: true), meanBody.Operands(), CodegenReduceKind.None, body: meanBody));
        temps.Add((long)rows * n);

        // squares[r, j] = sum over row r's entries of (x - mean)^2.
        passes.Add(CsrSegmentPass("csr_segmented_squares", rows, k, n, entries, "row_squares", CodegenReduceKind.Sum,
            preBiasMean: true));
        temps.Add((long)rows * n);

        // out[r, j] = sqrt(divide(squares, count) + epsilon), through the non-empty gate.
        var stdBody = CodegenExpr.Multiply(
            CodegenExpr.Call(sqrt, CodegenExpr.Add(CodegenExpr.Call(divide, Op(0), Op(1)), CodegenExpr.Constant(epsilon))),
            Op(3));
        passes.Add(new CodegenKernelSpec("csr_segmented_stddev", rowSpace,
            new[]
            {
                Bind(0, "row_squares", cells),
                new CodegenTensorBinding(1, "row_counts_f", new[] { rows }, new[] { A(0) }),
                RowCountsIndex(2, rows),
                NonEmptyGate(3, 2, entries),
            },
            Bind(4, "out", cells, isOutput: true), stdBody.Operands(), CodegenReduceKind.None, body: stdBody));
        return new CodegenProgram(passes, temps, "csr segmented stddev (backend contract)");
    }

    private static CodegenProgram CsrSegmentedExtreme(int rows, int k, int n, int entries,
        CodegenLibraryFunction notEqualZero, CodegenLibraryFunction clamp, bool minimum)
    {
        if (clamp is null) throw new ArgumentNullException(nameof(clamp));
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();
        CsrRowsOfEntries(rows, entries, notEqualZero, passes, temps);

        int[] cells = { rows, n };
        passes.Add(CsrSegmentPass(minimum ? "csr_segmented_min_reduce" : "csr_segmented_max_reduce", rows, k, n,
            entries, "row_extreme", CodegenReduceKind.Max, negate: minimum));
        temps.Add((long)rows * n);

        // out[r, j] = clamp(extreme, -+3.402823e38), through the non-empty gate.
        double bound = minimum ? 3.402823e+38 : -3.402823e+38;
        var body = CodegenExpr.Multiply(CodegenExpr.Call(clamp, Op(0), CodegenExpr.Constant((float)bound)), Op(2));
        passes.Add(new CodegenKernelSpec(minimum ? "csr_segmented_min" : "csr_segmented_max",
            new CodegenIterationSpace(CodegenAxis.Parallel("r", rows), CodegenAxis.Parallel("j", n)),
            new[] { Bind(0, "row_extreme", cells), RowCountsIndex(1, rows), NonEmptyGate(2, 1, entries) },
            Bind(3, "out", cells, isOutput: true), body.Operands(), CodegenReduceKind.None, body: body));
        return new CodegenProgram(passes, temps, minimum ? "csr segmented min (backend contract)" : "csr segmented max (backend contract)");
    }

    // ones[p] (from the column indices, as the scatter mean builds its ones), rowOf[p] (int32) = |{r : rowPtr[r + 1] <=
    // p}|, and row_counts[r] (int32) = |{p : rowOf[p] == r}|.
    private static void CsrRowsOfEntries(int rows, int entries, CodegenLibraryFunction notEqualZero,
        List<CodegenKernelSpec> passes, List<long> temps)
    {
        if (rows <= 0 || entries <= 0) throw new ArgumentOutOfRangeException(nameof(rows), "CSR extents must be positive.");
        if (notEqualZero is null) throw new ArgumentNullException(nameof(notEqualZero));
        var nonZero = CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0));
        var oneBody = CodegenExpr.Add(nonZero, CodegenExpr.Subtract(CodegenExpr.Constant(1.0), nonZero));
        passes.Add(new CodegenKernelSpec("csr_segmented_ones", new CodegenIterationSpace(CodegenAxis.Parallel("p", entries)),
            new[] { Bind(0, "colIndices", new[] { entries }) }, Bind(1, "ones", new[] { entries }, isOutput: true),
            oneBody.Operands(), CodegenReduceKind.None, body: oneBody));
        temps.Add(entries);

        // rowOf[p]: rowPtr[r + 1] <= p exactly when rowPtr[r + 1] - (p + 1 - entries) lies in [0, entries).
        var rowOfSpace = new CodegenIterationSpace(CodegenAxis.Parallel("p", entries), CodegenAxis.Reduce("r", rows));
        var ends = new CodegenTensorBinding(1, "ones", new[] { entries }, new[] { CodegenAffineExpr.Const(0) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, new CodegenAffineExpr(new[] { new CodegenAffineTerm(1, 1) }, 1), entries,
                    CodegenIndexOutOfRange.Skip, floatIndices: true,
                    offset: new CodegenAffineExpr(new[] { new CodegenAffineTerm(0, 1) }, 1 - entries)),
            });
        passes.Add(new CodegenKernelSpec("csr_segmented_row_of", rowOfSpace,
            new[] { new CodegenTensorBinding(0, "rowPointers", new[] { rows + 1 }, new[] { A(1) }), ends },
            new CodegenTensorBinding(2, "row_of", new[] { entries }, new[] { A(0) }, isOutput: true,
                elementType: CodegenElementType.Int32, integerValue: true),
            new[] { 1 }, CodegenReduceKind.Sum));
        temps.Add(entries);

        // row_counts[r] = the entries keyed to r.
        var countSpace = new CodegenIterationSpace(CodegenAxis.Parallel("r", rows), CodegenAxis.Reduce("p", entries));
        passes.Add(new CodegenKernelSpec("csr_segmented_counts", countSpace,
            new[]
            {
                new CodegenTensorBinding(0, "row_of", new[] { entries }, new[] { A(1) }, elementType: CodegenElementType.Int32),
                new CodegenTensorBinding(1, "ones", new[] { 1, entries }, new[] { CodegenAffineExpr.Const(0), A(1) },
                    indirect: new CodegenIndirectIndex?[]
                    {
                        new CodegenIndirectIndex(0, A(1), 1, CodegenIndexOutOfRange.Skip, floatIndices: false, offset: A(0)),
                        null,
                    }),
            },
            new CodegenTensorBinding(2, "row_counts", new[] { rows }, new[] { A(0) }, isOutput: true,
                elementType: CodegenElementType.Int32, integerValue: true),
            new[] { 1 }, CodegenReduceKind.Sum));
        temps.Add(rows);
    }

    // A reduction over the entries keyed to each row: input gathered through the column index and admitted where
    // rowOf[p] == r. With preBiasMean the term is (x - mean[r, j])^2; with negate the reduction is a maximum of -x.
    private static CodegenKernelSpec CsrSegmentPass(string name, int rows, int k, int n, int entries, string output,
        CodegenReduceKind reduce, bool negate = false, bool preBiasMean = false)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("r", rows), CodegenAxis.Parallel("j", n), CodegenAxis.Reduce("p", entries));
        var inputs = new List<CodegenTensorBinding>
        {
            new CodegenTensorBinding(0, "row_of", new[] { entries }, new[] { A(2) }, elementType: CodegenElementType.Int32),
            new CodegenTensorBinding(1, "colIndices", new[] { entries }, new[] { A(2) }),
            new CodegenTensorBinding(2, "input", new[] { 1, k, n },
                new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Const(0), A(1) },
                indirect: new CodegenIndirectIndex?[]
                {
                    new CodegenIndirectIndex(0, A(2), 1, CodegenIndexOutOfRange.Skip, floatIndices: false, offset: A(0)),
                    new CodegenIndirectIndex(1, A(2), k, CodegenIndexOutOfRange.Skip, floatIndices: true),
                    null,
                }),
        };
        if (preBiasMean) inputs.Add(Bind(3, "row_mean", new[] { rows, n }));
        var outBinding = Bind(inputs.Count, output, new[] { rows, n }, isOutput: true);
        if (negate)
        {
            var body = CodegenExpr.Negate(Op(2));
            return new CodegenKernelSpec(name, space, inputs.ToArray(), outBinding, body.Operands(), reduce,
                reduceScale: -1.0, body: body);
        }
        return new CodegenKernelSpec(name, space, inputs.ToArray(), outBinding, new[] { 2 }, reduce,
            preReduce: preBiasMean ? CodegenPreReduceOp.Square : CodegenPreReduceOp.None,
            preBiasInput: preBiasMean ? 3 : null, preBiasScale: preBiasMean ? -1.0 : 1.0);
    }

    private static CodegenTensorBinding RowCountsIndex(int parameter, int rows) =>
        new(parameter, "row_counts", new[] { rows }, new[] { A(0) }, elementType: CodegenElementType.Int32);

    // ones[row_counts[r] - 1]: in range exactly for a row with at least one entry; a skipped read makes the term 0.
    private static CodegenTensorBinding NonEmptyGate(int parameter, int countsInput, int entries) =>
        new(parameter, "ones", new[] { entries }, new[] { CodegenAffineExpr.Const(0) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(countsInput, A(0), entries, CodegenIndexOutOfRange.Skip, floatIndices: false,
                    offset: CodegenAffineExpr.Const(1)),
            });

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
    /// The deterministic scatter reduction (#1441), in CudaBackend.ScatterReduce's contract for its sum (mode 0),
    /// maximum (2) and minimum (3): <c>out[o, t, i] = seed op (every source[o, s, i] with index[o, s, i] == t)</c>, the
    /// sum in increasing s, an index outside [0, dstDim) contributing nowhere. The caller binds <c>seed</c> and
    /// <c>out</c> to the same buffer. The maximum and minimum reduce the matches first (a NaN never wins, as fmaxf)
    /// and fold the seed in through <paramref name="extreme"/> (the derived max_vectors or min_vectors).
    /// </summary>
    public static CodegenProgram ScatterReduce(int outer, int sourceDim, int destinationDim, int inner, int mode,
        CodegenLibraryFunction? extreme)
    {
        if (outer <= 0 || sourceDim <= 0 || destinationDim <= 0 || inner <= 0)
            throw new ArgumentOutOfRangeException(nameof(outer), "Scatter reduce extents must be positive.");
        if (mode is not (0 or 2 or 3))
            throw new NotSupportedException("Scatter reduce mode " + mode + " (a product) has no declared reduction.");
        if (mode != 0 && extreme is null) throw new ArgumentNullException(nameof(extreme));
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