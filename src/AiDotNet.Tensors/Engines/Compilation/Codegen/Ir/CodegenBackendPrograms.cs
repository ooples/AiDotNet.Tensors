// Copyright (c) AiDotNet. All rights reserved.
// The declared normalisation programs in the EXACT contract of the hand-written backend kernels they replace
// (#890 phase 4a), so a backend can route an existing call to them with no change to its callers.
//
// The programs keep their own internal statistics (a negated mean, rstd); these adapters add the row passes that
// produce what the hand-written kernels save and read:
//
//   LayerNorm / GroupNorm / BatchNorm   saveMean = mean, saveInvVar = rstd
//   RmsNorm                             saveRms  = rms = 1 / rstd
//   BatchNorm (training)                runningMean += momentum * (mean - runningMean),
//                                       runningVar  += momentum * (unbiased var - runningVar)
//
// and, in front of each backward program, the passes that turn the saved statistics back into what it reads. A row
// pass inside a fused kernel is a few instructions per row, so the contract costs nothing measurable.
//
// Every tensor a contract names is one the backend binds to a caller's buffer; any other name in the program is an
// intermediate the backend owns.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>How an attention call masks its scores (CodegenBackendPrograms.ScaledDotProductAttention).</summary>
public enum CodegenAttentionMask
{
    /// <summary>No mask.</summary>
    None,
    /// <summary>One [seqQ, seqK] mask every batch and head shares.</summary>
    Shared,
    /// <summary>A [batch, heads, seqQ, seqK] mask.</summary>
    Full,
}

/// <summary>Declared programs shaped as the hand-written backend normalisation kernels.</summary>
public static partial class CodegenBackendPrograms
{
    private static CodegenExpr Op(int input) => CodegenExpr.Operand(input);

    private static CodegenTensorBinding Bind(int parameter, string name, int[] shape, bool isOutput = false)
    {
        var map = new CodegenAffineExpr[shape.Length];
        for (int d = 0; d < map.Length; d++) map[d] = CodegenAffineExpr.Axis(d);
        return new CodegenTensorBinding(parameter, name, (int[])shape.Clone(), map, isOutput);
    }

    private static CodegenIterationSpace RowSpace(int[] shape)
    {
        var axes = new CodegenAxis[shape.Length];
        for (int d = 0; d < axes.Length; d++) axes[d] = CodegenAxis.Parallel("r" + d, shape[d]);
        return new CodegenIterationSpace(axes);
    }

    /// <summary>A per-row pass computing <paramref name="body"/> over the named per-row inputs.</summary>
    private static CodegenKernelSpec RowPass(string name, int[] shape, string[] inputs, string output, CodegenExpr body,
        CodegenActivationKind activation = CodegenActivationKind.None)
    {
        var bindings = new CodegenTensorBinding[inputs.Length];
        for (int i = 0; i < inputs.Length; i++) bindings[i] = Bind(i, inputs[i], shape);
        return new CodegenKernelSpec(name, RowSpace(shape), bindings,
            Bind(inputs.Length, output, shape, isOutput: true), body.Operands(), CodegenReduceKind.None,
            activation: activation, body: body);
    }

    private static CodegenProgram Append(CodegenProgram program, params CodegenKernelSpec[] extra)
    {
        var passes = new List<CodegenKernelSpec>(program.Passes);
        var temps = new List<long>(program.TempElements);
        foreach (var pass in extra) { passes.Add(pass); temps.Add(pass.Output.ElementCount); }
        return new CodegenProgram(passes, temps, program.Description + " (backend contract)");
    }

    private static CodegenProgram Prepend(CodegenProgram program, params CodegenKernelSpec[] first)
    {
        var passes = new List<CodegenKernelSpec>(first);
        var temps = new List<long>();
        foreach (var pass in first) temps.Add(pass.Output.ElementCount);
        passes.AddRange(program.Passes);
        temps.AddRange(program.TempElements);
        return new CodegenProgram(passes, temps, program.Description + " (backend contract)");
    }

    /// <summary>
    /// CudaBackend.LayerNorm: sources x, gamma, beta; outputs y, mean (saveMean), rstd (saveInvVar).
    /// </summary>
    // ---- linear layers (#890 phase 4c), in CudaBackend.FusedLinear*'s contract: x [batch, in], W [in, out] -------

    /// <summary><c>y[b, j] = act(sum_k x[b, k] W[k, j] + bias[j])</c>: one contraction with a bias and activation epilogue.</summary>
    public static CodegenProgram FusedLinear(int batch, int inFeatures, int outFeatures, CodegenActivationKind activation)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("j", outFeatures), CodegenAxis.Reduce("k", inFeatures));
        const int B = 0, J = 1, K = 2;
        var x = new CodegenTensorBinding(0, "x", new[] { batch, inFeatures }, new[] { A(B), A(K) });
        var w = new CodegenTensorBinding(1, "w", new[] { inFeatures, outFeatures }, new[] { A(K), A(J) });
        var bias = new CodegenTensorBinding(2, "bias", new[] { outFeatures }, new[] { A(J) });
        var y = new CodegenTensorBinding(3, "y", new[] { batch, outFeatures }, new[] { A(B), A(J) }, isOutput: true);
        var pass = new CodegenKernelSpec("fused_linear", space, new[] { x, w, bias }, y, new[] { 0, 1 },
            CodegenReduceKind.Sum, biasInput: 2, activation: activation);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "fused linear (backend contract)");
    }

    /// <summary>
    /// The backward of <see cref="FusedLinear"/>: <c>dpre = derivative(dy, saved)</c> -- the activation's backward,
    /// <paramref name="derivative"/> called with the gradient and the saved pre-activation or output -- then
    /// <c>dx = dpre W^T</c>, <c>dW = x^T dpre</c> and <c>dbias = sum_b dpre</c>, each overwriting its gradient.
    /// </summary>
    public static CodegenProgram FusedLinearBackward(int batch, int inFeatures, int outFeatures,
        CodegenLibraryFunction derivative)
    {
        if (derivative is null) throw new ArgumentNullException(nameof(derivative));
        var activationGrad = RowPass("fused_linear_dpre", new[] { batch, outFeatures }, new[] { "dy", "saved" }, "dpre",
            CodegenExpr.Call(derivative, Op(0), Op(1)));

        var dxSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("i", inFeatures), CodegenAxis.Reduce("j", outFeatures));
        var dx = new CodegenKernelSpec("fused_linear_dx", dxSpace,
            new[]
            {
                new CodegenTensorBinding(0, "dpre", new[] { batch, outFeatures }, new[] { A(0), A(2) }),
                new CodegenTensorBinding(1, "w", new[] { inFeatures, outFeatures }, new[] { A(1), A(2) }),
            },
            new CodegenTensorBinding(2, "dx", new[] { batch, inFeatures }, new[] { A(0), A(1) }, isOutput: true),
            new[] { 0, 1 }, CodegenReduceKind.Sum);

        var dwSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", inFeatures), CodegenAxis.Parallel("j", outFeatures), CodegenAxis.Reduce("b", batch));
        var dw = new CodegenKernelSpec("fused_linear_dw", dwSpace,
            new[]
            {
                new CodegenTensorBinding(0, "x", new[] { batch, inFeatures }, new[] { A(2), A(0) }),
                new CodegenTensorBinding(1, "dpre", new[] { batch, outFeatures }, new[] { A(2), A(1) }),
            },
            new CodegenTensorBinding(2, "dw", new[] { inFeatures, outFeatures }, new[] { A(0), A(1) }, isOutput: true),
            new[] { 0, 1 }, CodegenReduceKind.Sum);

        var dbSpace = new CodegenIterationSpace(CodegenAxis.Parallel("j", outFeatures), CodegenAxis.Reduce("b", batch));
        var db = new CodegenKernelSpec("fused_linear_db", dbSpace,
            new[] { new CodegenTensorBinding(0, "dpre", new[] { batch, outFeatures }, new[] { A(1), A(0) }) },
            new CodegenTensorBinding(1, "dbias", new[] { outFeatures }, new[] { A(0) }, isOutput: true),
            new[] { 0 }, CodegenReduceKind.Sum);

        return new CodegenProgram(new[] { activationGrad, dx, dw, db }, new long[] { (long)batch * outFeatures },
            "fused linear backward (backend contract)");
    }

    /// <summary>
    /// <c>C[m, n] = sum_k op(A)[m, k] op(B)[k, n]</c>, row-major, in CudaBackend.Gemm's contract (alpha 1, beta 0):
    /// op(A) is <c>A[M, K]</c>, or <c>A[K, M]</c> transposed when <paramref name="transA"/>; op(B) is <c>B[K, N]</c>, or
    /// <c>B[N, K]</c> transposed when <paramref name="transB"/>. A <paramref name="batch"/> above 1 adds a leading
    /// batch axis to all three tensors (CudaBackend.BatchedGemm's packed strides).
    /// </summary>
    public static CodegenProgram Gemm(int batch, int m, int n, int k, bool transA, bool transB)
    {
        if (batch <= 0 || m <= 0 || n <= 0 || k <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GEMM extents must be positive.");
        bool batched = batch > 1;
        int lead = batched ? 1 : 0;
        var axes = new List<CodegenAxis>();
        if (batched) axes.Add(CodegenAxis.Parallel("g", batch));
        axes.Add(CodegenAxis.Parallel("m", m));
        axes.Add(CodegenAxis.Parallel("n", n));
        axes.Add(CodegenAxis.Reduce("k", k));
        int M = lead, N = lead + 1, K = lead + 2;

        CodegenTensorBinding Matrix(int parameter, string name, int rows, int columns, int rowAxis, int columnAxis,
            bool isOutput = false)
        {
            var shape = batched ? new[] { batch, rows, columns } : new[] { rows, columns };
            var map = batched ? new[] { A(0), A(rowAxis), A(columnAxis) } : new[] { A(rowAxis), A(columnAxis) };
            return new CodegenTensorBinding(parameter, name, shape, map, isOutput);
        }

        var a = transA ? Matrix(0, "a", k, m, K, M) : Matrix(0, "a", m, k, M, K);
        var b = transB ? Matrix(1, "b", n, k, N, K) : Matrix(1, "b", k, n, K, N);
        var c = Matrix(2, "c", m, n, M, N, isOutput: true);
        var pass = new CodegenKernelSpec(batched ? "batched_gemm" : "gemm", new CodegenIterationSpace(axes.ToArray()),
            new[] { a, b }, c, new[] { 0, 1 }, CodegenReduceKind.Sum);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "gemm (backend contract)");
    }

    // ---- gather (#890 phase 4e), in CudaBackend.Embedding's contract -----------------------------------------------

    /// <summary>
    /// <c>out[t, e] = table[ids[t], e]</c>: ids int32 [tokens], table [vocabulary, width]. An id outside
    /// [0, vocabulary) yields a zero row (the hand-written kernel read outside the table).
    /// </summary>
    public static CodegenProgram Embedding(int tokens, int vocabulary, int width)
    {
        if (tokens <= 0 || vocabulary <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), "Embedding extents must be positive.");
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("t", tokens), CodegenAxis.Parallel("e", width));
        var ids = new CodegenTensorBinding(0, "ids", new[] { tokens }, new[] { A(0) }, elementType: CodegenElementType.Int32);
        var table = new CodegenTensorBinding(1, "table", new[] { vocabulary, width },
            new[] { CodegenAffineExpr.Const(0), A(1) },
            indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(0), vocabulary), null });
        var output = new CodegenTensorBinding(2, "out", new[] { tokens, width }, new[] { A(0), A(1) }, isOutput: true);
        var pass = new CodegenKernelSpec("embedding", space, new[] { ids, table }, output, new[] { 1 }, CodegenReduceKind.None);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "embedding (backend contract)");
    }

    /// <summary>
    /// The deterministic scatter-add (#890 phase 4e), in CudaBackend's deterministic-scatter contract:
    /// <c>out[d, f] = sum of source[i, f] over every i with indices[i] == d</c>, summed in increasing i, for d in
    /// [0, destinations). An index outside that range contributes nowhere, which is what a -1 sentinel means.
    /// </summary>
    /// <remarks>
    /// Written as a GATHER over destinations rather than a scatter over sources, which is what makes it
    /// deterministic: a scatter has to accumulate through <c>red.global.add.f32</c>, and an atomic float add sums in
    /// whatever order the warps arrive, so it returns a different answer per run. Here each destination reduces over
    /// the sources in reduce-axis order and source i contributes only where
    /// <c>indices[i] - d</c> lands in [0, 1) -- see <see cref="CodegenIndirectIndex.Offset"/>. No atomic is emitted
    /// and the result is bit-identical run to run. The cost is destinations x count, as the repo's own scanning
    /// deterministic kernels already pay; the sorted path remains the faster one for a large destination count.
    /// </remarks>
    public static CodegenProgram ScatterAddDeterministic(int count, int destinations, int features)
    {
        // The source's leading extent-1 dimension exists only to carry the test: its index is indices[i] - d, in
        // range for the one destination that matches and skipped for the rest.
        var (space, indices, source) = ScatterInversion(count, destinations, features);
        var output = new CodegenTensorBinding(2, "out", new[] { destinations, features }, new[] { A(0), A(1) },
            isOutput: true);
        var pass = new CodegenKernelSpec("scatter_add_deterministic", space, new[] { indices, source }, output,
            new[] { 1 }, CodegenReduceKind.Sum);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "deterministic scatter-add (backend contract)");
    }

    /// <summary>
    /// The deterministic segment sum (#890 phase 4e): <c>out[s, f] = sum of source[i, f] over every i whose segment
    /// id is s</c>, in increasing i. The same inversion as <see cref="ScatterAddDeterministic"/> -- a segment sum IS
    /// a scatter-add whose indices are segment ids -- so it is deterministic for the same reason, and it accepts
    /// unsorted ids, which an offsets-based segment kernel cannot.
    /// </summary>
    public static CodegenProgram SegmentSum(int count, int segments, int features) =>
        ScatterAddDeterministic(count, segments, features);

    // The three scatter programs share one shape: destinations x features in parallel, the sources reduced, and
    // source i admitted only where indices[i] - d lands in [0, 1).
    private static (CodegenIterationSpace Space, CodegenTensorBinding Indices, CodegenTensorBinding Source)
        ScatterInversion(int count, int destinations, int features, string sourceName = "source")
    {
        if (count <= 0 || destinations <= 0 || features <= 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Scatter extents must be positive.");
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("d", destinations), CodegenAxis.Parallel("f", features),
            CodegenAxis.Reduce("i", count));
        var indices = new CodegenTensorBinding(0, "indices", new[] { count }, new[] { A(2) },
            elementType: CodegenElementType.Int32);
        var source = new CodegenTensorBinding(1, sourceName, new[] { 1, count, features },
            new[] { CodegenAffineExpr.Const(0), A(2), A(1) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, A(2), 1, CodegenIndexOutOfRange.Skip, floatIndices: false, offset: A(0)),
                null, null,
            });
        return (space, indices, source);
    }

    /// <summary>
    /// The row-wise scatter mean (#1441), in CudaBackend.ScatterMeanRowsWithCounts' contract:
    /// <c>counts[g] = |{i : indices[i] == g}|</c> and <c>out[g, j] = sum of source[i, j] over those i, times
    /// 1 / counts[g]</c>, 0 for a group nothing targets. Four passes, with no atomic: a tensor of ones; the counts as
    /// the scatter inversion of those ones; the sums as the inversion of the source (both summed in increasing i,
    /// as the hand-written kernel loops); and the mean as the sums times the reciprocal of
    /// <c>count - notEqual(count, 0) + 1</c>, which is the count itself unless it is 0, when it is 1 -- an empty
    /// group's sum is exactly 0, so it means 0 without a branch, as the hand-written kernel's <c>count &gt; 0 ? ... :
    /// 0</c>. <paramref name="notEqualZero"/> is the derived <c>not_equal_scalar</c> (an IEEE unordered comparison:
    /// NaN differs from everything).
    /// </summary>
    public static CodegenProgram ScatterMeanRows(int sourceRows, int innerSize, int outputRows,
        CodegenLibraryFunction notEqualZero)
    {
        if (notEqualZero is null) throw new ArgumentNullException(nameof(notEqualZero));
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();

        // ones[i] = a + (1 - a) with a = notEqual(source[i, 0], 0): what the counts sum. A pass needs a value operand
        // (an index tensor cannot be one); a is exactly 0 or 1 for every x -- NaN and infinities included -- so this is
        // exactly 1 for every row, where x * 0 + 1 would be NaN for an infinite x.
        var rowInput = new CodegenTensorBinding(0, "source", new[] { sourceRows, innerSize },
            new[] { A(0), CodegenAffineExpr.Const(0) });
        var nonZero = CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0));
        var oneBody = CodegenExpr.Add(nonZero, CodegenExpr.Subtract(CodegenExpr.Constant(1.0), nonZero));
        passes.Add(new CodegenKernelSpec("scatter_mean_ones", new CodegenIterationSpace(CodegenAxis.Parallel("i", sourceRows)),
            new[] { rowInput },
            new CodegenTensorBinding(1, "ones", new[] { 1, sourceRows, 1 },
                new[] { CodegenAffineExpr.Const(0), A(0), CodegenAffineExpr.Const(0) }, isOutput: true),
            oneBody.Operands(),
            CodegenReduceKind.None, body: oneBody));
        temps.Add(sourceRows);

        var (countSpace, countIndices, ones) = ScatterInversion(sourceRows, outputRows, 1, "ones");
        passes.Add(new CodegenKernelSpec("scatter_mean_counts", countSpace, new[] { countIndices, ones },
            new CodegenTensorBinding(2, "counts", new[] { outputRows, 1 }, new[] { A(0), A(1) }, isOutput: true),
            new[] { 1 }, CodegenReduceKind.Sum));

        var (sumSpace, sumIndices, source) = ScatterInversion(sourceRows, outputRows, innerSize);
        passes.Add(new CodegenKernelSpec("scatter_mean_sums", sumSpace, new[] { sumIndices, source },
            new CodegenTensorBinding(2, "sums", new[] { outputRows, innerSize }, new[] { A(0), A(1) }, isOutput: true),
            new[] { 1 }, CodegenReduceKind.Sum));
        temps.Add((long)outputRows * innerSize);

        // recip[g] = 1 / (count - notEqual(count, 0) + 1): 1/count, or 1 for an empty group.
        var clamped = CodegenExpr.Add(
            CodegenExpr.Subtract(Op(0), CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0))),
            CodegenExpr.Constant(1.0));
        passes.Add(RowPass("scatter_mean_recip", new[] { outputRows, 1 }, new[] { "counts" }, "recip", clamped,
            CodegenActivationKind.Reciprocal));
        temps.Add(outputRows);

        var meanSpace = new CodegenIterationSpace(CodegenAxis.Parallel("g", outputRows), CodegenAxis.Parallel("j", innerSize));
        passes.Add(new CodegenKernelSpec("scatter_mean_out", meanSpace,
            new[]
            {
                Bind(0, "sums", new[] { outputRows, innerSize }),
                new CodegenTensorBinding(1, "recip", new[] { outputRows, 1 }, new[] { A(0), CodegenAffineExpr.Const(0) }),
            },
            Bind(2, "out", new[] { outputRows, innerSize }, isOutput: true), new[] { 0 }, CodegenReduceKind.None,
            scaleInput: 1));
        return new CodegenProgram(passes, temps, "row-wise scatter mean with counts (backend contract)");
    }

    /// <summary>
    /// CSR · dense → dense (#1441), in CudaBackend.CsrSpMM's contract: <c>out[r, j] = sum over p in
    /// [rowPtr[r], rowPtr[r + 1]) of values[p] * dense[colIdx[p], j]</c>, summed in increasing p as csr_spmm loops.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A row's extent is data-dependent, which the IR has no form for, so this is written as an inversion over every
    /// stored entry: rows x columns in parallel, the nnz entries reduced, and entry p admitted only where
    /// <c>rowPtr[r] &lt;= p &lt; rowPtr[r + 1]</c>. The IR's range test admits <c>raw = loaded - shift</c> in
    /// <c>[0, bound)</c>, so each half of that condition is one test on a tensor of ones of length nnz (any in-range
    /// address reads 1, a miss reads 0):
    /// </para>
    /// <list type="bullet">
    /// <item><description><c>p &gt;= rowPtr[r]</c>: <c>rowPtr[r] - (p - (nnz - 1))</c> lies in [0, nnz) exactly then,
    /// since p &lt; nnz;</description></item>
    /// <item><description><c>p &lt; rowPtr[r + 1]</c>: <c>rowPtr[r + 1] - (p + 1)</c> lies in [0, nnz) exactly then,
    /// since rowPtr[r + 1] &lt;= nnz.</description></item>
    /// </list>
    /// <para>
    /// The cost is rows x nnz x columns, against the hand-written kernels' nnz x columns, so it is admitted only for
    /// small matrices, as the scatter inversions are. A rejected entry contributes the reduction's identity, not 0 x
    /// its term, so a non-finite dense value reaches only the rows whose entries reference it, as in csr_spmm. The ones
    /// are built with <paramref name="notEqualZero"/> (the derived not_equal_scalar), as the scatter mean's are.
    /// </para>
    /// </remarks>
    public static CodegenProgram CsrSpMM(int rows, int cols, int n, int nnz, CodegenLibraryFunction notEqualZero) =>
        CsrSpMM(rows, cols, n, nnz, notEqualZero, bias: false);

    /// <summary>
    /// <see cref="CsrSpMM(int, int, int, int, CodegenLibraryFunction)"/> plus a per-column bias, in
    /// CudaBackend.CsrSpMMBias' contract: <c>out[r, j] = bias[j] + sum over row r's entries</c>. The bias is the
    /// reduction's epilogue, added after the entries' sum where csr_spmm_bias starts its sum from it, so the two can
    /// differ by the rounding of that one addition.
    /// </summary>
    public static CodegenProgram CsrSpMMBias(int rows, int cols, int n, int nnz, CodegenLibraryFunction notEqualZero) =>
        CsrSpMM(rows, cols, n, nnz, notEqualZero, bias: true);

    private static CodegenProgram CsrSpMM(int rows, int cols, int n, int nnz, CodegenLibraryFunction notEqualZero,
        bool bias)
    {
        if (rows <= 0 || cols <= 0 || n <= 0 || nnz <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "CSR SpMM extents must be positive.");
        if (notEqualZero is null) throw new ArgumentNullException(nameof(notEqualZero));
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();

        // ones[p] = a + (1 - a), a = notEqual(values[p], 0): exactly 1 for every entry.
        var nonZero = CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0));
        var oneBody = CodegenExpr.Add(nonZero, CodegenExpr.Subtract(CodegenExpr.Constant(1.0), nonZero));
        passes.Add(new CodegenKernelSpec("csr_spmm_ones", new CodegenIterationSpace(CodegenAxis.Parallel("p", nnz)),
            new[] { Bind(0, "values", new[] { nnz }) }, Bind(1, "ones", new[] { nnz }, isOutput: true),
            oneBody.Operands(), CodegenReduceKind.None, body: oneBody));
        temps.Add(nnz);

        // out[r, j] = sum_p values[p] * dense[colIdx[p], j] * [rowPtr[r] <= p] * [p < rowPtr[r + 1]].
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("r", rows), CodegenAxis.Parallel("j", n),
            CodegenAxis.Reduce("p", nnz));
        var rowStart = new CodegenTensorBinding(4, "ones", new[] { nnz }, new[] { CodegenAffineExpr.Const(0) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, A(0), nnz, CodegenIndexOutOfRange.Skip, floatIndices: false,
                    offset: new CodegenAffineExpr(new[] { new CodegenAffineTerm(2, 1) }, -(nnz - 1))),
            });
        var rowEnd = new CodegenTensorBinding(5, "ones", new[] { nnz }, new[] { CodegenAffineExpr.Const(0) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, new CodegenAffineExpr(new[] { new CodegenAffineTerm(0, 1) }, 1), nnz,
                    CodegenIndexOutOfRange.Skip, floatIndices: false,
                    offset: new CodegenAffineExpr(new[] { new CodegenAffineTerm(2, 1) }, 1)),
            });
        var inputs = new List<CodegenTensorBinding>
        {
            new CodegenTensorBinding(0, "rowPtr", new[] { rows + 1 }, new[] { A(0) },
                elementType: CodegenElementType.Int32),
            new CodegenTensorBinding(1, "colIdx", new[] { nnz }, new[] { A(2) }, elementType: CodegenElementType.Int32),
            new CodegenTensorBinding(2, "values", new[] { nnz }, new[] { A(2) }),
            new CodegenTensorBinding(3, "dense", new[] { cols, n }, new[] { CodegenAffineExpr.Const(0), A(1) },
                indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(1, A(2), cols), null }),
            rowStart,
            rowEnd,
        };
        if (bias) inputs.Add(new CodegenTensorBinding(6, "bias", new[] { n }, new[] { A(1) }));
        passes.Add(new CodegenKernelSpec(bias ? "csr_spmm_bias" : "csr_spmm", space, inputs.ToArray(),
            Bind(inputs.Count, "out", new[] { rows, n }, isOutput: true), new[] { 2, 3, 4, 5 }, CodegenReduceKind.Sum,
            biasInput: bias ? 6 : null));
        return new CodegenProgram(passes, temps, "csr spmm as an inversion over the stored entries (backend contract)");
    }

    /// <summary>
    /// The row-wise scatter softmax (#1441), in CudaBackend.ScatterSoftmaxRows' contract: for each source row i in
    /// group g = indices[i] and each column j, <c>out[i, j] = exp(source[i, j] - max_g[j]) / sum_g[j]</c>, with the
    /// maximum and the sum over the rows of g. The maximum is the segment maximum; the reciprocal sum is the scatter
    /// inversion of <c>exp(source - max)</c>, summed in increasing row as the hand-written kernel loops; each row then
    /// gathers its group's two by indices[i]. A row whose group lies outside [0, numGroups) writes exactly 0, as there,
    /// whatever its source values: the normalising pass reads a per-group one through indices[i], so such a row's term
    /// is skipped entirely (its bias and exponential included) rather than multiplied by a gathered zero, which an
    /// infinite or overflowing exponential would turn into NaN.
    /// </summary>
    /// <remarks>
    /// The hand-written kernel divides by the sum; the IR multiplies by its reciprocal, so the two can differ by one
    /// FP32 ulp, as the declared attention softmax does. Both write the same value wherever the sum is exactly a power
    /// of two.
    /// </remarks>
    public static CodegenProgram ScatterSoftmaxRows(int sourceRows, int innerSize, int numGroups,
        CodegenLibraryFunction notEqualZero)
    {
        if (notEqualZero is null) throw new ArgumentNullException(nameof(notEqualZero));
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();
        int[] groupShape = { numGroups, innerSize };

        // negmax[g, j] = -max over g's rows (a negated maximum is the bias the exponentials subtract).
        var (maxSpace, maxIndices, maxSource) = ScatterInversion(sourceRows, numGroups, innerSize);
        passes.Add(new CodegenKernelSpec("scatter_softmax_negmax", maxSpace, new[] { maxIndices, maxSource },
            new CodegenTensorBinding(2, "negmax", groupShape, new[] { A(0), A(1) }, isOutput: true),
            new[] { 1 }, CodegenReduceKind.Max, reduceScale: -1.0));
        temps.Add((long)numGroups * innerSize);

        // recipsum[g, j] = 1 / sum over g's rows of exp(source - max).
        var (sumSpace, sumIndices, sumSource) = ScatterInversion(sourceRows, numGroups, innerSize);
        passes.Add(new CodegenKernelSpec("scatter_softmax_recipsum", sumSpace,
            new[] { sumIndices, sumSource, Bind(2, "negmax", groupShape) },
            new CodegenTensorBinding(3, "recipsum", groupShape, new[] { A(0), A(1) }, isOutput: true),
            new[] { 1 }, CodegenReduceKind.Sum, activation: CodegenActivationKind.Reciprocal,
            preReduce: CodegenPreReduceOp.Exp, preBiasInput: 2));
        temps.Add((long)numGroups * innerSize);

        // Each row gathers its group's two statistics by indices[i] (zeros for a group outside [0, numGroups)), so
        // the normalising pass reads them per row: an indirect index cannot address a pre-reduce bias.
        int[] rowShape = { sourceRows, innerSize };
        var rowSpace = new CodegenIterationSpace(CodegenAxis.Parallel("i", sourceRows), CodegenAxis.Parallel("j", innerSize));
        foreach (string statistic in new[] { "negmax", "recipsum" })
        {
            passes.Add(new CodegenKernelSpec("scatter_softmax_gather_" + statistic, rowSpace,
                new[]
                {
                    new CodegenTensorBinding(0, "indices", new[] { sourceRows }, new[] { A(0) },
                        elementType: CodegenElementType.Int32),
                    new CodegenTensorBinding(1, statistic, groupShape, new[] { CodegenAffineExpr.Const(0), A(1) },
                        indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(0), numGroups), null }),
                },
                Bind(2, "row_" + statistic, rowShape, isOutput: true), new[] { 1 }, CodegenReduceKind.None));
            temps.Add((long)sourceRows * innerSize);
        }

        // group_ones[g] = a + (1 - a), a = notEqual(negmax[g, 0], 0): exactly 1 for every group, NaN included.
        var groupNonZero = CodegenExpr.Call(notEqualZero, Op(0), CodegenExpr.Constant(0.0));
        var groupOneBody = CodegenExpr.Add(groupNonZero, CodegenExpr.Subtract(CodegenExpr.Constant(1.0), groupNonZero));
        passes.Add(new CodegenKernelSpec("scatter_softmax_group_ones",
            new CodegenIterationSpace(CodegenAxis.Parallel("g", numGroups)),
            new[] { new CodegenTensorBinding(0, "negmax", groupShape, new[] { A(0), CodegenAffineExpr.Const(0) }) },
            Bind(1, "group_ones", new[] { numGroups }, isOutput: true), groupOneBody.Operands(), CodegenReduceKind.None,
            body: groupOneBody));
        temps.Add(numGroups);

        // out[i, j] = exp(source[i, j] * group_ones[indices[i]] + row_negmax[i, j]) * row_recipsum[i, j]. For a row
        // outside the groups the group_ones read is out of range, so the whole term -- bias and exponential included --
        // is the identity 0, and the row writes 0 even where its source is infinite.
        passes.Add(new CodegenKernelSpec("scatter_softmax_out", rowSpace,
            new[]
            {
                Bind(0, "source", rowShape), Bind(1, "row_negmax", rowShape), Bind(2, "row_recipsum", rowShape),
                new CodegenTensorBinding(3, "indices", new[] { sourceRows }, new[] { A(0) },
                    elementType: CodegenElementType.Int32),
                new CodegenTensorBinding(4, "group_ones", new[] { numGroups }, new[] { CodegenAffineExpr.Const(0) },
                    indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(3, A(0), numGroups) }),
            },
            Bind(5, "out", rowShape, isOutput: true), new[] { 0, 4 }, CodegenReduceKind.None,
            scaleInput: 2, preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1));
        return new CodegenProgram(passes, temps, "row-wise scatter softmax (backend contract)");
    }

    /// <summary>
    /// The deterministic scatter-add onto a seed (#890 phase 4e), in CudaBackend's
    /// scatter_add_accumulate_deterministic contract: <c>out[d, f] = seed[d, f] + sum of source[i, f] over every i
    /// with indices[i] == d</c>, in increasing i. The caller binds <c>seed</c> and <c>out</c> to the same buffer,
    /// which is what accumulating in place means; each output cell is owned by one thread, which reads that cell
    /// before it writes it, exactly as the scanning kernel does.
    /// </summary>
    public static CodegenProgram ScatterAddAccumulateDeterministic(int count, int destinations, int features)
    {
        var (space, indices, source) = ScatterInversion(count, destinations, features);
        int[] cells = { destinations, features };
        var seed = new CodegenTensorBinding(2, "seed", cells, new[] { A(0), A(1) });
        var output = new CodegenTensorBinding(3, "out", cells, new[] { A(0), A(1) }, isOutput: true);
        var pass = new CodegenKernelSpec("scatter_add_accumulate_deterministic", space,
            new[] { indices, source, seed }, output, new[] { 1 }, CodegenReduceKind.Sum, biasInput: 2);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(),
            "deterministic scatter-add onto a seed (backend contract)");
    }

    /// <summary>
    /// The deterministic segment maximum (#890 phase 4e): <c>out[s, f] = max of source[i, f] over every i whose
    /// segment id is s</c>. The same inversion as <see cref="ScatterAddDeterministic"/>, so it is deterministic for
    /// the same reason and accepts unsorted ids; a segment with no member reduces to the maximum's identity,
    /// -infinity. With features = 1 and the segment ids a sparse matrix's row indices this is the CSR segmented
    /// maximum over that matrix's rows.
    /// </summary>
    public static CodegenProgram SegmentMax(int count, int segments, int features)
    {
        var (space, indices, source) = ScatterInversion(count, segments, features);
        var output = new CodegenTensorBinding(2, "out", new[] { segments, features }, new[] { A(0), A(1) },
            isOutput: true);
        var pass = new CodegenKernelSpec("segment_max", space, new[] { indices, source }, output,
            new[] { 1 }, CodegenReduceKind.Max);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "deterministic segment max (backend contract)");
    }

    /// <summary>
    /// The deterministic segment minimum (#890 phase 4e): <see cref="SegmentMax"/> over the negated sources, negated
    /// again. The IR reduces with a sum or a maximum, and <c>min(x) = -max(-x)</c> is exact in floating point -- a
    /// negation only flips the sign bit -- so this is the minimum, not an approximation of it. A segment with no
    /// member reduces to +infinity, the minimum's identity.
    /// </summary>
    public static CodegenProgram SegmentMin(int count, int segments, int features)
    {
        var (space, indices, source) = ScatterInversion(count, segments, features);
        var output = new CodegenTensorBinding(2, "out", new[] { segments, features }, new[] { A(0), A(1) },
            isOutput: true);
        var body = CodegenExpr.Negate(Op(1));
        var pass = new CodegenKernelSpec("segment_min", space, new[] { indices, source }, output,
            body.Operands(), CodegenReduceKind.Max, reduceScale: -1.0, body: body);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "deterministic segment min (backend contract)");
    }

    /// <summary>
    /// <c>dst[i] = src[offset + i * stride]</c> for i in [0, count): CudaBackend.StridedGather's contract (#1441). The
    /// read is affine, so it needs no index tensor; the caller has checked that the last element read is inside src.
    /// </summary>
    public static CodegenProgram StridedGather(int count, int sourceLength, int offset, int stride)
    {
        ValidateStrided(count, sourceLength, offset, stride, "source");
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));
        var source = new CodegenTensorBinding(0, "src", new[] { sourceLength }, new[] { Strided(offset, stride) });
        var output = new CodegenTensorBinding(1, "dst", new[] { count }, new[] { A(0) }, isOutput: true);
        var pass = new CodegenKernelSpec("strided_gather", space, new[] { source }, output, new[] { 0 },
            CodegenReduceKind.None);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "strided_gather (backend contract)");
    }

    /// <summary>
    /// <c>dst[offset + i * stride] = src[i]</c> for i in [0, count): CudaBackend.StridedScatter's contract (#1441). The
    /// write is affine and injective (stride &gt; 0), so each written cell has one owner and no atomic is needed, and the
    /// cells between the strided ones are not written: the caller's destination keeps them, as with the hand-written
    /// kernel. (The interpreter's output starts at zero, so there they read 0.)
    /// </summary>
    public static CodegenProgram StridedScatter(int count, int destinationLength, int offset, int stride)
    {
        ValidateStrided(count, destinationLength, offset, stride, "destination");
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));
        var source = new CodegenTensorBinding(0, "src", new[] { count }, new[] { A(0) });
        var output = new CodegenTensorBinding(1, "dst", new[] { destinationLength }, new[] { Strided(offset, stride) },
            isOutput: true);
        var pass = new CodegenKernelSpec("strided_scatter", space, new[] { source }, output, new[] { 0 },
            CodegenReduceKind.None);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "strided_scatter (backend contract)");
    }

    private static CodegenAffineExpr Strided(int offset, int stride) =>
        new(new[] { new CodegenAffineTerm(0, stride) }, offset);

    private static void ValidateStrided(int count, int length, int offset, int stride, string side)
    {
        if (count <= 0 || length <= 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Strided extents must be positive.");
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (offset + (long)(count - 1) * stride >= length)
            throw new ArgumentOutOfRangeException(nameof(count),
                "The last strided element lies past the " + side + " (" + length + " elements).");
    }

    /// <summary>
    /// <c>out[o, j, i] = source[o, indices[j], i]</c>: indices int32 [indexAxis], source [outer, sourceAxis, inner],
    /// out [outer, indexAxis, inner] -- CudaBackend.IndexSelect's contract. An index outside [0, sourceAxis) yields
    /// zeros, which resident_index_select now bounds to as well.
    /// </summary>
    public static CodegenProgram IndexSelect(int outer, int sourceAxis, int indexAxis, int inner)
    {
        if (outer <= 0 || sourceAxis <= 0 || indexAxis <= 0 || inner <= 0)
            throw new ArgumentOutOfRangeException(nameof(outer), "IndexSelect extents must be positive.");
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("o", outer), CodegenAxis.Parallel("j", indexAxis),
            CodegenAxis.Parallel("i", inner));
        var indices = new CodegenTensorBinding(0, "indices", new[] { indexAxis }, new[] { A(1) },
            elementType: CodegenElementType.Int32);
        var source = new CodegenTensorBinding(1, "source", new[] { outer, sourceAxis, inner },
            new[] { A(0), CodegenAffineExpr.Const(0), A(2) },
            indirect: new CodegenIndirectIndex?[] { null, new CodegenIndirectIndex(0, A(1), sourceAxis), null });
        var output = new CodegenTensorBinding(2, "out", new[] { outer, indexAxis, inner }, new[] { A(0), A(1), A(2) },
            isOutput: true);
        var pass = new CodegenKernelSpec("index_select", space, new[] { indices, source }, output, new[] { 1 },
            CodegenReduceKind.None);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "index_select (backend contract)");
    }

    /// <summary>
    /// <c>out[o, j, i] = input[o, (int)indices[o, j, i], i]</c>: indices float32 [outer, axisOut, inner] (truncated
    /// toward zero, as the hand-written kernel's <c>(int)</c>), input [outer, axisIn, inner], out [outer, axisOut,
    /// inner] -- CudaBackend.TakeAlongDim's contract. An index outside [0, axisIn) yields 0, as there.
    /// </summary>
    public static CodegenProgram TakeAlongDim(int outer, int axisOut, int inner, int axisIn)
    {
        if (outer <= 0 || axisOut <= 0 || inner <= 0 || axisIn <= 0)
            throw new ArgumentOutOfRangeException(nameof(outer), "TakeAlongDim extents must be positive.");
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("o", outer), CodegenAxis.Parallel("j", axisOut),
            CodegenAxis.Parallel("i", inner));
        var indices = new CodegenTensorBinding(0, "indices", new[] { outer, axisOut, inner }, new[] { A(0), A(1), A(2) });
        // The position of an index within the flat indices tensor: (o * axisOut + j) * inner + i.
        var position = new CodegenAffineExpr(new[]
        {
            new CodegenAffineTerm(0, axisOut * inner), new CodegenAffineTerm(1, inner), new CodegenAffineTerm(2, 1),
        }, 0);
        var input = new CodegenTensorBinding(1, "input", new[] { outer, axisIn, inner },
            new[] { A(0), CodegenAffineExpr.Const(0), A(2) },
            indirect: new CodegenIndirectIndex?[]
            {
                null, new CodegenIndirectIndex(0, position, axisIn, floatIndices: true), null,
            });
        var output = new CodegenTensorBinding(2, "out", new[] { outer, axisOut, inner }, new[] { A(0), A(1), A(2) },
            isOutput: true);
        var pass = new CodegenKernelSpec("take_along_dim", space, new[] { indices, input }, output, new[] { 1 },
            CodegenReduceKind.None);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "take_along_dim (backend contract)");
    }

    // ---- attention (#890 phase 4d), in CudaBackend.ScaledDotProductAttention's contract ----------------------------

    /// <summary>The logit a masked key gets: its exponential underflows to exactly 0 against any unmasked logit.</summary>
    public const double AttentionMaskedLogit = -1e30;

    /// <summary>
    /// Scaled dot-product attention: <c>weights[b,h,q,:] = softmax_k(scale * q[b,h,q,:] . k[b,h/g,k,:])</c> over the
    /// unmasked keys, <c>out[b,h,q,:] = weights[b,h,q,:] . v[b,h/g,:,:]</c>, where query head h reads key/value head
    /// <c>h / (heads / kvHeads)</c>. q, out are [b, heads, seqQ, d]; k, v are [b, kvHeads, seqK, d]; weights is
    /// [b, heads, seqQ, seqK]. <paramref name="mask"/>: none, "mask" shared [seqQ, seqK], or full [b, heads, seqQ,
    /// seqK]; a zero element masks its key (NaN keeps it), through <paramref name="notEqualZero"/> (the derived
    /// <c>not_equal_scalar</c>). <paramref name="causal"/> adds the bound tensor "causal" [seqQ, seqK], 0 or
    /// <see cref="AttentionMaskedLogit"/> -- the backend supplies it, the index arithmetic it needs (key k later than
    /// query q + seqK - seqQ) being outside the IR. A row whose keys are all masked gets zero weights and a zero output,
    /// as the hand-written kernel's: a row-valid factor (1 when any key is kept) multiplies its reciprocal sum.
    /// With <paramref name="log"/> (the derived <c>log_vector</c>) the program also writes "stats" [b, heads, seqQ],
    /// each row's log-sum-exp <c>max + log(sum)</c> -- flash_attention_v2's softmaxStats, which its backward reads.
    /// </summary>
    public static CodegenProgram ScaledDotProductAttention(int batch, int heads, int kvHeads, int seqQ, int seqK,
        int headDim, double scale, CodegenAttentionMask mask, bool causal, CodegenLibraryFunction? notEqualZero,
        CodegenLibraryFunction? log = null)
    {
        if (batch <= 0 || heads <= 0 || kvHeads <= 0 || seqQ <= 0 || seqK <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Attention extents must be positive.");
        if (heads % kvHeads != 0) throw new ArgumentException("heads must be a multiple of kvHeads.", nameof(kvHeads));
        if (mask != CodegenAttentionMask.None && notEqualZero is null)
            throw new ArgumentNullException(nameof(notEqualZero), "A masked attention needs the not-equal function.");
        int group = heads / kvHeads;
        var kvHead = group == 1 ? A(1) : new CodegenAffineExpr(new[] { new CodegenAffineTerm(1, 1) }, 0, group);
        int[] rowShape = { batch, heads, seqQ, seqK };
        int[] pairShape = { seqQ, seqK };
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();
        long logitsElements = (long)batch * heads * seqQ * seqK;
        int rows = checked(batch * heads * seqQ);

        // ---- the additive bias: 0 for a kept key, AttentionMaskedLogit for a masked one (or the sum of two).
        string? bias = null;
        int[]? biasShape = null;
        if (mask != CodegenAttentionMask.None && notEqualZero is { } notEqual)
        {
            bool full = mask == CodegenAttentionMask.Full;
            biasShape = full ? rowShape : pairShape;
            var axes = new CodegenAxis[biasShape.Length];
            for (int d = 0; d < axes.Length; d++) axes[d] = CodegenAxis.Parallel("m" + d, biasShape[d]);
            var inputs = new List<CodegenTensorBinding> { Bind(0, "mask", biasShape) };
            // (keep - 1) * 1e30: keep is 1 (kept) or 0 (masked).
            CodegenExpr body = CodegenExpr.Multiply(
                CodegenExpr.Subtract(CodegenExpr.Call(notEqual, Op(0), CodegenExpr.Constant(0.0)), CodegenExpr.Constant(1.0)),
                CodegenExpr.Constant(-AttentionMaskedLogit));
            if (causal)
            {
                var causalMap = full ? new[] { A(2), A(3) } : new[] { A(0), A(1) };
                inputs.Add(new CodegenTensorBinding(1, "causal", pairShape, causalMap));
                body = CodegenExpr.Add(body, Op(1));
            }
            var ops = new int[inputs.Count];
            for (int i = 0; i < ops.Length; i++) ops[i] = i;
            passes.Add(new CodegenKernelSpec("attention_mask_bias", new CodegenIterationSpace(axes), inputs.ToArray(),
                Bind(inputs.Count, "bias", biasShape, isOutput: true), body.Operands(), CodegenReduceKind.None, body: body));
            temps.Add(full ? logitsElements : (long)seqQ * seqK);
            bias = "bias";
        }
        else if (causal)
        {
            bias = "causal";
            biasShape = pairShape;
        }
        bool fullBias = biasShape is { Length: 4 };

        // ---- logits[b,h,q,k] = scale * sum_d q[b,h,q,d] k[b,h/g,k,d] + bias
        var scoreSpace = new CodegenIterationSpace(CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("h", heads),
            CodegenAxis.Parallel("q", seqQ), CodegenAxis.Parallel("k", seqK), CodegenAxis.Reduce("d", headDim));
        var scoreInputs = new List<CodegenTensorBinding>
        {
            new(0, "q", new[] { batch, heads, seqQ, headDim }, new[] { A(0), A(1), A(2), A(4) }),
            new(1, "k", new[] { batch, kvHeads, seqK, headDim }, new[] { A(0), kvHead, A(3), A(4) }),
        };
        if (bias is not null && biasShape is not null)
            scoreInputs.Add(new CodegenTensorBinding(2, bias, biasShape,
                fullBias ? new[] { A(0), A(1), A(2), A(3) } : new[] { A(2), A(3) }));
        passes.Add(new CodegenKernelSpec("attention_scores", scoreSpace, scoreInputs.ToArray(),
            new CodegenTensorBinding(scoreInputs.Count, "logits", rowShape, new[] { A(0), A(1), A(2), A(3) }, isOutput: true),
            new[] { 0, 1 }, CodegenReduceKind.Sum, biasInput: bias is null ? null : 2, reduceScale: scale));
        temps.Add(logitsElements);

        // ---- row softmax over k (CodegenFusedStatistics.Softmax's three passes, on the logits as [rows, seqK]).
        int[] matrix = { rows, seqK };
        int[] vector = { rows };
        var reduceSpace = new CodegenIterationSpace(CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("j", seqK));
        passes.Add(new CodegenKernelSpec("attention_negmax", reduceSpace,
            new[] { Bind(0, "logits", matrix) }, Bind(1, "negmax", vector, isOutput: true), new[] { 0 },
            CodegenReduceKind.Max, reduceScale: -1.0));
        passes.Add(new CodegenKernelSpec("attention_recipsum", reduceSpace,
            new[] { Bind(0, "logits", matrix), Bind(1, "negmax", vector) }, Bind(2, "recipsum", vector, isOutput: true),
            new[] { 0 }, CodegenReduceKind.Sum, activation: CodegenActivationKind.Reciprocal,
            preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1));
        temps.Add(rows);
        temps.Add(rows);
        string recip = "recipsum";
        // A row can lose every key only to a mask, or to causality when there are fewer keys than queries.
        if (bias is not null && biasShape is not null && (mask != CodegenAttentionMask.None || seqK < seqQ))
        {
            // valid = relu(max_k(bias) * 1e-30 + 1): 1 when some key is kept (bias 0), 0 when none is (bias <= -1e30).
            var validSpace = new CodegenIterationSpace(CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("h", heads),
                CodegenAxis.Parallel("q", seqQ), CodegenAxis.Reduce("k", seqK));
            passes.Add(new CodegenKernelSpec("attention_rowvalid", validSpace,
                new[] { new CodegenTensorBinding(0, bias, biasShape, fullBias ? new[] { A(0), A(1), A(2), A(3) } : new[] { A(2), A(3) }) },
                new CodegenTensorBinding(1, "rowvalid", new[] { batch, heads, seqQ }, new[] { A(0), A(1), A(2) }, isOutput: true),
                new[] { 0 }, CodegenReduceKind.Max, activation: CodegenActivationKind.ReLU,
                reduceScale: -1.0 / AttentionMaskedLogit, reduceOffset: 1.0));
            passes.Add(RowPass("attention_recipvalid", vector, new[] { "recipsum", "rowvalid" }, "recipvalid",
                CodegenExpr.Multiply(Op(0), Op(1))));
            temps.Add(rows);
            temps.Add(rows);
            recip = "recipvalid";
        }
        var outSpace = new CodegenIterationSpace(CodegenAxis.Parallel("i", rows), CodegenAxis.Parallel("j", seqK));
        passes.Add(new CodegenKernelSpec("attention_normalise", outSpace,
            new[] { Bind(0, "logits", matrix), Bind(1, "negmax", vector), Bind(2, recip, vector) },
            Bind(3, "weights", matrix, isOutput: true), new[] { 0 }, CodegenReduceKind.None, scaleInput: 2,
            preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1));

        // ---- out[b,h,q,e] = sum_k weights[b,h,q,k] v[b,h/g,k,e]
        var valueSpace = new CodegenIterationSpace(CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("h", heads),
            CodegenAxis.Parallel("q", seqQ), CodegenAxis.Parallel("e", headDim), CodegenAxis.Reduce("k", seqK));
        passes.Add(new CodegenKernelSpec("attention_output", valueSpace,
            new[]
            {
                new CodegenTensorBinding(0, "weights", rowShape, new[] { A(0), A(1), A(2), A(4) }),
                new CodegenTensorBinding(1, "v", new[] { batch, kvHeads, seqK, headDim }, new[] { A(0), kvHead, A(4), A(3) }),
            },
            new CodegenTensorBinding(2, "out", new[] { batch, heads, seqQ, headDim }, new[] { A(0), A(1), A(2), A(3) },
                isOutput: true),
            new[] { 0, 1 }, CodegenReduceKind.Sum));
        // stats = max + log(sum) = -negmax - log(recipsum)
        if (log is not null)
            passes.Add(RowPass("attention_stats", vector, new[] { "negmax", "recipsum" }, "stats",
                CodegenExpr.Subtract(CodegenExpr.Negate(Op(0)), CodegenExpr.Call(log, Op(1)))));
        return new CodegenProgram(passes, temps, "scaled dot-product attention (backend contract)");
    }

    /// <summary>
    /// Paged attention's decode step (#890 phase 4d), in CudaBackend.PagedAttentionDecode's contract: one query per
    /// head against a KV cache held in pages. <c>block_table</c> is int32 [pages] and maps a logical page to its
    /// physical block, so the cache is [blocks, blockSize, kvHeads, headDim] and key <c>k = page * blockSize + pos</c>
    /// lives at block_table[page], pos. <c>key_bias</c> is [pages * blockSize], 0 for a key inside the sequence and
    /// <see cref="AttentionMaskedLogit"/> for one in the last page's unused tail -- the backend supplies it, as it
    /// already supplies the causal bias, because comparing k against a run-time seqLen is outside the IR. The caller
    /// guarantees at least one unmasked key, which decode always has.
    /// </summary>
    public static CodegenProgram PagedAttentionDecode(int heads, int kvHeads, int headDim, int blockSize, int pages,
        int blocks, double scale)
    {
        if (heads <= 0 || kvHeads <= 0 || headDim <= 0 || blockSize <= 0 || pages <= 0 || blocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(heads), "Paged attention extents must be positive.");
        if (heads % kvHeads != 0) throw new ArgumentException("heads must be a multiple of kvHeads.", nameof(kvHeads));
        int group = heads / kvHeads;
        // The kv head of query head h, as the grouped-query kernels compute it.
        CodegenAffineExpr KvHead() => group == 1
            ? A(0)
            : new CodegenAffineExpr(new[] { new CodegenAffineTerm(0, 1) }, 0, group);
        // k = page * blockSize + pos, affine in the two axes the cache is addressed by.
        CodegenAffineExpr Key(int pageAxis, int posAxis) => new(new[]
            { new CodegenAffineTerm(pageAxis, blockSize), new CodegenAffineTerm(posAxis, 1) });

        int seqK = checked(pages * blockSize);
        int[] matrix = { heads, seqK };
        int[] vector = { heads };
        int[] cache = { blocks, blockSize, kvHeads, headDim };
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();

        // ---- logits[h, k] = scale * sum_d q[h, d] * kcache[table[page], pos, h/g, d] + key_bias[k]
        var scoreSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("h", heads), CodegenAxis.Parallel("page", pages),
            CodegenAxis.Parallel("pos", blockSize), CodegenAxis.Reduce("d", headDim));
        var scoreInputs = new[]
        {
            new CodegenTensorBinding(0, "block_table", new[] { pages }, new[] { A(1) },
                elementType: CodegenElementType.Int32),
            new CodegenTensorBinding(1, "q", new[] { heads, headDim }, new[] { A(0), A(3) }),
            new CodegenTensorBinding(2, "kcache", cache,
                new[] { CodegenAffineExpr.Const(0), A(2), KvHead(), A(3) },
                indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(1), blocks), null, null, null }),
            new CodegenTensorBinding(3, "key_bias", new[] { seqK }, new[] { Key(1, 2) }),
        };
        passes.Add(new CodegenKernelSpec("paged_attention_scores", scoreSpace, scoreInputs,
            new CodegenTensorBinding(4, "logits", matrix, new[] { A(0), Key(1, 2) }, isOutput: true),
            new[] { 1, 2 }, CodegenReduceKind.Sum, biasInput: 3, reduceScale: scale));
        temps.Add((long)heads * seqK);

        // ---- the row softmax over k, as ScaledDotProductAttention's, on the logits as [heads, seqK].
        var reduceSpace = new CodegenIterationSpace(CodegenAxis.Parallel("i", heads), CodegenAxis.Reduce("j", seqK));
        passes.Add(new CodegenKernelSpec("paged_attention_negmax", reduceSpace,
            new[] { Bind(0, "logits", matrix) }, Bind(1, "negmax", vector, isOutput: true), new[] { 0 },
            CodegenReduceKind.Max, reduceScale: -1.0));
        passes.Add(new CodegenKernelSpec("paged_attention_recipsum", reduceSpace,
            new[] { Bind(0, "logits", matrix), Bind(1, "negmax", vector) },
            Bind(2, "recipsum", vector, isOutput: true), new[] { 0 }, CodegenReduceKind.Sum,
            activation: CodegenActivationKind.Reciprocal, preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1));
        temps.Add(heads);
        temps.Add(heads);
        var weightSpace = new CodegenIterationSpace(CodegenAxis.Parallel("i", heads), CodegenAxis.Parallel("j", seqK));
        passes.Add(new CodegenKernelSpec("paged_attention_normalise", weightSpace,
            new[] { Bind(0, "logits", matrix), Bind(1, "negmax", vector), Bind(2, "recipsum", vector) },
            Bind(3, "weights", matrix, isOutput: true), new[] { 0 }, CodegenReduceKind.None, scaleInput: 2,
            preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1));
        temps.Add((long)heads * seqK);

        // ---- out[h, e] = sum_k weights[h, k] * vcache[table[page], pos, h/g, e]
        var valueSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("h", heads), CodegenAxis.Parallel("e", headDim),
            CodegenAxis.Reduce("page", pages), CodegenAxis.Reduce("pos", blockSize));
        passes.Add(new CodegenKernelSpec("paged_attention_output", valueSpace,
            new[]
            {
                new CodegenTensorBinding(0, "block_table", new[] { pages }, new[] { A(2) },
                    elementType: CodegenElementType.Int32),
                new CodegenTensorBinding(1, "weights", matrix, new[] { A(0), Key(2, 3) }),
                new CodegenTensorBinding(2, "vcache", cache,
                    new[] { CodegenAffineExpr.Const(0), A(3), KvHead(), A(1) },
                    indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(2), blocks), null, null, null }),
            },
            new CodegenTensorBinding(3, "out", new[] { heads, headDim }, new[] { A(0), A(1) }, isOutput: true),
            new[] { 1, 2 }, CodegenReduceKind.Sum));
        return new CodegenProgram(passes, temps, "paged attention decode (backend contract)");
    }

    /// <summary>
    /// Paged attention's prefill step (#890 phase 4d), in CudaBackend.PagedAttentionPrefill's contract:
    /// <paramref name="numQueries"/> queries per head against a KV cache held in pages, causal in the logical
    /// position. The cache is addressed exactly as in <see cref="PagedAttentionDecode"/>. <c>causal_bias</c> is
    /// [numQueries, pages * blockSize]: 0 where query qi may see key k and <see cref="AttentionMaskedLogit"/>
    /// otherwise, so it carries both the causal rule (<c>k &lt;= startPos + qi</c>) and the last page's unused tail.
    /// The backend supplies it because neither comparison is a quasi-affine index; startPos therefore does not enter
    /// the program, only the bias it builds. Every row keeps key 0, so no row can lose every key.
    /// </summary>
    public static CodegenProgram PagedAttentionPrefill(int heads, int kvHeads, int headDim, int blockSize, int pages,
        int blocks, int numQueries, double scale)
    {
        if (heads <= 0 || kvHeads <= 0 || headDim <= 0 || blockSize <= 0 || pages <= 0 || blocks <= 0 ||
            numQueries <= 0)
            throw new ArgumentOutOfRangeException(nameof(heads), "Paged attention extents must be positive.");
        if (heads % kvHeads != 0) throw new ArgumentException("heads must be a multiple of kvHeads.", nameof(kvHeads));
        int group = heads / kvHeads;
        // The kv head of query head h (axis 1 here, where decode has it on axis 0).
        CodegenAffineExpr KvHead() => group == 1
            ? A(1)
            : new CodegenAffineExpr(new[] { new CodegenAffineTerm(1, 1) }, 0, group);
        CodegenAffineExpr Key(int pageAxis, int posAxis) => new(new[]
            { new CodegenAffineTerm(pageAxis, blockSize), new CodegenAffineTerm(posAxis, 1) });
        // The logits are [numQueries * heads, seqK] so the row softmax below is the same three passes decode runs;
        // row (qi, h) is qi * heads + h, which is what the [numQueries, heads, seqK] layout already puts there.
        CodegenAffineExpr Row(int queryAxis, int headAxis) => new(new[]
            { new CodegenAffineTerm(queryAxis, heads), new CodegenAffineTerm(headAxis, 1) });

        int seqK = checked(pages * blockSize);
        int rows = checked(numQueries * heads);
        int[] matrix = { rows, seqK };
        int[] vector = { rows };
        int[] cache = { blocks, blockSize, kvHeads, headDim };
        var passes = new List<CodegenKernelSpec>();
        var temps = new List<long>();

        // ---- logits[qi*heads+h, k] = scale * sum_d q[qi,h,d] * kcache[table[page], pos, h/g, d] + bias[qi, k]
        var scoreSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("qi", numQueries), CodegenAxis.Parallel("h", heads),
            CodegenAxis.Parallel("page", pages), CodegenAxis.Parallel("pos", blockSize),
            CodegenAxis.Reduce("d", headDim));
        var scoreInputs = new[]
        {
            new CodegenTensorBinding(0, "block_table", new[] { pages }, new[] { A(2) },
                elementType: CodegenElementType.Int32),
            new CodegenTensorBinding(1, "q", new[] { numQueries, heads, headDim }, new[] { A(0), A(1), A(4) }),
            new CodegenTensorBinding(2, "kcache", cache,
                new[] { CodegenAffineExpr.Const(0), A(3), KvHead(), A(4) },
                indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(2), blocks), null, null, null }),
            new CodegenTensorBinding(3, "causal_bias", new[] { numQueries, seqK }, new[] { A(0), Key(2, 3) }),
        };
        passes.Add(new CodegenKernelSpec("paged_prefill_scores", scoreSpace, scoreInputs,
            new CodegenTensorBinding(4, "logits", matrix, new[] { Row(0, 1), Key(2, 3) }, isOutput: true),
            new[] { 1, 2 }, CodegenReduceKind.Sum, biasInput: 3, reduceScale: scale));
        temps.Add((long)rows * seqK);

        // ---- the row softmax over k, as decode's, on the logits as [numQueries * heads, seqK].
        var reduceSpace = new CodegenIterationSpace(CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("j", seqK));
        passes.Add(new CodegenKernelSpec("paged_prefill_negmax", reduceSpace,
            new[] { Bind(0, "logits", matrix) }, Bind(1, "negmax", vector, isOutput: true), new[] { 0 },
            CodegenReduceKind.Max, reduceScale: -1.0));
        passes.Add(new CodegenKernelSpec("paged_prefill_recipsum", reduceSpace,
            new[] { Bind(0, "logits", matrix), Bind(1, "negmax", vector) },
            Bind(2, "recipsum", vector, isOutput: true), new[] { 0 }, CodegenReduceKind.Sum,
            activation: CodegenActivationKind.Reciprocal, preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1));
        temps.Add(rows);
        temps.Add(rows);
        var weightSpace = new CodegenIterationSpace(CodegenAxis.Parallel("i", rows), CodegenAxis.Parallel("j", seqK));
        passes.Add(new CodegenKernelSpec("paged_prefill_normalise", weightSpace,
            new[] { Bind(0, "logits", matrix), Bind(1, "negmax", vector), Bind(2, "recipsum", vector) },
            Bind(3, "weights", matrix, isOutput: true), new[] { 0 }, CodegenReduceKind.None, scaleInput: 2,
            preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1));
        temps.Add((long)rows * seqK);

        // ---- out[qi,h,e] = sum_k weights[qi*heads+h, k] * vcache[table[page], pos, h/g, e]
        var valueSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("qi", numQueries), CodegenAxis.Parallel("h", heads),
            CodegenAxis.Parallel("e", headDim),
            CodegenAxis.Reduce("page", pages), CodegenAxis.Reduce("pos", blockSize));
        passes.Add(new CodegenKernelSpec("paged_prefill_output", valueSpace,
            new[]
            {
                new CodegenTensorBinding(0, "block_table", new[] { pages }, new[] { A(3) },
                    elementType: CodegenElementType.Int32),
                new CodegenTensorBinding(1, "weights", matrix, new[] { Row(0, 1), Key(3, 4) }),
                new CodegenTensorBinding(2, "vcache", cache,
                    new[] { CodegenAffineExpr.Const(0), A(4), KvHead(), A(2) },
                    indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, A(3), blocks), null, null, null }),
            },
            new CodegenTensorBinding(3, "out", new[] { numQueries, heads, headDim }, new[] { A(0), A(1), A(2) },
                isOutput: true),
            new[] { 1, 2 }, CodegenReduceKind.Sum));
        return new CodegenProgram(passes, temps, "paged attention prefill (backend contract)");
    }

    /// <summary>
    /// The backward of <see cref="ScaledDotProductAttention"/> (heads == kvHeads), in
    /// CudaBackend.ScaledDotProductAttentionBackward's contract: from the output gradient dout and the forward's stored
    /// weights P, <c>dv = P^T dout</c>, <c>dP = dout v^T</c>, <c>dS = scale * P * (dP - rowsum(P * dP))</c>,
    /// <c>dq = dS k</c>, <c>dk = dS^T q</c>, each overwriting its gradient. A masked key has P = 0, so it gets no
    /// gradient, as in the forward. Every product reads its operands in place -- no transposed copies.
    /// </summary>
    public static CodegenProgram ScaledDotProductAttentionBackward(int batch, int heads, int seqQ, int seqK, int headDim,
        double scale)
    {
        if (batch <= 0 || heads <= 0 || seqQ <= 0 || seqK <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Attention extents must be positive.");
        var passes = AttentionBackwardPasses(batch, heads, seqQ, seqK, headDim, scale, "weights", rowdotFromOutput: false);
        long p = (long)batch * heads * seqQ * seqK;
        return new CodegenProgram(passes, new[] { p, (long)batch * heads * seqQ, p },
            "scaled dot-product attention backward (backend contract)");
    }

    /// <summary>
    /// FlashAttentionBackward in its contract: the weights recomputed from the forward's row log-sum-exp
    /// (<c>P = exp(scale * q.k + causal - stats)</c>, the causal bias "causal" [seqQ, seqK] bound by the backend), the
    /// row term <c>D = sum_e dout * out</c> from the forward's output, then the same products as
    /// <see cref="ScaledDotProductAttentionBackward"/>.
    /// </summary>
    public static CodegenProgram FlashAttentionBackward(int batch, int heads, int seqQ, int seqK, int headDim, double scale,
        bool causal)
    {
        if (batch <= 0 || heads <= 0 || seqQ <= 0 || seqK <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Attention extents must be positive.");
        int[] pShape = { batch, heads, seqQ, seqK };
        int rows = checked(batch * heads * seqQ);
        var passes = new List<CodegenKernelSpec>();
        var scoreSpace = new CodegenIterationSpace(CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("h", heads),
            CodegenAxis.Parallel("q", seqQ), CodegenAxis.Parallel("k", seqK), CodegenAxis.Reduce("d", headDim));
        var scoreInputs = new List<CodegenTensorBinding>
        {
            new(0, "q", new[] { batch, heads, seqQ, headDim }, new[] { A(0), A(1), A(2), A(4) }),
            new(1, "k", new[] { batch, heads, seqK, headDim }, new[] { A(0), A(1), A(3), A(4) }),
        };
        if (causal) scoreInputs.Add(new CodegenTensorBinding(2, "causal", new[] { seqQ, seqK }, new[] { A(2), A(3) }));
        passes.Add(new CodegenKernelSpec("attention_scores", scoreSpace, scoreInputs.ToArray(),
            new CodegenTensorBinding(scoreInputs.Count, "logits", pShape, new[] { A(0), A(1), A(2), A(3) }, isOutput: true),
            new[] { 0, 1 }, CodegenReduceKind.Sum, biasInput: causal ? 2 : null, reduceScale: scale));
        int[] vector = { rows }, matrix = { rows, seqK };
        passes.Add(RowPass("attention_negstats", vector, new[] { "stats" }, "negstats", CodegenExpr.Negate(Op(0))));
        passes.Add(new CodegenKernelSpec("attention_recompute",
            new CodegenIterationSpace(CodegenAxis.Parallel("i", rows), CodegenAxis.Parallel("j", seqK)),
            new[] { Bind(0, "logits", matrix), Bind(1, "negstats", vector) }, Bind(2, "p", matrix, isOutput: true),
            new[] { 0 }, CodegenReduceKind.None, preReduce: CodegenPreReduceOp.Exp, preBiasInput: 1));
        passes.AddRange(AttentionBackwardPasses(batch, heads, seqQ, seqK, headDim, scale, "p", rowdotFromOutput: true));
        long pElements = (long)batch * heads * seqQ * seqK;
        return new CodegenProgram(passes, new[] { pElements, rows, pElements, pElements, rows, pElements },
            "flash attention backward (backend contract)");
    }

    // The backward's products from the weights tensor `weights`: dv, dP, the row term (sum_k P dP, or sum_e dout out),
    // dS, dq, dk.
    private static List<CodegenKernelSpec> AttentionBackwardPasses(int batch, int heads, int seqQ, int seqK, int headDim,
        double scale, string weights, bool rowdotFromOutput)
    {
        int[] qShape = { batch, heads, seqQ, headDim }, kShape = { batch, heads, seqK, headDim };
        int[] pShape = { batch, heads, seqQ, seqK };
        CodegenTensorBinding T(int parameter, string name, int[] shape, int a0, int a1, int a2, int a3, bool output = false) =>
            new(parameter, name, shape, new[] { A(a0), A(a1), A(a2), A(a3) }, output);
        CodegenIterationSpace Space(string m, int mExtent, string n, int nExtent, string r, int rExtent) =>
            new(CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("h", heads), CodegenAxis.Parallel(m, mExtent),
                CodegenAxis.Parallel(n, nExtent), CodegenAxis.Reduce(r, rExtent));
        var passes = new List<CodegenKernelSpec>();

        // dv[b,h,k,e] = sum_q P[b,h,q,k] dout[b,h,q,e]
        passes.Add(new CodegenKernelSpec("attention_dv", Space("k", seqK, "e", headDim, "q", seqQ),
            new[] { T(0, weights, pShape, 0, 1, 4, 2), T(1, "dout", qShape, 0, 1, 4, 3) },
            T(2, "dv", kShape, 0, 1, 2, 3, output: true), new[] { 0, 1 }, CodegenReduceKind.Sum));
        // dP[b,h,q,k] = sum_e dout[b,h,q,e] v[b,h,k,e]
        passes.Add(new CodegenKernelSpec("attention_dp", Space("q", seqQ, "k", seqK, "e", headDim),
            new[] { T(0, "dout", qShape, 0, 1, 2, 4), T(1, "v", kShape, 0, 1, 3, 4) },
            T(2, "dp", pShape, 0, 1, 2, 3, output: true), new[] { 0, 1 }, CodegenReduceKind.Sum));
        // dot[b,h,q] = sum_k P dP
        var dotSpace = new CodegenIterationSpace(CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("h", heads),
            CodegenAxis.Parallel("q", seqQ), CodegenAxis.Reduce("k", rowdotFromOutput ? headDim : seqK));
        passes.Add(new CodegenKernelSpec("attention_rowdot", dotSpace,
            rowdotFromOutput
                ? new[] { T(0, "dout", qShape, 0, 1, 2, 3), T(1, "out", qShape, 0, 1, 2, 3) }
                : new[] { T(0, weights, pShape, 0, 1, 2, 3), T(1, "dp", pShape, 0, 1, 2, 3) },
            new CodegenTensorBinding(2, "rowdot", new[] { batch, heads, seqQ }, new[] { A(0), A(1), A(2) }, isOutput: true),
            new[] { 0, 1 }, CodegenReduceKind.Sum));
        // dS = scale * P * (dP - dot)
        var dsSpace = new CodegenIterationSpace(CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("h", heads),
            CodegenAxis.Parallel("q", seqQ), CodegenAxis.Parallel("k", seqK));
        var dsBody = CodegenExpr.Multiply(CodegenExpr.Multiply(CodegenExpr.Constant(scale), Op(0)),
            CodegenExpr.Subtract(Op(1), Op(2)));
        passes.Add(new CodegenKernelSpec("attention_ds", dsSpace,
            new[]
            {
                T(0, weights, pShape, 0, 1, 2, 3), T(1, "dp", pShape, 0, 1, 2, 3),
                new CodegenTensorBinding(2, "rowdot", new[] { batch, heads, seqQ }, new[] { A(0), A(1), A(2) }),
            },
            T(3, "ds", pShape, 0, 1, 2, 3, output: true), dsBody.Operands(), CodegenReduceKind.None, body: dsBody));
        // dq[b,h,q,d] = sum_k dS[b,h,q,k] k[b,h,k,d]
        passes.Add(new CodegenKernelSpec("attention_dq", Space("q", seqQ, "d", headDim, "k", seqK),
            new[] { T(0, "ds", pShape, 0, 1, 2, 4), T(1, "k", kShape, 0, 1, 4, 3) },
            T(2, "dq", qShape, 0, 1, 2, 3, output: true), new[] { 0, 1 }, CodegenReduceKind.Sum));
        // dk[b,h,k,d] = sum_q dS[b,h,q,k] q[b,h,q,d]
        passes.Add(new CodegenKernelSpec("attention_dk", Space("k", seqK, "d", headDim, "q", seqQ),
            new[] { T(0, "ds", pShape, 0, 1, 4, 2), T(1, "q", qShape, 0, 1, 4, 3) },
            T(2, "dk", kShape, 0, 1, 2, 3, output: true), new[] { 0, 1 }, CodegenReduceKind.Sum));
        return passes;
    }

    private static CodegenAffineExpr A(int axis) => CodegenAffineExpr.Axis(axis);

    public static CodegenProgram LayerNorm(int rows, int columns, double epsilon) =>
        Append(CodegenFusedStatistics.LayerNorm(rows, columns, epsilon),
            RowPass("layernorm_save_mean", new[] { rows }, new[] { "negmean" }, "mean", CodegenExpr.Negate(Op(0))));

    /// <summary>CudaBackend.RmsNorm: sources x, gamma; outputs y, rms (saveRms = 1/rstd).</summary>
    public static CodegenProgram RmsNorm(int rows, int columns, double epsilon) =>
        Append(CodegenFusedStatistics.RmsNorm(rows, columns, epsilon),
            RowPass("rmsnorm_save_rms", new[] { rows }, new[] { "rstd" }, "rms", Op(0), CodegenActivationKind.Reciprocal));

    /// <summary>
    /// CudaBackend.GroupNorm over [batch, channels, spatial]: sources x, gamma, beta; outputs y, mean, rstd, each
    /// statistic [batch, groups].
    /// </summary>
    public static CodegenProgram GroupNorm(int batch, int channels, int groups, int spatial, double epsilon) =>
        Append(CodegenFusedStatistics.GroupNorm(batch, channels, groups, spatial, epsilon),
            RowPass("groupnorm_save_mean", new[] { batch, groups }, new[] { "negmean" }, "mean", CodegenExpr.Negate(Op(0))));

    /// <summary>
    /// CudaBackend.BatchNorm in training mode: sources x, gamma, beta, running_mean, running_var; outputs y, mean,
    /// rstd, running_mean_out and running_var_out. The backend binds running_mean and running_mean_out (and the
    /// variances) to the SAME buffer: each element is read and then written by one thread, as the hand-written kernel
    /// updates them in place.
    /// </summary>
    public static CodegenProgram BatchNorm2DTraining(int batch, int channels, int height, int width,
        double epsilon, double momentum)
    {
        if (!(momentum >= 0 && momentum <= 1)) throw new ArgumentOutOfRangeException(nameof(momentum));
        long count = (long)batch * height * width;
        // The running variance takes the UNBIASED variance, as the hand-written kernel (and PyTorch) do; a single
        // sample has none, so it is left unchanged by folding in the biased one.
        double unbias = count > 1 ? count / (double)(count - 1) : 1.0;
        var c = new[] { channels };
        var program = CodegenFusedStatistics.BatchNorm2D(batch, channels, height, width, epsilon);
        return Append(program,
            RowPass("batchnorm_save_mean", c, new[] { "negmean" }, "mean", CodegenExpr.Negate(Op(0))),
            // running_mean*(1 - m) + mean*m
            RowPass("batchnorm_running_mean", c, new[] { "running_mean", "mean" }, "running_mean_out",
                CodegenExpr.Add(CodegenExpr.Product(Op(0), CodegenExpr.Constant(1 - momentum)),
                                CodegenExpr.Product(Op(1), CodegenExpr.Constant(momentum)))),
            // var = m2s - dmean^2 (the program's own shifted variance), unbiased, folded in.
            RowPass("batchnorm_running_var", c, new[] { "running_var", "m2s", "dmean" }, "running_var_out",
                CodegenExpr.Add(
                    CodegenExpr.Product(Op(0), CodegenExpr.Constant(1 - momentum)),
                    CodegenExpr.Product(CodegenExpr.Constant(momentum * unbias),
                        CodegenExpr.Subtract(Op(1), CodegenExpr.Product(Op(2), Op(2)))))));
    }

    /// <summary>CudaBackend.LayerNormBackward: sources dy, x, gamma, mean, rstd; outputs dx, dgamma, dbeta.</summary>
    public static CodegenProgram LayerNormBackward(int rows, int columns) =>
        Prepend(CodegenFusedStatistics.LayerNormBackward(rows, columns),
            RowPass("layernorm_bwd_negmean", new[] { rows }, new[] { "mean" }, "negmean", CodegenExpr.Negate(Op(0))));

    /// <summary>CudaBackend.RmsNormBackward: sources dy, x, gamma, rms; outputs dx, dgamma.</summary>
    public static CodegenProgram RmsNormBackward(int rows, int columns) =>
        Prepend(CodegenFusedStatistics.RmsNormBackward(rows, columns),
            RowPass("rmsnorm_bwd_rstd", new[] { rows }, new[] { "rms" }, "rstd", Op(0), CodegenActivationKind.Reciprocal));

    /// <summary>
    /// CudaBackend's GroupNorm backward over [batch, channels, spatial]: sources dy, x, gamma, mean, rstd (each
    /// statistic [batch, groups], rstd = 1/sqrt(var + eps) as the forward saves it); outputs dx, dgamma, dbeta.
    /// </summary>
    public static CodegenProgram GroupNormBackward(int batch, int channels, int groups, int spatial) =>
        Prepend(CodegenFusedStatistics.GroupNormBackward(batch, channels, groups, spatial),
            RowPass("groupnorm_bwd_negmean", new[] { batch, groups }, new[] { "mean" }, "negmean",
                CodegenExpr.Negate(Op(0))));

    /// <summary>CudaBackend.BatchNormBackward: sources dy, x, gamma, mean, rstd; outputs dx, dgamma, dbeta.</summary>
    public static CodegenProgram BatchNorm2DBackward(int batch, int channels, int height, int width) =>
        Prepend(CodegenFusedStatistics.BatchNorm2DBackward(batch, channels, height, width),
            RowPass("batchnorm_bwd_negmean", new[] { channels }, new[] { "mean" }, "negmean", CodegenExpr.Negate(Op(0))));
}
