// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The rest of the sparse kernels as declared programs (#1441) on the fp64 interpreter: CSR SpMM with a bias, SDDMM and
// the graph edge scatter-add, each against the hand-written kernel's loop.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenSparseFamilyTests
{
    private static readonly CodegenLibraryFunction NotEqual =
        new("not_equal_scalar", "test", 2, a => a[0] != a[1] ? 1.0 : 0.0);

    private static double[] D(params int[] v) => Array.ConvertAll(v, x => (double)x);

    [Fact]
    public void CsrSpMMBias_AddsTheColumnBiasToEveryRowIncludingEmptyOnes()
    {
        int[] rowPtr = { 0, 2, 2, 3 };
        int[] colIdx = { 1, 0, 2 };
        double[] values = { 2, -1, 0.5 };
        double[] dense = { 1, 2, 3, 4, 5, 6 };   // [3, 2]
        double[] bias = { 10, 20 };
        var outputs = CodegenBackendPrograms.CsrSpMMBias(3, 3, 2, 3, NotEqual).Interpret(new Dictionary<string, double[]>
        {
            ["rowPtr"] = D(rowPtr), ["colIdx"] = D(colIdx), ["values"] = values, ["dense"] = dense, ["bias"] = bias,
        });
        var expected = new double[6];
        for (int r = 0; r < 3; r++)
            for (int j = 0; j < 2; j++)
            {
                double sum = bias[j];
                for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) sum += values[p] * dense[colIdx[p] * 2 + j];
                expected[r * 2 + j] = sum;
            }
        Assert.Equal(expected, outputs["out"]);
    }

    [Fact]
    public void CsrSpMMDouble_SumsInDoublePrecisionAndLowersToF64()
    {
        int[] rowPtr = { 0, 2, 2, 3 };
        int[] colIdx = { 1, 0, 2 };
        double[] values = { 1.0 / 3, -2.0 / 7, 1e-9 };
        double[] dense = { 1, 2, 3, 4, 5, double.PositiveInfinity };
        var program = CodegenBackendPrograms.CsrSpMMDouble(3, 3, 2, 3);
        var outputs = program.Interpret(new Dictionary<string, double[]>
        {
            ["rowPtr"] = D(rowPtr), ["colIdx"] = D(colIdx), ["values"] = values, ["dense"] = dense,
        });
        var expected = new double[6];
        for (int r = 0; r < 3; r++)
            for (int j = 0; j < 2; j++)
                for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) expected[r * 2 + j] += values[p] * dense[colIdx[p] * 2 + j];
        Assert.Equal(expected, outputs["out"]);

        string ptx = new AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx.PtxAffineEmitter().Emit(program.Passes[0], 8, 6);
        Assert.Contains("ld.global.nc.f64", ptx);
        Assert.Contains("mul.rn.f64", ptx);
        Assert.Contains("add.rn.f64", ptx);
        Assert.Contains("st.global.f64", ptx);
        Assert.DoesNotContain("ld.global.nc.f32", ptx);
        Assert.False(CodegenSplitReduction.IsSplittable(program.Passes[0]));
        Assert.Null(CodegenRowFusion.TryPlan(program, out _));
    }

    /// <summary>
    /// A CSR row's entries are a data-dependent reduction extent: the plain lowering loads [rowPtr[r], rowPtr[r + 1])
    /// and walks only those entries, skipping an empty row's body, instead of testing every stored entry per row.
    /// </summary>
    [Fact]
    public void CsrSpMM_WalksOnlyItsOwnRowsEntries()
    {
        var pass = CodegenBackendPrograms.CsrSpMM(64, 48, 16, 300, NotEqual).Passes[0];
        Assert.NotNull(pass.ReductionRange);
        string ptx = new AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx.PtxAffineEmitter().Emit(pass, 8, 6);
        Assert.Contains("(ranged)", ptx);
        Assert.Contains("PLAINSKIP0", ptx);
        Assert.Matches(@"setp\.lt\.s32 %p\d+, %r\d+, %r\d+;", ptx);   // the loop ends at the loaded bound
        Assert.DoesNotMatch(@"setp\.lt\.s32 %p\d+, %r\d+, 300;", ptx);  // never at the declared nnz extent (only clamps)
        Assert.False(CodegenSplitReduction.IsSplittable(pass));
        Assert.Null(CodegenRowFusion.TryPlan(new CodegenProgram(new[] { pass }, Array.Empty<long>(), "p"), out _));
    }

    [Fact]
    public void Sddmm_DotsTheSampledRowsAndColumns()
    {
        int[] rows = { 0, 2, 1, 2 }, cols = { 1, 0, 1, 3 };
        double[] x = { 1, 2, 3, 4, 5, 6 };          // [3, 2]
        double[] y = { 1, -1, 2, 0.5, 3, 3, -2, 4 }; // [4, 2]
        var outputs = CodegenBackendPrograms.CsrSddmm(4, 2, 3, 4).Interpret(new Dictionary<string, double[]>
        {
            ["rowIndices"] = D(rows), ["colIndices"] = D(cols), ["x"] = x, ["y"] = y,
        });
        var expected = new double[4];
        for (int p = 0; p < 4; p++)
            for (int k = 0; k < 2; k++) expected[p] += x[rows[p] * 2 + k] * y[cols[p] * 2 + k];
        Assert.Equal(expected, outputs["out"]);
    }

    private static readonly CodegenLibraryFunction Max = new("max_vectors", "test", 2, a => a[0] > a[1] ? a[0] : a[1]);
    private static readonly CodegenLibraryFunction Min = new("min_vectors", "test", 2, a => a[0] < a[1] ? a[0] : a[1]);
    private static readonly CodegenLibraryFunction Divide = new("divide_vectors", "test", 2, a => a[1] != 0 ? a[0] / a[1] : 0);
    private static readonly CodegenLibraryFunction Sqrt = new("sqrt_vector", "test", 1, a => Math.Sqrt(a[0]));
    private static readonly CodegenLibraryFunction Select = new("where_select", "test", 3, a => a[0] != 0 ? a[1] : a[2]);

    // 4 rows (row 1 empty), 3 input rows, 2 columns; 6 column-index slots, the last past the final row pointer.
    private static readonly double[] SegRowPtr = { 0, 2, 2, 4, 5 };
    private static readonly double[] SegCol = { 2.7, 0, 1, 1, 2, 0 };   // float-encoded, truncated as (int)
    private static readonly double[] SegInput = { 1, -2, double.NegativeInfinity, 4, 5, double.NaN };

    private static Dictionary<string, double[]> SegSources() => new()
    {
        ["colIndices"] = SegCol, ["rowPointers"] = SegRowPtr, ["input"] = SegInput,
    };

    private static double SegReference(int r, int j, Func<List<double>, double> reduce)
    {
        var values = new List<double>();
        for (int p = (int)SegRowPtr[r]; p < (int)SegRowPtr[r + 1]; p++) values.Add(SegInput[(int)SegCol[p] * 2 + j]);
        return values.Count == 0 ? 0 : reduce(values);
    }

    [Fact]
    public void CsrSegmentedMaxAndMin_MatchTheKernelsLoopsWithEmptyRowsAndNonFiniteValues()
    {
        var max = CodegenBackendPrograms.CsrSegmentedMax(4, 3, 2, 6, NotEqual, Max).Interpret(SegSources())["out"];
        var min = CodegenBackendPrograms.CsrSegmentedMin(4, 3, 2, 6, NotEqual, Min).Interpret(SegSources())["out"];
        for (int r = 0; r < 4; r++)
            for (int j = 0; j < 2; j++)
            {
                // fmaxf/fminf from the kernel's +-3.402823e38 start: NaN never wins, -inf/+inf clamp to the start.
                double m = SegReference(r, j, v => { double a = (float)-3.402823e+38; foreach (var x in v) if (x > a) a = x; return a; });
                double n = SegReference(r, j, v => { double a = (float)3.402823e+38; foreach (var x in v) if (x < a) a = x; return a; });
                Assert.Equal(m, max[r * 2 + j]);
                Assert.Equal(n, min[r * 2 + j]);
            }
    }

    [Fact]
    public void CsrSegmentedStdDev_IsTheTwoPassStandardDeviation()
    {
        var input = new double[] { 1, -2, 3, 4, 5, 0.5 };
        var sources = SegSources();
        sources["input"] = input;
        var std = CodegenBackendPrograms.CsrSegmentedStdDev(4, 3, 2, 6, 1e-8f, NotEqual, Divide, Sqrt, Select).Interpret(sources)["out"];
        for (int r = 0; r < 4; r++)
            for (int j = 0; j < 2; j++)
            {
                var values = new List<double>();
                for (int p = (int)SegRowPtr[r]; p < (int)SegRowPtr[r + 1]; p++) values.Add(input[(int)SegCol[p] * 2 + j]);
                double expected = 0;
                if (values.Count > 0)
                {
                    double mean = 0;
                    foreach (var v in values) mean += v;
                    mean /= values.Count;
                    double acc = 0;
                    foreach (var v in values) acc += (v - mean) * (v - mean);
                    expected = Math.Sqrt(acc / values.Count + (double)1e-8f);
                }
                Assert.Equal(expected, std[r * 2 + j], 12);
            }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScatterAddEdges_SumsEachTargetsIncomingEdgesInEdgeOrder(bool weighted)
    {
        int[] src = { 0, 2, 1, 3, 0 }, tgt = { 1, 1, 0, 1, 3 };
        double[] input = { 1, 2, 3, 4, 5, 6, double.PositiveInfinity, 8 };   // [4 nodes, 2]
        double[] weights = { 0.5, 2, -1, 0, 3 };
        var sources = new Dictionary<string, double[]>
        {
            ["targetIndices"] = D(tgt), ["sourceIndices"] = D(src), ["input"] = input,
        };
        if (weighted) sources["edgeValues"] = weights;
        var outputs = CodegenBackendPrograms.ScatterAddEdges(4, 5, 2, weighted).Interpret(sources);
        var expected = new double[8];
        for (int e = 0; e < 5; e++)
            for (int f = 0; f < 2; f++)
                expected[tgt[e] * 2 + f] += input[src[e] * 2 + f] * (weighted ? weights[e] : 1);
        // Node 2 receives no edge: exactly 0. Edge 3 (weight 0) reads +Inf for target 1: Inf * 0 is NaN, as there.
        Assert.Equal(0.0, outputs["output"][4]);
        for (int k = 0; k < 8; k++)
        {
            if (double.IsNaN(expected[k])) Assert.True(double.IsNaN(outputs["output"][k]), "index " + k);
            else Assert.Equal(expected[k], outputs["output"][k]);
        }
    }
}