// Copyright (c) AiDotNet. All rights reserved.
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