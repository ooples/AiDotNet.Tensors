// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CSR SpMM as a declared program (#890 phase 4e, #1441): an inversion over the stored entries, each admitted only
// inside its row's [rowPtr[r], rowPtr[r + 1]) range. The interpreter against csr_spmm's loop; the PTX for no atomic.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenCsrSpMMTests
{
    private static readonly CodegenLibraryFunction NotEqual =
        new("not_equal_scalar", "test", 2, a => a[0] != a[1] ? 1.0 : 0.0);

    [Theory]
    [InlineData(new[] { 0, 2, 2, 5, 6 }, 5, 3)]       // row 1 empty, between non-empty rows
    [InlineData(new[] { 0, 0, 0, 3 }, 4, 2)]          // the leading rows empty
    [InlineData(new[] { 0, 3, 3, 3 }, 3, 1)]          // the trailing rows empty
    [InlineData(new[] { 0, 1, 2, 3, 4 }, 4, 5)]       // one entry per row
    public void CsrSpMM_SumsEachRowsEntriesAndNothingElse(int[] rowPtr, int cols, int n)
    {
        int rows = rowPtr.Length - 1, nnz = rowPtr[rows];
        var colIdx = new int[nnz];
        var values = new double[nnz];
        for (int p = 0; p < nnz; p++)
        {
            colIdx[p] = (p * 3 + 1) % cols;
            values[p] = ((p * 5) % 7) - 3 + 0.5 * p;
        }
        var dense = new double[cols * n];
        for (int i = 0; i < dense.Length; i++) dense[i] = ((i * 11) % 9) - 4 + 0.25 * i;

        var outputs = CodegenBackendPrograms.CsrSpMM(rows, cols, n, nnz, NotEqual).Interpret(new Dictionary<string, double[]>
        {
            ["rowPtr"] = Array.ConvertAll(rowPtr, v => (double)v),
            ["colIdx"] = Array.ConvertAll(colIdx, v => (double)v),
            ["values"] = values,
            ["dense"] = dense,
        });

        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++)
            {
                double expected = 0;
                for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) expected += values[p] * dense[colIdx[p] * n + j];
                Assert.Equal(expected, outputs["out"][r * n + j], 12);
            }
    }

    /// <summary>
    /// csr_spmm reads only a row's own entries, so a non-finite dense row that only row 0's entry references must
    /// leave every other row finite; a rejected entry must contribute nothing, not 0 x Inf.
    /// </summary>
    [Fact]
    public void CsrSpMM_ANonFiniteDenseRowReachesOnlyTheRowsThatReferenceIt()
    {
        int[] rowPtr = { 0, 1, 3, 3, 4 };
        int[] colIdx = { 2, 0, 1, 1 };
        double[] values = { 1.5, -2, 0.5, 3 };
        const int cols = 3, n = 2;
        double[] dense = { 1, 2, 3, 4, double.PositiveInfinity, double.NaN };

        var outputs = CodegenBackendPrograms.CsrSpMM(4, cols, n, 4, NotEqual).Interpret(new Dictionary<string, double[]>
        {
            ["rowPtr"] = Array.ConvertAll(rowPtr, v => (double)v),
            ["colIdx"] = Array.ConvertAll(colIdx, v => (double)v),
            ["values"] = values,
            ["dense"] = dense,
        });

        Assert.Equal(double.PositiveInfinity, outputs["out"][0]);
        Assert.True(double.IsNaN(outputs["out"][1]));
        double[] expected = { -2 * 1 + 0.5 * 3, -2 * 2 + 0.5 * 4, 0, 0, 3 * 3, 3 * 4.0 };
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], outputs["out"][2 + i], 12);
    }

    [Fact]
    public void CsrSpMM_EmitsNoAtomicAndReadsTheRowPointersAsIndices()
    {
        foreach (var pass in CodegenBackendPrograms.CsrSpMM(32, 16, 8, 96, NotEqual).Passes)
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(pass, 7, 5);
            Assert.Equal(0, emitter.AtomicStores);
            Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
            if (pass.Name == "csr_spmm")
                Assert.True(emitter.IndirectIndexLoads >= 3, "rowPtr twice and colIdx once must be read as indices");
        }
    }
}
