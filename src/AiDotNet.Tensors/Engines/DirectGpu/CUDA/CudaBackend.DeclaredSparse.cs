// Copyright (c) AiDotNet. All rights reserved.
// The rest of the sparse kernels through declared programs (#890 phase 4e, #1441): CSR SpMM with a bias (the warp and
// vec4 layouts route to the CSR SpMM itself), SDDMM, and the graph edge scatter-add. They share CSR SpMM's switch and
// cap; the inversions (bias SpMM, edges) share its size limit.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private static readonly string[] s_csrSpMMBiasNames = { "rowPtr", "colIdx", "values", "dense", "bias", "out" };
    private static readonly string[] s_csrSddmmNames = { "rowIndices", "colIndices", "x", "y", "out" };
    private static readonly string[] s_scatterAddEdgesNames = { "targetIndices", "sourceIndices", "input", "output" };
    private static readonly string[] s_scatterAddEdgesWeightedNames =
        { "targetIndices", "sourceIndices", "input", "edgeValues", "output" };

    private Func<DeclaredKey, CodegenProgram>? _buildCsrSpMMBias;

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildCsrSddmm =
        key => CodegenBackendPrograms.CsrSddmm(key.D0, key.D1, key.D2, key.D3);

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildScatterAddEdges =
        key => CodegenBackendPrograms.ScatterAddEdges(key.D0, key.D1, key.D2, weighted: key.D3 != 0);

    private CodegenProgram BuildCsrSpMMBias(DeclaredKey key) =>
        CodegenBackendPrograms.CsrSpMMBias(key.D0, key.D1, key.D2, key.D3, NotEqualScalar());

    /// <summary>CsrSpMMBias through its declared program (#1441). False -- nothing launched -- when declined.</summary>
    internal bool TryDeclaredCsrSpMMBias(IGpuBuffer values, IGpuBuffer colIndices, IGpuBuffer rowPointers,
        IGpuBuffer dense, IGpuBuffer bias, IGpuBuffer output, int rows, int cols, int n, int nnz)
    {
        if (!DeclaredCsrSpMMEnabled || rows <= 0 || cols <= 0 || n <= 0 || nnz <= 0 ||
            (long)rows * nnz > DeclaredCsrSpMMMaxRowEntries || rowPointers.Size < rows + 1 ||
            colIndices.Size < nnz || values.Size < nnz || (long)dense.Size < (long)cols * n || bias.Size < n ||
            (long)output.Size < (long)rows * n)
            return false;
        var key = new DeclaredKey(DeclaredFamily.CsrSpMMBias, rows, cols, n, nnz, 0, 0);
        if (!ClaimDeclaredSlot(key, ref _declaredCsrSpMMPrograms, DeclaredCsrSpMMProgramCap)) return false;
        return TryRunDeclaredProgram(key, _buildCsrSpMMBias ??= BuildCsrSpMMBias, s_csrSpMMBiasNames,
            new DeclaredBindings(rowPointers, colIndices, values, dense, bias, output));
    }

    /// <summary>
    /// CsrSddmm through its declared program (#1441). x's and y's row counts come from their sizes. False -- nothing
    /// launched -- when declined.
    /// </summary>
    internal bool TryDeclaredCsrSddmm(IGpuBuffer rowIndices, IGpuBuffer colIndices, IGpuBuffer x, IGpuBuffer y,
        IGpuBuffer output, int nnz, int innerK)
    {
        if (!DeclaredCsrSpMMEnabled || nnz <= 0 || innerK <= 0 || rowIndices.Size < nnz || colIndices.Size < nnz ||
            output.Size < nnz)
            return false;
        int xRows = x.Size / innerK, yRows = y.Size / innerK;
        if (xRows <= 0 || yRows <= 0) return false;
        var key = new DeclaredKey(DeclaredFamily.CsrSddmm, nnz, innerK, xRows, yRows, 0, 0);
        if (!ClaimDeclaredSlot(key, ref _declaredCsrSpMMPrograms, DeclaredCsrSpMMProgramCap)) return false;
        return TryRunDeclaredProgram(key, s_buildCsrSddmm, s_csrSddmmNames,
            new DeclaredBindings(rowIndices, colIndices, x, y, output));
    }

    /// <summary>
    /// ScatterAddEdges through its declared program (#1441): the edges' fixed-order sums with no atomic, so it serves
    /// deterministic mode and the default alike. Admitted where nodes x edges stays within the inversions' budget; the
    /// sorted and hand-written kernels serve larger graphs. It writes every output, so nothing needs zeroing first.
    /// </summary>
    internal bool TryDeclaredScatterAddEdges(IGpuBuffer input, IGpuBuffer sourceIndices, IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues, IGpuBuffer output, int numNodes, int numEdges, int features)
    {
        if (!DeclaredCsrSpMMEnabled || numNodes <= 0 || numEdges <= 0 || features <= 0 ||
            (long)numNodes * numEdges > DeclaredCsrSpMMMaxRowEntries || sourceIndices.Size < numEdges ||
            targetIndices.Size < numEdges || (long)input.Size < (long)numNodes * features ||
            (long)output.Size < (long)numNodes * features || (edgeValues is not null && edgeValues.Size < numEdges))
            return false;
        bool weighted = edgeValues is not null;
        var key = new DeclaredKey(DeclaredFamily.ScatterAddEdges, numNodes, numEdges, features, weighted ? 1 : 0, 0, 0);
        if (!ClaimDeclaredSlot(key, ref _declaredCsrSpMMPrograms, DeclaredCsrSpMMProgramCap)) return false;
        return weighted
            ? TryRunDeclaredProgram(key, s_buildScatterAddEdges, s_scatterAddEdgesWeightedNames,
                new DeclaredBindings(targetIndices, sourceIndices, input, edgeValues!, output))
            : TryRunDeclaredProgram(key, s_buildScatterAddEdges, s_scatterAddEdgesNames,
                new DeclaredBindings(targetIndices, sourceIndices, input, output));
    }
}