// Copyright (c) AiDotNet. All rights reserved.
// CSR SpMM through its declared program (#890 phase 4e, #1441).

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    // AIDOTNET_DECLARED_CSR_SPMM=1 opts CsrSpMM (every layout, fp64 and the segmented reductions) into the declared
    // route. It is verified on an NVIDIA GPU but off by default: walking each row's entries it measured 0.45-1.0x of the
    // hand-written kernels on an RTX 3080, never robustly faster (see CsrSpMMDeclaredPromoted).
    private static readonly bool s_declaredCsrSpMMOptIn =
        Environment.GetEnvironmentVariable("AIDOTNET_DECLARED_CSR_SPMM") == "1";

    /// <summary>
    /// Whether the declared CSR SpMM family (#1441) runs by default. It is verified on an NVIDIA GPU
    /// (DeclaredCsrSpMMRouteTests, DeclaredSparseFamilyRouteTests) but measured no faster than the hand-written kernels
    /// (--kernel-family-gate: 0.45-1.0x), so it stays false: it runs only with AIDOTNET_DECLARED_CSR_SPMM=1 or
    /// <see cref="DeclaredCsrSpMMSwitch"/>. SDDMM and the small edge scatter are gated separately (CsrFamilyAllowed).
    /// </summary>
    internal const bool CsrSpMMDeclaredPromoted = false;

    /// <summary>Overrides AIDOTNET_DECLARED_CSR_SPMM for this backend (verification tooling and tests).</summary>
    internal bool? DeclaredCsrSpMMSwitch { get; set; }

    private bool DeclaredCsrSpMMEnabled => DeclaredCsrSpMMSwitch ?? (CsrSpMMDeclaredPromoted || s_declaredCsrSpMMOptIn);

    /// <summary>
    /// The routes of this family that measured faster than their hand-written kernels at some shapes (SDDMM, the small
    /// edge scatter) run there by default, whatever <see cref="CsrSpMMDeclaredPromoted"/> says; DeclaredCsrSpMMSwitch =
    /// false still turns them off, and forcing the family (switch true or the env opt-in) runs them at every shape.
    /// </summary>
    private bool CsrFamilyAllowed => DeclaredCsrSpMMSwitch != false;

    private bool CsrFamilyForced => DeclaredCsrSpMMSwitch == true || s_declaredCsrSpMMOptIn;

    /// <summary>
    /// The most CSR SpMM programs built per backend. Each bakes rows, columns, the dense width and nnz, and nnz varies
    /// with the data, so past the cap a new shape keeps the hand-written kernel.
    /// </summary>
    internal const int DeclaredCsrSpMMProgramCap = 32;

    /// <summary>
    /// The largest nodes x edges an edge scatter admits (an inversion: every edge for every node), the scatter
    /// inversions' budget. CSR SpMM and the segmented reductions walk their own rows' entries and carry no such limit.
    /// </summary>
    internal const long DeclaredCsrSpMMMaxRowEntries = DeclaredScatterRowsWorkLimit;

    private int _declaredCsrSpMMPrograms;

    private static readonly string[] s_csrSpMMNames = { "rowPtr", "colIdx", "values", "dense", "out" };

    private Func<DeclaredKey, CodegenProgram>? _buildCsrSpMM;

    // The ones pass calls the derived not_equal_scalar, so the builder needs this backend.
    private CodegenProgram BuildCsrSpMM(DeclaredKey key)
    {
        var notEqual = (ResolveElementwise("not_equal_scalar")
            ?? throw new NotSupportedException("not_equal_scalar has no derived function: " +
                ElementwiseRefusal("not_equal_scalar"))).Handle;
        return CodegenBackendPrograms.CsrSpMM(key.D0, key.D1, key.D2, key.D3, notEqual);
    }

    /// <summary>The key of a CSR SpMM: rows, columns, the dense width, nnz.</summary>
    internal static DeclaredKey CsrSpMMKey(int rows, int cols, int n, int nnz) =>
        new(DeclaredFamily.CsrSpMM, rows, cols, n, nnz, 0, 0);

    /// <summary>
    /// CsrSpMM through its declared program (#1441), with no atomic. On by default
    /// (<see cref="CsrSpMMDeclaredPromoted"/>). False -- nothing launched -- when off or declined.
    /// </summary>
    internal bool TryDeclaredCsrSpMM(IGpuBuffer values, IGpuBuffer colIndices, IGpuBuffer rowPointers,
        IGpuBuffer dense, IGpuBuffer output, int rows, int cols, int n, int nnz)
    {
        if (!DeclaredCsrSpMMEnabled || rows <= 0 || cols <= 0 || n <= 0 || nnz <= 0 ||
            rowPointers.Size < rows + 1 ||
            colIndices.Size < nnz || values.Size < nnz || (long)dense.Size < (long)cols * n ||
            (long)output.Size < (long)rows * n)
            return false;
        var key = CsrSpMMKey(rows, cols, n, nnz);
        if (!ClaimDeclaredSlot(key, ref _declaredCsrSpMMPrograms, DeclaredCsrSpMMProgramCap)) return false;
        return TryRunDeclaredProgram(key, _buildCsrSpMM ??= BuildCsrSpMM, s_csrSpMMNames,
            new DeclaredBindings(rowPointers, colIndices, values, dense, output), residentWhenSmall: true);
    }
}
