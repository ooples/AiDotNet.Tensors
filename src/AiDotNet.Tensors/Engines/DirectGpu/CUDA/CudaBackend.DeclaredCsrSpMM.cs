// Copyright (c) AiDotNet. All rights reserved.
// CSR SpMM through its declared program (#890 phase 4e, #1441).

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    // AIDOTNET_DECLARED_CSR_SPMM=1 opts CsrSpMM into the declared route; it is off by default until verified on an
    // NVIDIA GPU (see CsrSpMMDeclaredPromoted).
    private static readonly bool s_declaredCsrSpMMOptIn =
        Environment.GetEnvironmentVariable("AIDOTNET_DECLARED_CSR_SPMM") == "1";

    /// <summary>
    /// Whether the declared CSR SpMM (#1441) has been verified on an NVIDIA GPU. While false it runs only with
    /// AIDOTNET_DECLARED_CSR_SPMM=1 (or <see cref="DeclaredCsrSpMMSwitch"/>); the hand-written csr_spmm serves every
    /// call otherwise. Set it to true once DeclaredCsrSpMMRouteTests passes on the device: the route is then on by
    /// default, like every other declared family, and AIDOTNET_DECLARED_KERNELS=0 still turns it off.
    /// </summary>
    internal const bool CsrSpMMDeclaredPromoted = false;

    /// <summary>Overrides AIDOTNET_DECLARED_CSR_SPMM for this backend (verification tooling and tests).</summary>
    internal bool? DeclaredCsrSpMMSwitch { get; set; }

    private bool DeclaredCsrSpMMEnabled => DeclaredCsrSpMMSwitch ?? (CsrSpMMDeclaredPromoted || s_declaredCsrSpMMOptIn);

    /// <summary>
    /// The most CSR SpMM programs built per backend. Each bakes rows, columns, the dense width and nnz, and nnz varies
    /// with the data, so past the cap a new shape keeps the hand-written kernel.
    /// </summary>
    internal const int DeclaredCsrSpMMProgramCap = 32;

    /// <summary>
    /// The largest rows x nnz admitted. The inversion visits every stored entry for every row, against csr_spmm's own
    /// row's entries only, so it is admitted only where that product stays small, as the scatter inversions are.
    /// </summary>
    internal const long DeclaredCsrSpMMMaxRowEntries = 1L << 22;

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
    /// CsrSpMM through its declared program (#1441), with no atomic. Off by default
    /// (<see cref="CsrSpMMDeclaredPromoted"/>). False -- nothing launched -- when off or declined.
    /// </summary>
    internal bool TryDeclaredCsrSpMM(IGpuBuffer values, IGpuBuffer colIndices, IGpuBuffer rowPointers,
        IGpuBuffer dense, IGpuBuffer output, int rows, int cols, int n, int nnz)
    {
        if (!DeclaredCsrSpMMEnabled || rows <= 0 || cols <= 0 || n <= 0 || nnz <= 0 ||
            (long)rows * nnz > DeclaredCsrSpMMMaxRowEntries || rowPointers.Size < rows + 1 ||
            colIndices.Size < nnz || values.Size < nnz || (long)dense.Size < (long)cols * n ||
            (long)output.Size < (long)rows * n)
            return false;
        var key = CsrSpMMKey(rows, cols, n, nnz);
        lock (_directPtxLock)
        {
            if (!_declaredPrograms.ContainsKey(key))
            {
                if (_declaredCsrSpMMPrograms >= DeclaredCsrSpMMProgramCap) return false;
                _declaredCsrSpMMPrograms++;
            }
        }
        return TryRunDeclaredProgram(key, _buildCsrSpMM ??= BuildCsrSpMM, s_csrSpMMNames,
            new DeclaredBindings(rowPointers, colIndices, values, dense, output));
    }
}
