using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSolverCoverageStatus
{
    ExperimentalDirectPtx,
    CoveredByExperimentalPrimitive
}

internal sealed record DirectPtxSolverCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string LayoutAndDType,
    DirectPtxSolverCoverageStatus Status,
    string DirectPtxAssignment);

/// <summary>
/// Executable issue-#853 inventory. Every decomposition/solver entry point in
/// the issue has an exact direct-PTX route or is composed from an explicitly
/// named direct-PTX primitive; no public name is left implicit.
/// </summary>
internal static class DirectPtxSolverCoverageManifest
{
    internal static IReadOnlyList<DirectPtxSolverCoverageCell> All { get; } =
    [
        new("Linalg.Cholesky", "CudaBackend parity211_cholesky or managed CholeskyDecomposition",
            "lower/upper SPD factor; throws through CholeskyEx policy", "generic public; direct FP32 [B,4,4] lower",
            DirectPtxSolverCoverageStatus.CoveredByExperimentalPrimitive, "lower exact-batch route through CholeskyEx v1"),
        new("Linalg.CholeskyEx", "CudaBackend parity211_cholesky (NVRTC)",
            "factor plus first failing leading-minor info", "FP32 row-major [B,4,4] + int32 [B]",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "register Cholesky v1; upper and other extents fail closed"),
        new("CudaBackend.LinalgCholesky", "parity211_cholesky NVRTC", "batched lower/upper Cholesky",
            "FP32 row-major", DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "pointer-only lower 4x4 batch=1024/4096/16384/65536"),
        new("Linalg.LuFactor", "CudaBackend parity211_lu_factor or managed Doolittle",
            "packed L/U plus zero-based partial pivots", "FP32 [B,4,4] + int32 [B,4]",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "register pivoted LU v1"),
        new("CudaBackend.LinalgLuFactor", "parity211_lu_factor NVRTC",
            "batched rectangular partial-pivot LU", "FP32 row-major",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "exact square 4x4 register LU; all other shapes fallback"),
        new("Linalg.LU", "LuDecomposition.Compute via packed LuFactor then unpack",
            "explicit P,L,U", "generic public tensors",
            DirectPtxSolverCoverageStatus.CoveredByExperimentalPrimitive,
            "packed factor primitive is direct PTX; explicit unpack remains established tensor work"),
        new("Linalg.LuSolve", "managed LuDecomposition.Solve",
            "apply zero-based pivots then forward/back substitution", "FP32 LU [B,4,4], pivots [B,4], vector RHS [B,4]",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "register vector LU solve v1 through extended backend"),
        new("IExtendedLinalgBackend.LinalgLuSolve", "no prior CUDA route",
            "pre-factored LU solve", "FP32 exact 4x4, one vector RHS",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "pointer-only register LU solve v1"),
        new("Linalg.QR(reduced)", "CudaBackend parity211_qr_reduced or managed Householder",
            "reduced Q,R with rank-deficient columns zeroed on CUDA baseline", "FP32 [B,4,4] Q/R",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "register modified Gram-Schmidt v1"),
        new("CudaBackend.LinalgQrReduced", "parity211_qr_reduced NVRTC",
            "batched reduced QR", "FP32 row-major",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "exact square 4x4 register QR; other shapes fallback"),
        new("IExtendedLinalgBackend.LinalgQrReducedDirect", "no prior direct-only route",
            "direct-only reduced QR for composed operations", "FP32 exact square 4x4",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "pointer-only register QR v1"),
        new("Linalg.QR(complete/r)", "managed QrDecomposition", "complete Q or R-only modes",
            "FP32 square [B,4,4]", DirectPtxSolverCoverageStatus.CoveredByExperimentalPrimitive,
            "square complete/reduced extents coincide; R-only projects the register QR R output"),
        new("Linalg.Eigh(upper)", "CudaBackend parity211_eigh or managed Jacobi",
            "ascending eigenvalues, eigenvectors as columns, upper triangle authoritative", "FP32 [B,4,4]",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "register cyclic-Jacobi v1"),
        new("CudaBackend.LinalgEigh", "parity211_eigh NVRTC",
            "batched symmetric eigendecomposition", "FP32 row-major",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "exact 4x4 upper-triangle register Eigh; other sizes fallback"),
        new("Linalg.Eigh(lower)", "managed lower-authoritative Jacobi",
            "ascending eigenvalues and column eigenvectors", "FP32 [B,4,4]",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "separate lower-authoritative register Jacobi entry point"),
        new("IExtendedLinalgBackend.LinalgEighSymmetric", "upper established CUDA; lower managed",
            "upper/lower authoritative symmetric Eigh", "FP32 exact square 4x4",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "separate pointer-only upper/lower register Jacobi ABIs"),
        new("Linalg.Eigvalsh", "Linalg.Eigh values projection", "symmetric eigenvalues only",
            "generic public tensors", DirectPtxSolverCoverageStatus.CoveredByExperimentalPrimitive,
            "routes through the eligible direct Eigh primitive and returns its values output"),
        new("Linalg.Svd", "managed SvdWrapper", "full/reduced SVD, descending singular values",
            "FP32 square [B,4,4]", DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "AtA Jacobi + U reconstruction register SVD v1; square reduced/full extents coincide"),
        new("Linalg.SvdVals", "managed SvdWrapper.ValuesOnly", "descending singular values only",
            "FP32 square [B,4,4]", DirectPtxSolverCoverageStatus.CoveredByExperimentalPrimitive,
            "routes through direct register SVD and returns S"),
        new("IExtendedLinalgBackend.LinalgSvdReduced", "no prior CUDA route",
            "U,S,Vh reduced SVD", "FP32 exact square 4x4",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "pointer-only register SVD v1"),
        new("Linalg.SvdLowRank", "managed randomized subspace iteration", "rank-q approximate SVD",
            "FP32 square [B,4,4]", DirectPtxSolverCoverageStatus.CoveredByExperimentalPrimitive,
            "current q-reserved semantics route through register SVD and truncate top-k outputs"),
        new("Linalg.LdlFactor", "managed pivoted LDL decomposition", "lower packed LDL plus pivots",
            "FP32 [B,4,4] + int32 [B,4]", DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "register symmetric-diagonal-pivot LDL factor v1"),
        new("Linalg.LdlSolve", "managed LDL solve", "lower LDL vector solve",
            "FP32 [B,4,4] factor and [B,4] RHS", DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "register three-phase lower LDL vector solve v1"),
        new("IExtendedLinalgBackend.LinalgLdlFactor/LinalgLdlSolve", "no prior CUDA route",
            "lower pivoted LDL factor and vector solve", "FP32 exact square 4x4",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "pointer-only register LDL factor/solve ABIs"),
        new("Linalg.Solve", "managed structure-aware LinearSolvers", "general unfactored vector solve",
            "FP32 [B,4,4] and [B,4]", DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "fused register partial-pivot factor-plus-solve v1"),
        new("Linalg.SolveEx", "managed LU factor/solve plus info", "solution plus first zero-pivot info",
            "FP32 [B,4,4], [B,4], int32 [B]", DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "same fused general-solve kernel writes solution and info"),
        new("Linalg.SolveTriangular", "managed BLAS TRSM", "upper/lower non-unit vector solve",
            "FP32 [B,4,4] and [B,4]", DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "separate upper/lower register substitution entry points"),
        new("IExtendedLinalgBackend.LinalgSolveVector/LinalgTriangularSolveVector", "no prior CUDA route",
            "general and triangular vector solve", "FP32 exact square 4x4",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx, "pointer-only fused general and substitution solve ABIs"),
        new("Linalg.Lstsq", "managed QR least squares", "square vector least squares and diagnostics",
            "FP32 [B,4,4] and [B,4]", DirectPtxSolverCoverageStatus.CoveredByExperimentalPrimitive,
            "direct fused solve plus register QR composition; public diagnostics preserve established square semantics"),
        new("Linalg.Cholesky backward", "LinalgBackward.CholeskyBackward using engine tensor primitives",
            "Murray/Stan Cholesky derivative", "same logical tensor contract as forward",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "fused lower 4x4 register Cholesky-gradient kernel v1"),
        new("Linalg.Solve backward", "LinalgBackward.SolveBackward primitive graph",
            "transpose solve plus negative outer-product gradient", "FP32 matrix [B,4,4] and vectors [B,4]",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "fused register vector-solve backward writes gradA and gradB"),
        new("IExtendedLinalgBackend solver backward", "no prior fused CUDA route",
            "Cholesky and general vector-solve gradients", "FP32 exact square 4x4",
            DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
            "pointer-only fused CholeskyBackwardLower and SolveBackwardVector ABIs")
    ];

    internal static DirectPtxSolverCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        foreach (DirectPtxSolverCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException($"Solver API '{api}' is not assigned in the #853 coverage manifest.");
    }
}
