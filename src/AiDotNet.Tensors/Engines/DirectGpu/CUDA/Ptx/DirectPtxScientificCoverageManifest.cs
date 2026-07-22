using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxScientificCoverageStatus
{
    BaselineOnly,
    ExperimentalDirectPtx,
    PlannedDirectPtx
}

/// <summary>
/// One auditable assignment in issue #854 (specialized scientific, hypercomplex, hyperbolic
/// and quantum kernels). Code rather than a markdown snapshot so tests reject duplicate,
/// missing, or accidentally unassigned domain-specific entry points as the backend evolves.
/// </summary>
internal sealed record DirectPtxScientificCoverageCell(
    string Api,
    string ExistingCudaImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    DirectPtxScientificCoverageStatus Status,
    string DirectPtxAssignment);

internal static class DirectPtxScientificCoverageManifest
{
    private const string InterleavedComplex = "contiguous interleaved [real,imag] pairs";
    private const string SplitComplex = "separate real/imag buffers";
    private const string Planned =
        "shape/dtype-specific PTX kernel; each domain op is an independently benchmarked specialization";

    internal static IReadOnlyList<DirectPtxScientificCoverageCell> All { get; } =
    [
        new("CudaBackend.ComplexMultiply", "NVRTC complex multiply",
            "(ar+ai j)(br+bi j)", InterleavedComplex, "FP32",
            DirectPtxScientificCoverageStatus.ExperimentalDirectPtx,
            "elementwise interleaved-complex PTX kernel (PtxComplexMultiplyKernel); fails closed until GPU-validated and promoted"),
        new("CudaBackend.ComplexConjugate", "NVRTC complex conjugate", "(ar, -ai)", InterleavedComplex, "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "elementwise interleaved-complex PTX kernel (PtxComplexConjugateKernel); fails closed until GPU-validated and promoted"),
        new("CudaBackend.ComplexMagnitude", "NVRTC complex magnitude", "sqrt(re^2+im^2)", SplitComplex, "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "elementwise split-complex PTX kernel (PtxComplexMagnitudeKernel) using sqrt.rn.f32; fails closed until GPU-validated and promoted"),
        new("CudaBackend.ComplexPhase", "NVRTC complex phase", "atan2(im, re)", SplitComplex, "FP32", DirectPtxScientificCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.MobiusAdd", "NVRTC mobius add", "Poincare-ball Mobius addition", "Poincare vectors", "FP32", DirectPtxScientificCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.PoincareProject", "NVRTC poincare project", "project onto the Poincare ball", "Poincare vectors", "FP32", DirectPtxScientificCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.PoincareExpMap", "NVRTC poincare exp-map", "tangent-space exponential map", "Poincare vectors", "FP32", DirectPtxScientificCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.PoincareDistance", "NVRTC poincare distance", "hyperbolic geodesic distance", "Poincare vectors", "FP32", DirectPtxScientificCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.OctonionMultiply", "NVRTC octonion multiply", "8-component hypercomplex product", "octonion 8-tuples", "FP32", DirectPtxScientificCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.OctonionAdd", "NVRTC octonion add", "componentwise octonion sum", "octonion 8-tuples", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "one-thread-per-octonion 8-lane add PTX kernel (PtxOctonionAddKernel); fails closed until GPU-validated and promoted")
    ];

    internal static DirectPtxScientificCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        foreach (DirectPtxScientificCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException($"Scientific API '{api}' is not assigned in the #854 coverage manifest.");
    }
}
