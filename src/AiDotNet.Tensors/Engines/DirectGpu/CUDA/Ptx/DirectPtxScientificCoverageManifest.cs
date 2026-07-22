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
        new("CudaBackend.ComplexPhase", "NVRTC complex phase", "atan2(im, re)", SplitComplex, "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "elementwise PTX kernel (PtxComplexPhaseKernel): minimax atan poly + quadrant folding; fails closed until GPU-validated and promoted"),
        new("CudaBackend.MobiusAdd", "NVRTC mobius add", "Poincare-ball Mobius addition", "Poincare vectors", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "one-block-per-vector 128-lane tree-reduced PTX kernel (PtxMobiusAddKernel) matching the NVRTC formula; fails closed until GPU-validated and promoted"),
        new("CudaBackend.PoincareProject", "NVRTC poincare project", "project onto the Poincare ball", "Poincare vectors", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-vector PTX kernel (PtxPoincareProjectKernel): rescale by maxNorm/||x|| when ||x||^2>=maxNorm^2, else copy; fails closed until GPU-validated and promoted"),
        new("CudaBackend.PoincareExpMap", "NVRTC poincare exp-map", "tangent-space exponential map", "Poincare vectors", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-vector PTX kernel (PtxPoincareExpMapKernel): tanh-scaled Mobius combination matching the NVRTC exp-map; fails closed until GPU-validated and promoted"),
        new("CudaBackend.PoincareDistance", "NVRTC poincare distance", "hyperbolic geodesic distance", "Poincare vectors", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "one-block-per-vector 4-reduction PTX kernel (PtxPoincareDistanceKernel) reusing the Mobius coeffs plus arctanh via lg2; fails closed until GPU-validated and promoted"),
        new("CudaBackend.OctonionMultiply", "NVRTC octonion multiply", "8-component hypercomplex product", "octonion 8-tuples", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "register-resident Cayley-Dickson PTX kernel (PtxOctonionMultiplyKernel) matching the NVRTC table; fails closed until GPU-validated and promoted"),
        new("CudaBackend.OctonionAdd", "NVRTC octonion add", "componentwise octonion sum", "octonion 8-tuples", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "one-thread-per-octonion 8-lane add PTX kernel (PtxOctonionAddKernel); fails closed until GPU-validated and promoted"),
        new("CudaBackend.RbfForward", "NVRTC rbf_forward", "exp(-epsilon_c * ||x_b - center_c||^2)", "row-major [batch,dim] / [centers,dim]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-(batch,center) serial-dim PTX kernel (PtxRbfForwardKernel) with expf via ex2.approx; fails closed until GPU-validated and promoted"),
        new("CudaBackend.PairwiseDistance", "NVRTC pairwise_distance", "output[i,j] = ||a[i] - b[j]||", "row-major [M,dim] / [N,dim]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-(i,j) serial-dim PTX kernel (PtxPairwiseDistanceKernel, L2 variant) with sqrt.rn.f32; fails closed until GPU-validated and promoted"),
        new("CudaBackend.PairwiseDistanceSquared", "NVRTC pairwise_distance_squared", "output[i,j] = ||a[i] - b[j]||^2", "row-major [M,dim] / [N,dim]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-(i,j) serial-dim PTX kernel (PtxPairwiseDistanceKernel, squared variant); fails closed until GPU-validated and promoted"),
        new("CudaBackend.QuantumMeasurement", "NVRTC quantum_measurement", "probabilities[i] = real[i]^2 + imag[i]^2 (unnormalized)", SplitComplex, "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "elementwise split-complex PTX kernel (PtxQuantumMeasurementKernel); normalization stays a separate reduction op; fails closed until GPU-validated and promoted"),
        new("CudaBackend.ComplexMatVec", "NVRTC complex_matvec", "y[b] = A * x[b] over complex (shared [dim,dim] matrix)", "split real/imag; [dim,dim] matrix, [batch,dim] vectors", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-(batch,row) serial-column PTX kernel (PtxComplexMatVecKernel) with complex FMA accumulation; fails closed until GPU-validated and promoted"),
        new("CudaBackend.SphericalHarmonics", "NVRTC spherical_harmonics", "clamp01(sum_b coeff[i,b,ch] * SH_basis_b(dir_i))", "row-major coeff [points,basis,channels]; dir [points,3]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-(point,channel) PTX kernel (PtxSphericalHarmonicsKernel): register-resident degree-0..3 real SH basis + unrolled coeff dot-product + clamp01; fails closed until GPU-validated and promoted"),
        new("CudaBackend.SphericalHarmonicsBackward", "NVRTC spherical_harmonics_backward", "shGrad[i,b,ch] = clampMask * outGrad[i,ch] * SH_basis_b(dir_i)", "row-major coeff/grad [points,basis,channels]; dir [points,3]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-(point,basis,channel) PTX kernel (PtxSphericalHarmonicsBackwardKernel): recomputes the SH basis + forward pre-clamp mask, selects basis[b] via a bounded selp chain; fails closed until GPU-validated and promoted"),
        new("CudaBackend.CapsulePredictions", "NVRTC capsule_predictions", "output[b,i,c,d] = sum_k input[b,i,k] * weights[i,c,k,d]", "row-major input [B,I,K]; weights [I,C,K,D]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-output-element serial-contraction PTX kernel (PtxCapsuleContractionKernel, Predictions variant); fails closed until GPU-validated and promoted"),
        new("CudaBackend.CapsuleTransform", "NVRTC capsule_transform", "output[b,i,j,d] = sum_k input[b,i,k] * weights[i,k,j,d]", "row-major input [B,I,K]; weights [I,K,J,D]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-output-element serial-contraction PTX kernel (PtxCapsuleContractionKernel, Transform variant); fails closed until GPU-validated and promoted")
    ];

    internal static DirectPtxScientificCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        foreach (DirectPtxScientificCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException($"Scientific API '{api}' is not assigned in the #854 coverage manifest.");
    }
}
