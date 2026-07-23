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
        new("CudaBackend.CapsuleTransform", "NVRTC capsule_transform", "output[b,i,j,d] = sum_k input[b,i,k] * weights[i,k,j,d]", "row-major input [B,I,K]; weights [I,K,J,D]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-output-element serial-contraction PTX kernel (PtxCapsuleContractionKernel, Transform variant); fails closed until GPU-validated and promoted"),
        new("CudaBackend.CapsuleWeightedSum", "NVRTC capsule_weighted_sum", "output[b,c,d] = sum_i coupling[b,i,c] * predictions[b,i,c,d]", "row-major coupling [B,I,C]; predictions [B,I,C,D]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-output-element serial-i-reduction PTX kernel (PtxCapsuleWeightedSumKernel); fails closed until GPU-validated and promoted"),
        new("CudaBackend.CapsuleAgreement", "NVRTC capsule_agreement", "agreement[b,i,c] = sum_d predictions[b,i,c,d] * output[b,c,d]", "row-major predictions [B,I,C,D]; output [B,C,D]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-output-element serial-d-reduction PTX kernel (PtxCapsuleAgreementKernel); fails closed until GPU-validated and promoted"),
        new("CudaBackend.CosineSimilarity", "NVRTC cosine_similarity", "output[b] = dot(a[b],b[b]) / (||a[b]||*||b[b]|| + 1e-8)", "row-major [B,dim] pairs", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-batch serial-dim PTX kernel (PtxCosineSimilarityKernel) accumulating dot + both squared norms; fails closed until GPU-validated and promoted"),
        new("CudaBackend.SphericalSoftmax", "NVRTC spherical_softmax", "softmax(L2-normalize(x)) per row", "row-major [outer,inner]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-row 4-pass PTX kernel (PtxSphericalSoftmaxKernel): L2 norm, normalize+max, stable exp+sum, scale; expf via ex2.approx; fails closed until GPU-validated and promoted"),
        new("CudaBackend.NormalizeProbabilities", "NVRTC normalize_probabilities", "probabilities[b,i] /= max(row-sum, 1e-10)", "row-major [batch,state]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "one-block-per-row 256-lane shared-memory tree-reduction PTX kernel (PtxNormalizeProbabilitiesKernel), in-place; fails closed until GPU-validated and promoted"),
        new("CudaBackend.MeasurementForward", "NVRTC measurement_forward", "output[b,i] = |z_i|^2 / max(sum_i |z_i|^2, 1e-10)", "interleaved complex input [batch,state,2]; output [batch,state]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "one-block-per-row 256-lane shared-memory tree-reduction PTX kernel (PtxMeasurementForwardKernel) fusing |z|^2 evaluation and normalization; fails closed until GPU-validated and promoted"),
        new("CudaBackend.QuantumRotation", "NVRTC quantum_rotation", "apply Ry(angles[q]) to each qubit q in turn on the split-complex state", "split real/imag [batch,2^numQubits]; angles [numQubits]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "one-block-per-batch PTX kernel (PtxQuantumRotationKernel) with the numQubits loop unrolled, cos/sin via cos.approx/sin.approx and a bar.sync between disjoint-pair butterfly steps; fails closed until GPU-validated and promoted"),
        new("CudaBackend.AnnComputeDistances", "NVRTC ann_compute_distances", "distances[q,j] = metric(query_q, db_j); L2-squared or inner-product", "row-major queries [Q,dim]; database [DB,dim]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-cell serial-dim PTX kernel (PtxAnnComputeDistancesKernel) with baked metric; fails closed until GPU-validated and promoted"),
        new("CudaBackend.AnnPqDistanceTables", "NVRTC ann_pq_distance_tables", "tables[q,s,c] = metric(query subvec, codebook subcentroid)", "row-major queries [Q,m*dsub]; codebooks [m,ksub,dsub]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-cell serial-subdim PTX kernel (PtxAnnPqDistanceTablesKernel) with baked metric; fails closed until GPU-validated and promoted"),
        new("CudaBackend.AnnIvfAssign", "NVRTC ann_ivf_assign", "assignments[i] = argmin/argmax_c metric(vector_i, centroid_c), ties to lowest c", "row-major vectors [V,dim]; centroids [C,dim]; int32 assignments", "FP32/INT32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-vector centroid-scan PTX kernel (PtxAnnIvfAssignKernel) with baked metric, strict-improvement argmin/argmax and int32 output; fails closed until GPU-validated and promoted"),
        new("CudaBackend.AnnPqAdcScan", "NVRTC ann_pq_adc_scan", "distances[q,i] = sum_s tables[q, s, codes[i,s]]", "uint8 codes [codes,m]; tables [Q,m,ksub]; distances [Q,codes]", "FP32/UINT8", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-cell PTX kernel (PtxAnnPqAdcScanKernel) gathering per-subspace table lookups via uint8 codes (ld.global.nc.u8); fails closed until GPU-validated and promoted"),
        new("CudaBackend.HashGridEncodeLevel", "NVRTC instant_ngp_hash_encode_level", "trilinear interpolation over 8 hashed voxel corners of clamp01(pos)*resolution", "positions [P,3]; hashTable [tableSize,featuresPerLevel]; output [P,outputStride]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-(point,feature) PTX kernel (PtxInstantNgpHashEncodeKernel): floor via cvt.rmi.s32.f32, reference spatial hash (mul/xor/mod), register-resident trilinear blend; fails closed until GPU-validated and promoted"),
        new("CudaBackend.HashGridEncodeLevelBackward", "NVRTC instant_ngp_hash_encode_level_backward", "tableGradient[entry,f] = sum_n sum_corner [hash(corner_n)==entry] * grad_n * w_corner", "positions [P,3]; outputGradient [P,outputStride]; tableGradient [tableSize,featuresPerLevel]", "FP32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-(entry,feature) point-scan PTX kernel (PtxInstantNgpHashEncodeBackwardKernel): recomputes the 8 hashes/weights per point and accumulates under a hash-match predicate; fails closed until GPU-validated and promoted"),
        new("CudaBackend.UniformMeshLaplacian", "NVRTC resident_uniform_mesh_laplacian", "dense combinatorial mesh Laplacian from a triangle face list", "int32 faces [F,3]; output [V,V]", "FP32/INT32", DirectPtxScientificCoverageStatus.ExperimentalDirectPtx, "thread-per-(row,col) face-scan PTX kernel (PtxMeshLaplacianKernel) accumulating +/-1 edge contributions under equality predicates; fails closed until GPU-validated and promoted"),
        new("CudaBackend.GenerateSpiralIndices", "NVRTC generate_spiral_indices", "per-vertex spiral neighbor ordering via ring-expansion BFS", "vertices/faces; scratch rings; output [V,spiralLength]", "FP32", DirectPtxScientificCoverageStatus.BaselineOnly, "single-threaded (block0/thread0) control-flow-heavy graph BFS with insertion-sort and mutable scratch buffers; direct-PTX offers no benefit for a 1-thread kernel and it cannot meet the parallel championship gate, so it is intentionally retained on the NVRTC baseline")
    ];

    internal static DirectPtxScientificCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        foreach (DirectPtxScientificCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException($"Scientific API '{api}' is not assigned in the #854 coverage manifest.");
    }
}
