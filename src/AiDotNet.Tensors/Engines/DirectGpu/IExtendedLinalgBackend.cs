// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional extension for dense solver operations not present in the original
/// parity-211 backend contract. CUDA implements this for the exact direct-PTX
/// solver-family specializations; other backends continue to fall back.
/// </summary>
public interface IExtendedLinalgBackend
{
    void LinalgQrReducedDirect(
        IGpuBuffer input,
        IGpuBuffer q,
        IGpuBuffer r,
        int batchCount,
        int m,
        int n);

    void LinalgSvdReduced(
        IGpuBuffer input,
        IGpuBuffer u,
        IGpuBuffer singularValues,
        IGpuBuffer vh,
        int batchCount,
        int m,
        int n);

    void LinalgLuSolve(
        IGpuBuffer lu,
        IGpuBuffer pivots,
        IGpuBuffer rhs,
        IGpuBuffer solution,
        int batchCount,
        int n,
        int rightHandSides,
        bool rhsIsVector);

    void LinalgEighSymmetric(
        IGpuBuffer input,
        IGpuBuffer eigenvalues,
        IGpuBuffer eigenvectors,
        int batchCount,
        int n,
        bool upper);

    void LinalgLdlFactor(
        IGpuBuffer input,
        IGpuBuffer ld,
        IGpuBuffer pivots,
        int batchCount,
        int n,
        bool upper);

    void LinalgLdlSolve(
        IGpuBuffer ld,
        IGpuBuffer pivots,
        IGpuBuffer rhs,
        IGpuBuffer solution,
        int batchCount,
        int n,
        bool upper,
        bool rhsIsVector);

    void LinalgSolveVector(
        IGpuBuffer input,
        IGpuBuffer rhs,
        IGpuBuffer solution,
        IGpuBuffer info,
        int batchCount,
        int n);

    void LinalgTriangularSolveVector(
        IGpuBuffer input,
        IGpuBuffer rhs,
        IGpuBuffer solution,
        int batchCount,
        int n,
        bool upper,
        bool unitDiagonal);

    void LinalgCholeskyBackwardLower(
        IGpuBuffer factor,
        IGpuBuffer gradOutput,
        IGpuBuffer gradInput,
        int batchCount,
        int n);

    void LinalgSolveBackwardVector(
        IGpuBuffer input,
        IGpuBuffer solution,
        IGpuBuffer gradOutput,
        IGpuBuffer gradInput,
        IGpuBuffer gradRhs,
        int batchCount,
        int n);
}
