#if !NET462
// Copyright (c) AiDotNet. All rights reserved.
// OpenCL IExtendedLinalgBackend (#1142): the portable solver family over the shared la_ kernels.
using P = AiDotNet.Tensors.Engines.DirectGpu.PortableExtendedLinalg;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IExtendedLinalgBackend
{
    void IExtendedLinalgBackend.LinalgQrReducedDirect(IGpuBuffer input, IGpuBuffer q, IGpuBuffer r, int batchCount, int m, int n)
        => P.QrReduced(this, LaunchBatch, input, q, r, batchCount, m, n);

    void IExtendedLinalgBackend.LinalgSvdReduced(IGpuBuffer input, IGpuBuffer u, IGpuBuffer singularValues, IGpuBuffer vh,
        int batchCount, int m, int n)
        => P.SvdReduced(LaunchBatch, input, u, singularValues, vh, batchCount, m, n);

    void IExtendedLinalgBackend.LinalgLuSolve(IGpuBuffer lu, IGpuBuffer pivots, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, int rightHandSides, bool rhsIsVector)
        => P.LuSolve(LaunchBatch, lu, pivots, rhs, solution, batchCount, n, rightHandSides, rhsIsVector);

    void IExtendedLinalgBackend.LinalgEighSymmetric(IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n, bool upper)
        => P.EighSymmetric(this, this, LaunchBatch, input, eigenvalues, eigenvectors, batchCount, n, upper);

    void IExtendedLinalgBackend.LinalgLdlFactor(IGpuBuffer input, IGpuBuffer ld, IGpuBuffer pivots, int batchCount, int n, bool upper)
        => P.LdlFactor(LaunchBatch, input, ld, pivots, batchCount, n, upper);

    void IExtendedLinalgBackend.LinalgLdlSolve(IGpuBuffer ld, IGpuBuffer pivots, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, bool upper, bool rhsIsVector)
        => P.LdlSolve(LaunchBatch, ld, pivots, rhs, solution, batchCount, n, upper, rhsIsVector);

    void IExtendedLinalgBackend.LinalgSolveVector(IGpuBuffer input, IGpuBuffer rhs, IGpuBuffer solution, IGpuBuffer info,
        int batchCount, int n)
        => P.SolveVector(LaunchBatch, input, rhs, solution, info, batchCount, n);

    void IExtendedLinalgBackend.LinalgTriangularSolveVector(IGpuBuffer input, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, bool upper, bool unitDiagonal)
        => P.TriangularSolveVector(LaunchBatch, input, rhs, solution, batchCount, n, upper, unitDiagonal);

    void IExtendedLinalgBackend.LinalgCholeskyBackwardLower(IGpuBuffer factor, IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batchCount, int n)
        => P.CholeskyBackwardLower(LaunchBatch, factor, gradOutput, gradInput, batchCount, n);

    void IExtendedLinalgBackend.LinalgSolveBackwardVector(IGpuBuffer input, IGpuBuffer solution, IGpuBuffer gradOutput,
        IGpuBuffer gradInput, IGpuBuffer gradRhs, int batchCount, int n)
        => P.SolveBackwardVector(LaunchBatch, input, solution, gradOutput, gradInput, gradRhs, batchCount, n);
}
#endif
