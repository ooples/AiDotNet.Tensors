// Copyright (c) AiDotNet. All rights reserved.
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IExtendedLinalgBackend"/> for every backend without CUDA's direct-PTX solver family (#1142). The
/// symmetric eigendecomposition runs the backend's own <see cref="ILinalgBackend.LinalgEigh"/> after symmetrizing from
/// the requested triangle, so both triangles are honoured; everything else runs the shared <c>la_</c> kernels, ports
/// of the managed references the engine otherwise falls back to. Kernels that factor in registers take matrices up to
/// <see cref="K.LaMaxN"/> on a side; QR, LDL and the substitutions work in device memory and take any size.
/// </summary>
internal static class PortableExtendedLinalg
{
    /// <summary>
    /// Householder QR as <c>QrDecomposition</c> and LAPACK <c>geqrf</c> compute it, so R's diagonal signs match the
    /// CPU and PyTorch (Gram-Schmidt would force them positive). One thread per matrix over a device workspace.
    /// </summary>
    public static void QrReduced(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer input,
        IGpuBuffer q, IGpuBuffer r, int batchCount, int m, int n)
    {
        if (m < 1) throw new ArgumentOutOfRangeException(nameof(m));
        if (!Validate(batchCount, n)) return;
        int k = Math.Min(m, n);
        using var work = backend.AllocateBuffer(checked(batchCount * (m * n + 2 * k)));
        launch(K.LaQrReduced, batchCount, input, q, r, work, batchCount, m, n);
    }

    public static void EighSymmetric(IDirectGpuBackend backend, ILinalgBackend linalg, PortableGenericFft.Launcher launch,
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors, int batchCount, int n, bool upper)
    {
        if (!Validate(batchCount, n)) return;
        int count = checked(batchCount * n * n);
        using var symmetric = backend.AllocateBuffer(count);
        launch(K.LaSymmetrize, count, input, symmetric, batchCount, n, upper ? 1 : 0);
        linalg.LinalgEigh(symmetric, eigenvalues, eigenvectors, batchCount, n);
    }

    public static void SvdReduced(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer u,
        IGpuBuffer singularValues, IGpuBuffer vh, int batchCount, int m, int n)
    {
        if (m < 1 || m > K.LaMaxN) throw new ArgumentOutOfRangeException(nameof(m), m, $"SVD takes 1..{K.LaMaxN} rows.");
        if (!Validate(batchCount, n, registers: true)) return;
        launch(K.LaSvdReduced, batchCount, input, u, singularValues, vh, batchCount, m, n);
    }

    public static void LuSolve(PortableGenericFft.Launcher launch, IGpuBuffer lu, IGpuBuffer pivots, IGpuBuffer rhs,
        IGpuBuffer solution, int batchCount, int n, int rightHandSides, bool rhsIsVector)
    {
        int columns = rhsIsVector ? 1 : rightHandSides;
        if (columns < 1) throw new ArgumentOutOfRangeException(nameof(rightHandSides));
        if (!Validate(batchCount, n)) return;
        launch(K.LaLuSolve, batchCount, lu, pivots, rhs, solution, batchCount, n, columns);
    }

    public static void LdlFactor(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer ld, IGpuBuffer pivots,
        int batchCount, int n, bool upper)
    {
        if (!Validate(batchCount, n)) return;
        launch(K.LaLdlFactor, batchCount, input, ld, pivots, batchCount, n, upper ? 1 : 0);
    }

    public static void LdlSolve(PortableGenericFft.Launcher launch, IGpuBuffer ld, IGpuBuffer pivots, IGpuBuffer rhs,
        IGpuBuffer solution, int batchCount, int n, bool upper, bool rhsIsVector)
    {
        // The contract carries no column count, so a matrix right-hand side is ambiguous here (as on CUDA).
        if (!rhsIsVector)
            throw new NotSupportedException("LDL solve takes one vector right-hand side per factor.");
        if (!Validate(batchCount, n)) return;
        launch(K.LaLdlSolve, batchCount, ld, pivots, rhs, solution, batchCount, n, 1, upper ? 1 : 0);
    }

    public static void SolveVector(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer rhs,
        IGpuBuffer solution, IGpuBuffer info, int batchCount, int n)
    {
        if (!Validate(batchCount, n, registers: true)) return;
        launch(K.LaSolveVector, batchCount, input, rhs, solution, info, batchCount, n);
    }

    public static void TriangularSolveVector(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer rhs,
        IGpuBuffer solution, int batchCount, int n, bool upper, bool unitDiagonal)
    {
        if (!Validate(batchCount, n)) return;
        launch(K.LaTriangularSolveVector, batchCount, input, rhs, solution, batchCount, n, upper ? 1 : 0,
            unitDiagonal ? 1 : 0);
    }

    public static void CholeskyBackwardLower(PortableGenericFft.Launcher launch, IGpuBuffer factor,
        IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchCount, int n)
    {
        if (!Validate(batchCount, n, registers: true)) return;
        launch(K.LaCholeskyBackwardLower, batchCount, factor, gradOutput, gradInput, batchCount, n);
    }

    public static void SolveBackwardVector(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer solution,
        IGpuBuffer gradOutput, IGpuBuffer gradInput, IGpuBuffer gradRhs, int batchCount, int n)
    {
        if (!Validate(batchCount, n, registers: true)) return;
        launch(K.LaSolveBackwardVector, batchCount, input, solution, gradOutput, gradInput, gradRhs, batchCount, n);
    }

    /// <summary>False for an empty batch (nothing to launch); throws on a size the kernel cannot hold.</summary>
    private static bool Validate(int batchCount, int n, bool registers = false)
    {
        if (batchCount < 0) throw new ArgumentOutOfRangeException(nameof(batchCount));
        int limit = registers ? K.LaMaxN : int.MaxValue;
        if (n < 1 || n > limit) throw new ArgumentOutOfRangeException(nameof(n), n, $"Matrix size must be 1..{limit}.");
        _ = checked(batchCount * n * n);
        return batchCount > 0;
    }
}
