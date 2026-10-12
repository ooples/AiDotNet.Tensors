// Copyright (c) AiDotNet. All rights reserved.
// Vulkan IExtendedLinalgBackend (#1142): the portable solver family over the shared la_ kernels.
using System.Collections.Generic;
using P = AiDotNet.Tensors.Engines.DirectGpu.PortableExtendedLinalg;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IExtendedLinalgBackend
{
    /// <summary>Runs a shared la_ solver: buffers bind in argument order, every int is one push constant.</summary>
    private void LaunchSolver(string name, int threads, params object[] args)
    {
        if (!VulkanBatchExecutionKernels.Solvers.TryGetValue(name, out var source))
            throw new InvalidOperationException($"Vulkan solver kernel not found: {name}");
        var buffers = new List<IGpuBuffer>();
        var push = new List<uint>();
        foreach (var arg in args)
        {
            switch (arg)
            {
                case IGpuBuffer buffer: buffers.Add(buffer); break;
                case int value: push.Add(Bits(value)); break;
                default:
                    throw new ArgumentException($"Unsupported argument type {arg?.GetType().Name ?? "null"} for {name}.");
            }
        }
        GlslNaryOp(source, buffers.ToArray(), threads, push.ToArray());
    }

    void IExtendedLinalgBackend.LinalgQrReducedDirect(IGpuBuffer input, IGpuBuffer q, IGpuBuffer r, int batchCount, int m, int n)
        => P.QrReduced(this, input, q, r, batchCount, m, n);

    void IExtendedLinalgBackend.LinalgSvdReduced(IGpuBuffer input, IGpuBuffer u, IGpuBuffer singularValues, IGpuBuffer vh,
        int batchCount, int m, int n)
        => P.SvdReduced(LaunchSolver, input, u, singularValues, vh, batchCount, m, n);

    void IExtendedLinalgBackend.LinalgLuSolve(IGpuBuffer lu, IGpuBuffer pivots, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, int rightHandSides, bool rhsIsVector)
        => P.LuSolve(LaunchSolver, lu, pivots, rhs, solution, batchCount, n, rightHandSides, rhsIsVector);

    void IExtendedLinalgBackend.LinalgEighSymmetric(IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n, bool upper)
        => P.EighSymmetric(this, this, LaunchSolver, input, eigenvalues, eigenvectors, batchCount, n, upper);

    void IExtendedLinalgBackend.LinalgLdlFactor(IGpuBuffer input, IGpuBuffer ld, IGpuBuffer pivots, int batchCount, int n, bool upper)
        => P.LdlFactor(LaunchSolver, input, ld, pivots, batchCount, n, upper);

    void IExtendedLinalgBackend.LinalgLdlSolve(IGpuBuffer ld, IGpuBuffer pivots, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, bool upper, bool rhsIsVector)
        => P.LdlSolve(LaunchSolver, ld, pivots, rhs, solution, batchCount, n, upper, rhsIsVector);

    void IExtendedLinalgBackend.LinalgSolveVector(IGpuBuffer input, IGpuBuffer rhs, IGpuBuffer solution, IGpuBuffer info,
        int batchCount, int n)
        => P.SolveVector(LaunchSolver, input, rhs, solution, info, batchCount, n);

    void IExtendedLinalgBackend.LinalgTriangularSolveVector(IGpuBuffer input, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, bool upper, bool unitDiagonal)
        => P.TriangularSolveVector(LaunchSolver, input, rhs, solution, batchCount, n, upper, unitDiagonal);

    void IExtendedLinalgBackend.LinalgCholeskyBackwardLower(IGpuBuffer factor, IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batchCount, int n)
        => P.CholeskyBackwardLower(LaunchSolver, factor, gradOutput, gradInput, batchCount, n);

    void IExtendedLinalgBackend.LinalgSolveBackwardVector(IGpuBuffer input, IGpuBuffer solution, IGpuBuffer gradOutput,
        IGpuBuffer gradInput, IGpuBuffer gradRhs, int batchCount, int n)
        => P.SolveBackwardVector(LaunchSolver, input, solution, gradOutput, gradInput, gradRhs, batchCount, n);
}
