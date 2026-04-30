// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Decompositions;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// CPU implementation of <see cref="IDeviceLinalgOps"/>. Delegates to
/// the existing in-tree managed decompositions (<see cref="CholeskyDecomposition"/>,
/// <see cref="QrDecomposition"/>, <see cref="SvdWrapper"/>,
/// <see cref="EighDecomposition"/>, <see cref="LuDecomposition"/>).
/// All return numerically-equivalent factorisations to what
/// cuSOLVER produces, so the CUDA wrapper can pick either tier per
/// shape without changing observable output.
/// </summary>
public sealed class CpuLinalgOps : IDeviceLinalgOps
{
    /// <inheritdoc/>
    public Tensor<T> Cholesky<T>(Tensor<T> a, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // CholeskyDecomposition signals non-SPD batch entries via the
        // `info` tensor (LAPACK convention: zero == success, k == minor
        // of order k that's not positive definite). Discarding `_` here
        // would silently return a partial factor on a non-SPD input,
        // which downstream solvers would then use to produce garbage.
        // Mirror cuSOLVER / NumPy semantics by throwing on the first
        // non-zero info code.
        var (factor, info) = CholeskyDecomposition.Compute(a, upper);
        var infoSpan = info.AsSpan();
        for (int i = 0; i < infoSpan.Length; i++)
        {
            if (infoSpan[i] != 0)
                throw new InvalidOperationException(
                    $"Cholesky failed for batch entry {i}: matrix is not symmetric positive definite (info={infoSpan[i]}).");
        }
        return factor;
    }

    /// <inheritdoc/>
    public (Tensor<T> Q, Tensor<T> R) Qr<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => QrDecomposition.Compute(a, mode: "reduced");

    /// <inheritdoc/>
    public (Tensor<T> U, Tensor<T> S, Tensor<T> Vt) Svd<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => SvdWrapper.Full(a, fullMatrices: false);

    /// <inheritdoc/>
    public (Tensor<T> Eigenvalues, Tensor<T> Eigenvectors) SymmetricEig<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => EighDecomposition.Compute(a, upper: false);

    /// <inheritdoc/>
    public (Tensor<T> Lu, Tensor<int> Pivots) Lu<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LuDecomposition.Factor(a);

    /// <inheritdoc/>
    public Tensor<T> LuSolve<T>(Tensor<T> lu, Tensor<int> pivots, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LuDecomposition.Solve(lu, pivots, b);
}
