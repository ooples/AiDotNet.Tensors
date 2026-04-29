// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// MPS-backed <see cref="IDeviceLinalgOps"/>. Apple-platform mirror of
/// <c>CuSolver</c> / <c>RocSolver</c>. Cholesky / QR / SVD / symmetric
/// eigen / LU + LU-solve currently delegate to <see cref="CpuLinalgOps"/>;
/// the <see cref="MpsLinalgNative"/> probe and binding classes are in
/// place, with the device-pointer plumbing flipping in once the
/// device-tensor pipeline lands.
/// </summary>
public sealed class MpsLinalg : IDeviceLinalgOps
{
    private readonly CpuLinalgOps _cpuFallback = new();

    /// <summary>Whether the MPS dense-decomposition kernels are loadable.</summary>
    public static bool IsAvailable => MpsLinalgNative.IsAvailable;

    /// <inheritdoc/>
    public Tensor<T> Cholesky<T>(Tensor<T> a, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => _cpuFallback.Cholesky(a, upper);

    /// <inheritdoc/>
    public (Tensor<T> Q, Tensor<T> R) Qr<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => _cpuFallback.Qr(a);

    /// <inheritdoc/>
    public (Tensor<T> U, Tensor<T> S, Tensor<T> Vt) Svd<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => _cpuFallback.Svd(a);

    /// <inheritdoc/>
    public (Tensor<T> Eigenvalues, Tensor<T> Eigenvectors) SymmetricEig<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => _cpuFallback.SymmetricEig(a);

    /// <inheritdoc/>
    public (Tensor<T> Lu, Tensor<int> Pivots) Lu<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => _cpuFallback.Lu(a);

    /// <inheritdoc/>
    public Tensor<T> LuSolve<T>(Tensor<T> lu, Tensor<int> pivots, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => _cpuFallback.LuSolve(lu, pivots, b);
}
