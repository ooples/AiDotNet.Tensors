// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// High-level <see cref="IDeviceLinalgOps"/> wired in advance for
/// cuSOLVER. Cholesky (<c>potrf</c>), QR (<c>geqrf</c> + <c>orgqr</c>),
/// SVD (<c>gesvdj</c>), symmetric eigen (<c>syevj</c>), and LU
/// (<c>getrf</c> + <c>getrs</c>) currently delegate to
/// <see cref="CpuLinalgOps"/> in every method on this class — the
/// <see cref="CuSolverNative"/> P/Invoke layer is in place but the
/// device-pointer plumbing (CudaBackend stream/handle ownership +
/// device-tensor marshalling) is a #219 follow-up. Until that lands
/// the CPU tier carries correctness; the cuSOLVER bindings stay ready
/// for the dispatch flip without changing the public API.
/// <see cref="IsAvailable"/> reports whether the shared library can be
/// loaded so callers can probe future hardware-accelerated paths.
/// </summary>
public sealed class CuSolver : IDeviceLinalgOps
{
    private readonly CpuLinalgOps _cpuFallback = new();

    /// <summary>Whether the cuSOLVER shared library is loadable.</summary>
    public static bool IsAvailable => CuSolverNative.IsAvailable;

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
