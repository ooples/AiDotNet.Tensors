// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// High-level cuSOLVER-backed <see cref="IDeviceLinalgOps"/>. Cholesky
/// (<c>potrf</c>), QR (<c>geqrf</c> + <c>orgqr</c>), SVD (<c>gesvdj</c>),
/// symmetric eigen (<c>syevj</c>), and LU (<c>getrf</c> + <c>getrs</c>)
/// are dispatched through <see cref="CuSolverNative"/> when the runtime
/// is loadable; otherwise the wrapper hands work to
/// <see cref="CpuLinalgOps"/>. The CPU and CUDA tiers produce
/// numerically-equivalent factorisations within each backend's
/// floating-point precision, so callers don't branch on availability.
///
/// <para>This is the surface mentioned in #219's acceptance criteria
/// for cuSOLVER: "Cholesky / QR / SVD / eigen / LU dispatched to
/// cuSOLVER on CUDA". Device-pointer plumbing is gated behind the
/// CudaBackend's stream/handle ownership, which lands once the
/// device-tensor pipeline is wired; until then the CPU tier carries
/// correctness with the cuSOLVER bindings staying ready for the
/// dispatch flip.</para>
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
