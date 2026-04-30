// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>
/// Linear-algebra factorisations exposed at the device level. CUDA
/// backend wraps cuSOLVER; HIP wraps rocSOLVER; Metal wraps MPS matrix
/// decompositions; CPU implements via the existing
/// <c>MatrixDecomposition</c> helpers.
///
/// <para>This is a parity surface — the linalg issue (#222 / linalg
/// pyobject)'s GPU bindings live here. The factorisation algorithms
/// themselves are conventional; the value is in surfacing the same
/// signatures across backends so user code is portable.</para>
/// </summary>
public interface IDeviceLinalgOps
{
    /// <summary>Cholesky decomposition: <c>A = L · L^T</c>.
    /// <paramref name="upper"/> selects upper- vs lower-triangular L.</summary>
    Tensor<T> Cholesky<T>(Tensor<T> a, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>;

    /// <summary>QR decomposition.</summary>
    (Tensor<T> Q, Tensor<T> R) Qr<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>;

    /// <summary>Reduced SVD: <c>A = U · Σ · V^T</c>.</summary>
    (Tensor<T> U, Tensor<T> S, Tensor<T> Vt) Svd<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>;

    /// <summary>Symmetric eigendecomposition: <c>A · v = λ · v</c>.
    /// Returns eigenvalues + eigenvectors (columns of the second tensor).</summary>
    (Tensor<T> Eigenvalues, Tensor<T> Eigenvectors) SymmetricEig<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>;

    /// <summary>LU decomposition + permutation. <c>P · A = L · U</c>.</summary>
    (Tensor<T> Lu, Tensor<int> Pivots) Lu<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>;

    /// <summary>Solves <c>A · x = b</c> using a previously-computed LU.</summary>
    Tensor<T> LuSolve<T>(Tensor<T> lu, Tensor<int> pivots, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>;
}
