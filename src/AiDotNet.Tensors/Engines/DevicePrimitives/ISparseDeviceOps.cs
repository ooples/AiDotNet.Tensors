// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>
/// Sparse-matrix kernel surface. CUDA backend wraps cuSPARSE; HIP wraps
/// rocSPARSE; CPU implements CSR-format SpMM/SpMV directly.
///
/// <para>The autotune-driven dispatch from #219's "How we beat PyTorch"
/// point #2 lives in the calling op (see <c>TensorMatMul</c>'s sparse
/// fast-path) — the kernel surface here just exposes the bare
/// SpMM/SpMV/SpGEMM primitives.</para>
/// </summary>
public interface ISparseDeviceOps
{
    /// <summary>Sparse × dense matmul. <paramref name="csrValues"/> +
    /// <paramref name="csrRowPtr"/> + <paramref name="csrColIdx"/> are
    /// the CSR triple for the sparse left operand.</summary>
    Tensor<T> SpMM<T>(
        Tensor<T> csrValues, Tensor<int> csrRowPtr, Tensor<int> csrColIdx,
        int rows, int cols, Tensor<T> dense);

    /// <summary>Sparse × dense matvec. Same CSR triple as
    /// <see cref="SpMM{T}"/>.</summary>
    Tensor<T> SpMV<T>(
        Tensor<T> csrValues, Tensor<int> csrRowPtr, Tensor<int> csrColIdx,
        int rows, int cols, Tensor<T> denseVec);

    /// <summary>Sparse × sparse matmul (cuSPARSE SpGEMM). Returns the
    /// product as a CSR triple.</summary>
    (Tensor<T> Values, Tensor<int> RowPtr, Tensor<int> ColIdx) SpGEMM<T>(
        Tensor<T> aValues, Tensor<int> aRowPtr, Tensor<int> aColIdx, int aRows, int aCols,
        Tensor<T> bValues, Tensor<int> bRowPtr, Tensor<int> bColIdx, int bRows, int bCols);
}
