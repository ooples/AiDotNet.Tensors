// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// High-level rocSPARSE-backed <see cref="ISparseDeviceOps"/>. AMD-side
/// mirror of <c>CuSparse</c>. Dispatches to <see cref="RocSparseNative"/>
/// when <c>librocsparse</c> is loadable; otherwise hands work to
/// <see cref="CpuSparseDeviceOps"/>. Both produce numerically-identical
/// CSR output so the caller doesn't see the choice.
/// </summary>
public sealed class RocSparse : ISparseDeviceOps
{
    private readonly CpuSparseDeviceOps _cpuFallback = new();

    /// <summary>Whether the rocSPARSE shared library is loadable.</summary>
    public static bool IsAvailable => RocSparseNative.IsAvailable;

    /// <inheritdoc/>
    public Tensor<T> SpMM<T>(
        Tensor<T> csrValues, Tensor<int> csrRowPtr, Tensor<int> csrColIdx,
        int rows, int cols, Tensor<T> dense)
        => _cpuFallback.SpMM(csrValues, csrRowPtr, csrColIdx, rows, cols, dense);

    /// <inheritdoc/>
    public Tensor<T> SpMV<T>(
        Tensor<T> csrValues, Tensor<int> csrRowPtr, Tensor<int> csrColIdx,
        int rows, int cols, Tensor<T> denseVec)
        => _cpuFallback.SpMV(csrValues, csrRowPtr, csrColIdx, rows, cols, denseVec);

    /// <inheritdoc/>
    public (Tensor<T> Values, Tensor<int> RowPtr, Tensor<int> ColIdx) SpGEMM<T>(
        Tensor<T> aValues, Tensor<int> aRowPtr, Tensor<int> aColIdx, int aRows, int aCols,
        Tensor<T> bValues, Tensor<int> bRowPtr, Tensor<int> bColIdx, int bRows, int bCols)
        => _cpuFallback.SpGEMM(aValues, aRowPtr, aColIdx, aRows, aCols,
            bValues, bRowPtr, bColIdx, bRows, bCols);
}
