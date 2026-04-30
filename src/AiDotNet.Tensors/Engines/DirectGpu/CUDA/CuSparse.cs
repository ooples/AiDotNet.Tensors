// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// High-level cuSPARSE-backed <see cref="ISparseDeviceOps"/>. SpMM /
/// SpMV / SpGEMM dispatch through <see cref="CuSparseNative"/> when the
/// runtime is loadable; otherwise the wrapper hands work to
/// <see cref="CpuSparseDeviceOps"/>. Both produce numerically-equal CSR
/// output, so any caller (the autotune-driven sparse path in
/// <c>TensorMatMul</c>, the embedding gather/scatter helpers, etc.) is
/// transparent to the choice.
///
/// <para>Like <see cref="CuRand"/>, the CPU path is the default until
/// the device-tensor pipeline lands; the cuSPARSE bindings stay wired
/// for the future on-device dispatch flip. This means the API surface
/// is committed today — callers don't rewrite once the dispatch
/// switches. cuSPARSE's device-pointer plumbing also wants
/// per-stream handles which the broader CudaBackend owns; that
/// integration sits behind the <see cref="IsAvailable"/> probe.</para>
/// </summary>
public sealed class CuSparse : ISparseDeviceOps
{
    private readonly CpuSparseDeviceOps _cpuFallback = new();

    /// <summary>Whether the cuSPARSE shared library is loadable.</summary>
    public static bool IsAvailable => CuSparseNative.IsAvailable;

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
