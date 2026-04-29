// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>
/// Device-wide reduce / scan / sort / histogram primitives — the
/// Thrust/CUB equivalents we surface as a building block for custom
/// kernels and high-level tensor ops alike. PyTorch hides these behind
/// tensor methods; we expose them so users (and other internal subsystems
/// like the data-loading sampler and the distributed reducer) can call
/// the same warp-friendly implementations directly.
///
/// <para><b>Backends:</b> CPU (managed SIMD), CUDA (Thrust/CUB),
/// HIP (rocThrust), Metal (MPS reduce/sort), OpenCL/Vulkan/WebGPU
/// (managed kernels). Every backend implements the same surface so
/// callers don't branch on backend.</para>
/// </summary>
public interface IDevicePrimitives
{
    /// <summary>Sums <paramref name="input"/> along the given <paramref name="axis"/>
    /// (or all elements if <paramref name="axis"/> is <c>-1</c>).</summary>
    Tensor<T> Reduce<T>(Tensor<T> input, int axis = -1, ReductionKind kind = ReductionKind.Sum);

    /// <summary>Cumulative reduction (prefix sum / prefix max / etc.) along
    /// <paramref name="axis"/>. <paramref name="exclusive"/> = false matches
    /// <c>torch.cumsum</c>; true gives the exclusive-scan variant Thrust
    /// uses internally.</summary>
    Tensor<T> Scan<T>(Tensor<T> input, int axis = -1, ReductionKind kind = ReductionKind.Sum, bool exclusive = false);

    /// <summary>Sorts <paramref name="input"/> along <paramref name="axis"/>
    /// and returns the sorted tensor. Use <see cref="ArgSort{T}"/> when
    /// you need the index permutation that maps original positions to
    /// sorted positions — this overload returns sorted values only.</summary>
    Tensor<T> Sort<T>(Tensor<T> input, int axis = -1, bool descending = false);

    /// <summary>Returns the index permutation that would sort the tensor.
    /// Equivalent to <c>torch.argsort</c>.</summary>
    Tensor<int> ArgSort<T>(Tensor<T> input, int axis = -1, bool descending = false);

    /// <summary>Histogram with <paramref name="bins"/> equal-width bins
    /// over <c>[lo, hi)</c>. Returns int32 counts.</summary>
    Tensor<int> Histogram<T>(Tensor<T> input, int bins, T lo, T hi);

    /// <summary>Run-length encode: returns parallel arrays
    /// <c>(values, counts)</c> over consecutive equal runs.</summary>
    (Tensor<T> Values, Tensor<int> Counts) RunLengthEncode<T>(Tensor<T> input);
}

/// <summary>Reduction operator. Used by both
/// <see cref="IDevicePrimitives.Reduce{T}"/> and
/// <see cref="IDevicePrimitives.Scan{T}"/>.</summary>
public enum ReductionKind
{
    /// <summary>Sum.</summary>
    Sum,
    /// <summary>Element-wise minimum.</summary>
    Min,
    /// <summary>Element-wise maximum.</summary>
    Max,
    /// <summary>Element-wise product.</summary>
    Product,
}
