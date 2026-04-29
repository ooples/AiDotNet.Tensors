// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Named;

/// <summary>
/// Name-aware ops over <see cref="NamedTensor{T}"/>. Element-wise ops
/// align dimensions by name (not position) before computing; reduction
/// ops accept axis names instead of axis indices. Mirrors the named
/// surface PyTorch's <c>named_tensor</c> module exposes.
/// </summary>
public static class NamedOps
{
    private static readonly IEngine Engine = new CpuEngine();

    /// <summary>Element-wise add — broadcasts by name. <paramref name="b"/>'s
    /// names must be a subset of <paramref name="a"/>'s; missing names
    /// are inserted as size-1 broadcast axes. Routes through
    /// <c>TensorBroadcastAdd</c> so size-1 broadcast dims expand
    /// without an explicit tile copy.</summary>
    public static NamedTensor<T> Add<T>(NamedTensor<T> a, NamedTensor<T> b)
    {
        var (aligned, namesOut) = AlignForBroadcast(a, b);
        var result = Engine.TensorBroadcastAdd(a.Tensor, aligned);
        return new NamedTensor<T>(result, namesOut);
    }

    /// <summary>Element-wise multiply — same alignment rule as
    /// <see cref="Add{T}"/>.</summary>
    public static NamedTensor<T> Multiply<T>(NamedTensor<T> a, NamedTensor<T> b)
    {
        var (aligned, namesOut) = AlignForBroadcast(a, b);
        // Multiply broadcasts on the same shape rules as add; expand
        // the broadcast axis into a copy first so TensorMultiply
        // (shape-strict) sees matching dims.
        var expanded = ExpandToShape(aligned, a.Tensor._shape);
        var result = Engine.TensorMultiply(a.Tensor, expanded);
        return new NamedTensor<T>(result, namesOut);
    }

    private static Tensor<T> ExpandToShape<T>(Tensor<T> source, int[] targetShape)
    {
        // If source already matches, return as-is.
        if (source.Rank == targetShape.Length)
        {
            bool same = true;
            for (int i = 0; i < targetShape.Length; i++)
                if (source._shape[i] != targetShape[i]) { same = false; break; }
            if (same) return source;
        }
        // Copy source into a target-shape tensor, broadcasting any size-1
        // axes. Cheap fallback for the multiply path; ops that already
        // broadcast natively (Add via TensorBroadcastAdd) skip this.
        var result = new Tensor<T>(targetShape);
        var srcStrides = ComputeStrides(source._shape);
        var dstSpan = result.AsWritableSpan();
        var srcSpan = source.AsSpan();
        var idx = new int[targetShape.Length];
        for (int linear = 0; linear < result.Length; linear++)
        {
            int rem = linear;
            for (int d = targetShape.Length - 1; d >= 0; d--)
            {
                idx[d] = rem % targetShape[d];
                rem /= targetShape[d];
            }
            int srcLinear = 0;
            for (int d = 0; d < source.Rank; d++)
            {
                int dim = source._shape[d];
                int coord = dim == 1 ? 0 : idx[d];
                srcLinear += coord * srcStrides[d];
            }
            dstSpan[linear] = srcSpan[srcLinear];
        }
        return result;
    }

    /// <summary>
    /// Sums along the axis with name <paramref name="dimName"/>.
    /// Equivalent to <c>tensor.sum(dim="batch")</c> in PyTorch's named
    /// surface — the axis is removed (or kept with size-1 if
    /// <paramref name="keepDim"/> is true) and its label dropped.
    /// </summary>
    public static NamedTensor<T> Sum<T>(NamedTensor<T> a, string dimName, bool keepDim = false)
    {
        int idx = Array.IndexOf(a.Names, dimName);
        if (idx < 0) throw new InvalidOperationException(
            $"Sum: dim '{dimName}' not found in [{string.Join(", ", a.Names)}].");
        var summed = Engine.ReduceSum(a.Tensor, new[] { idx }, keepDim);
        var newNames = BuildNamesAfterReduce(a.Names, idx, keepDim);
        return new NamedTensor<T>(summed, newNames);
    }

    /// <summary>Mean along the named axis.</summary>
    public static NamedTensor<T> Mean<T>(NamedTensor<T> a, string dimName, bool keepDim = false)
    {
        int idx = Array.IndexOf(a.Names, dimName);
        if (idx < 0) throw new InvalidOperationException(
            $"Mean: dim '{dimName}' not found in [{string.Join(", ", a.Names)}].");
        var ops = MathHelper.GetNumericOperations<T>();
        var summed = Engine.ReduceSum(a.Tensor, new[] { idx }, keepDim);
        T scale = ops.Divide(ops.One, ops.FromDouble(a.Tensor._shape[idx]));
        var meaned = Engine.TensorMultiplyScalar(summed, scale);
        var newNames = BuildNamesAfterReduce(a.Names, idx, keepDim);
        return new NamedTensor<T>(meaned, newNames);
    }

    /// <summary>
    /// Aligns <paramref name="b"/> to <paramref name="a"/>'s name layout
    /// for broadcasting. The aligned tensor has the same shape as
    /// <paramref name="a"/> with size-1 axes wherever <paramref name="b"/>
    /// lacked the corresponding name. Returns the aligned dense tensor
    /// + the result's names array (same as <c>a.Names</c>).
    /// </summary>
    private static (Tensor<T> aligned, string?[] names) AlignForBroadcast<T>(NamedTensor<T> a, NamedTensor<T> b)
    {
        var alignedShape = new int[a.Rank];
        for (int i = 0; i < a.Rank; i++) alignedShape[i] = 1;

        // For each name in `a`, look it up in `b`.
        var bDimToA = new Dictionary<int, int>();
        for (int aIdx = 0; aIdx < a.Rank; aIdx++)
        {
            string? n = a.Names[aIdx];
            if (n is null) continue;
            int bIdx = Array.IndexOf(b.Names, n);
            if (bIdx >= 0)
            {
                if (b.Tensor._shape[bIdx] != a.Tensor._shape[aIdx] && b.Tensor._shape[bIdx] != 1)
                    throw new InvalidOperationException(
                        $"Broadcast mismatch on axis '{n}': a={a.Tensor._shape[aIdx]} vs b={b.Tensor._shape[bIdx]}.");
                alignedShape[aIdx] = b.Tensor._shape[bIdx];
                bDimToA[bIdx] = aIdx;
            }
        }
        // Reject any names in b not in a.
        for (int bIdx = 0; bIdx < b.Rank; bIdx++)
        {
            if (b.Names[bIdx] is null) continue;
            if (!bDimToA.ContainsKey(bIdx))
                throw new InvalidOperationException(
                    $"Broadcast: name '{b.Names[bIdx]}' from b is not in a's names.");
        }

        // Build the aligned dense tensor by copying b values into the
        // matching positions; fill broadcast axes with the (already 1)
        // shape so the engine's TensorAdd/Multiply broadcasts naturally.
        var aligned = new Tensor<T>(alignedShape);
        // Cheap path when b's names are exactly a subset and ordered the
        // same way: we just reshape b to alignedShape — no per-element move.
        // General path: iterate b's elements, project into alignedShape index.
        var bSpan = b.Tensor.AsSpan();
        var dst = aligned.AsWritableSpan();
        var bShape = b.Tensor._shape;
        var bStrides = ComputeStrides(bShape);
        var alignedStrides = ComputeStrides(alignedShape);

        var bIdxBuf = new int[b.Rank];
        for (int linearB = 0; linearB < b.Tensor.Length; linearB++)
        {
            int rem = linearB;
            for (int d = b.Rank - 1; d >= 0; d--)
            {
                bIdxBuf[d] = rem % bShape[d];
                rem /= bShape[d];
            }
            int linearA = 0;
            for (int d = 0; d < b.Rank; d++)
            {
                if (bDimToA.TryGetValue(d, out int aDim))
                    linearA += bIdxBuf[d] * alignedStrides[aDim];
                // unnamed/missing dims contribute nothing — broadcast-1 in aligned.
            }
            dst[linearA] = bSpan[linearB];
        }
        _ = bStrides;
        return (aligned, (string?[])a.Names.Clone());
    }

    private static int[] ComputeStrides(int[] shape)
    {
        var s = new int[shape.Length];
        int v = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            s[i] = v;
            v *= shape[i];
        }
        return s;
    }

    private static string?[] BuildNamesAfterReduce(string?[] src, int reducedAxis, bool keepDim)
    {
        if (keepDim)
        {
            var copy = (string?[])src.Clone();
            copy[reducedAxis] = null; // reduced axes lose their label.
            return copy;
        }
        var result = new string?[src.Length - 1];
        int w = 0;
        for (int i = 0; i < src.Length; i++)
            if (i != reducedAxis) result[w++] = src[i];
        return result;
    }
}
