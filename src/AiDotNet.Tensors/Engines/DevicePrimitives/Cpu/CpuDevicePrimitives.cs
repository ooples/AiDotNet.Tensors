// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// CPU reference implementation of <see cref="IDevicePrimitives"/>. Uses
/// the existing <see cref="MathHelper"/> numeric ops + a small set of
/// straightforward sequential algorithms. The GPU backends (Thrust/CUB,
/// rocThrust) are expected to massively outperform this on large
/// inputs; the CPU path is here so backend-agnostic code (the data
/// loader, the distributed reducer, custom user kernels) doesn't need
/// to special-case "no GPU available".
/// </summary>
public sealed class CpuDevicePrimitives : IDevicePrimitives
{
    /// <inheritdoc/>
    public Tensor<T> Reduce<T>(Tensor<T> input, int axis = -1, ReductionKind kind = ReductionKind.Sum)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        if (axis == -1)
        {
            // Reduce all elements to a scalar tensor.
            var src = input.AsSpan();
            // Empty input has no elements to seed the accumulator. NumPy/
            // PyTorch return the reduction's identity element here (0 for
            // Sum, 1 for Prod, +Inf/-Inf for Min/Max), but we don't have
            // a reliable cross-numeric-type identity for Min/Max without
            // pulling in INumericOperations.MaxValue/MinValue — so the
            // safest contract is to fail loudly with a diagnostic rather
            // than silently produce an unsound zero. Sum and Prod still
            // get a defined identity (Zero / One via the ops surface).
            if (src.Length == 0)
            {
                if (kind == ReductionKind.Sum)
                {
                    var s = new Tensor<T>(new[] { 1 });
                    s.AsWritableSpan()[0] = ops.Zero;
                    return s;
                }
                if (kind == ReductionKind.Product)
                {
                    var s = new Tensor<T>(new[] { 1 });
                    s.AsWritableSpan()[0] = ops.One;
                    return s;
                }
                throw new ArgumentException(
                    $"Cannot reduce an empty tensor with kind={kind} — Min/Max have no defined identity.",
                    nameof(input));
            }
            T acc = src[0];
            for (int i = 1; i < src.Length; i++) acc = Combine(acc, src[i], ops, kind);
            var scalar = new Tensor<T>(new[] { 1 });
            scalar.AsWritableSpan()[0] = acc;
            return scalar;
        }
        // Reduce along a single axis.
        if (axis < 0 || axis >= input.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        int outer = 1, axisLen = input._shape[axis], inner = 1;
        for (int d = 0; d < axis; d++) outer *= input._shape[d];
        for (int d = axis + 1; d < input.Rank; d++) inner *= input._shape[d];

        var outShape = new int[input.Rank - 1];
        for (int d = 0, j = 0; d < input.Rank; d++) if (d != axis) outShape[j++] = input._shape[d];
        var output = new Tensor<T>(outShape);
        var inSpan = input.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                T acc = inSpan[(o * axisLen) * inner + i];
                for (int a = 1; a < axisLen; a++)
                    acc = Combine(acc, inSpan[(o * axisLen + a) * inner + i], ops, kind);
                outSpan[o * inner + i] = acc;
            }
        }
        return output;
    }

    /// <inheritdoc/>
    public Tensor<T> Scan<T>(Tensor<T> input, int axis = -1, ReductionKind kind = ReductionKind.Sum, bool exclusive = false)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        int actualAxis = axis < 0 ? input.Rank - 1 : axis;
        if (actualAxis < 0 || actualAxis >= input.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        int outer = 1, axisLen = input._shape[actualAxis], inner = 1;
        for (int d = 0; d < actualAxis; d++) outer *= input._shape[d];
        for (int d = actualAxis + 1; d < input.Rank; d++) inner *= input._shape[d];

        var output = new Tensor<T>((int[])input._shape.Clone());
        var inSpan = input.AsSpan();
        var outSpan = output.AsWritableSpan();
        T identity = kind switch
        {
            ReductionKind.Sum => ops.Zero,
            ReductionKind.Product => ops.One,
            ReductionKind.Min => ops.MaxValue,
            ReductionKind.Max => ops.MinValue,
            _ => ops.Zero,
        };

        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                T acc = identity;
                for (int a = 0; a < axisLen; a++)
                {
                    int idx = (o * axisLen + a) * inner + i;
                    if (exclusive)
                    {
                        outSpan[idx] = acc;
                        acc = Combine(acc, inSpan[idx], ops, kind);
                    }
                    else
                    {
                        acc = Combine(acc, inSpan[idx], ops, kind);
                        outSpan[idx] = acc;
                    }
                }
            }
        }
        return output;
    }

    /// <inheritdoc/>
    public Tensor<T> Sort<T>(Tensor<T> input, int axis = -1, bool descending = false)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        // Resolve and validate the axis BEFORE any shape indexing — without
        // this, a bad axis hit `_shape[axis]` deep inside SortAxis and
        // surfaced as IndexOutOfRangeException with no parameter context.
        int actualAxis = axis < 0 ? input.Rank - 1 : axis;
        if (actualAxis < 0 || actualAxis >= input.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        var output = new Tensor<T>((int[])input._shape.Clone());
        input.AsSpan().CopyTo(output.AsWritableSpan());
        SortAxis<T>(output, actualAxis, descending, ops);
        return output;
    }

    /// <inheritdoc/>
    public Tensor<int> ArgSort<T>(Tensor<T> input, int axis = -1, bool descending = false)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        int actualAxis = axis < 0 ? input.Rank - 1 : axis;
        if (actualAxis < 0 || actualAxis >= input.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        int outer = 1, axisLen = input._shape[actualAxis], inner = 1;
        for (int d = 0; d < actualAxis; d++) outer *= input._shape[d];
        for (int d = actualAxis + 1; d < input.Rank; d++) inner *= input._shape[d];

        var indices = new Tensor<int>((int[])input._shape.Clone());
        var inSpan = input.AsSpan();
        var idxSpan = indices.AsWritableSpan();
        var keys = new T[axisLen];
        var idx = new int[axisLen];

        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                for (int a = 0; a < axisLen; a++)
                {
                    keys[a] = inSpan[(o * axisLen + a) * inner + i];
                    idx[a] = a;
                }
                Array.Sort(keys, idx, Comparer<T>.Create((x, y) =>
                {
                    int c = ops.LessThan(x, y) ? -1 : ops.GreaterThan(x, y) ? 1 : 0;
                    return descending ? -c : c;
                }));
                for (int a = 0; a < axisLen; a++)
                    idxSpan[(o * axisLen + a) * inner + i] = idx[a];
            }
        }
        return indices;
    }

    /// <inheritdoc/>
    public Tensor<int> Histogram<T>(Tensor<T> input, int bins, T lo, T hi)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (bins < 1) throw new ArgumentOutOfRangeException(nameof(bins));
        var ops = MathHelper.GetNumericOperations<T>();
        // Reject hi <= lo before any width math. Generic
        // ops.Divide on integer T with `(hi - lo) / bins` could collapse
        // to 0 even when hi > lo (e.g. T=int, lo=0, hi=3, bins=4 →
        // width=0). Computing widthD in double space side-steps both
        // the integer-division collapse and any NaN-from-zero-width
        // fallout downstream.
        double loD = ops.ToDouble(lo);
        double hiD = ops.ToDouble(hi);
        if (!(hiD > loD))
            throw new ArgumentException("Histogram requires hi > lo.", nameof(hi));
        double widthD = (hiD - loD) / bins;

        var output = new Tensor<int>(new[] { bins });
        var counts = output.AsWritableSpan();
        var src = input.AsSpan();
        for (int i = 0; i < src.Length; i++)
        {
            T v = src[i];
            if (ops.LessThan(v, lo) || !ops.LessThan(v, hi)) continue;
            double offD = ops.ToDouble(v) - loD;
            int bin = (int)(offD / widthD);
            if (bin == bins) bin = bins - 1;
            counts[bin]++;
        }
        return output;
    }

    /// <inheritdoc/>
    public (Tensor<T> Values, Tensor<int> Counts) RunLengthEncode<T>(Tensor<T> input)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        if (src.Length == 0)
            return (new Tensor<T>(new[] { 0 }), new Tensor<int>(new[] { 0 }));

        var values = new List<T>();
        var counts = new List<int>();
        T cur = src[0]; int run = 1;
        for (int i = 1; i < src.Length; i++)
        {
            if (ops.Equals(src[i], cur)) run++;
            else { values.Add(cur); counts.Add(run); cur = src[i]; run = 1; }
        }
        values.Add(cur); counts.Add(run);

        var vTensor = new Tensor<T>(new[] { values.Count });
        var cTensor = new Tensor<int>(new[] { counts.Count });
        var vSpan = vTensor.AsWritableSpan();
        var cSpan = cTensor.AsWritableSpan();
        for (int i = 0; i < values.Count; i++) { vSpan[i] = values[i]; cSpan[i] = counts[i]; }
        return (vTensor, cTensor);
    }

    private static T Combine<T>(T a, T b, Interfaces.INumericOperations<T> ops, ReductionKind kind) => kind switch
    {
        ReductionKind.Sum => ops.Add(a, b),
        ReductionKind.Product => ops.Multiply(a, b),
        ReductionKind.Min => ops.LessThan(b, a) ? b : a,
        ReductionKind.Max => ops.GreaterThan(b, a) ? b : a,
        _ => throw new ArgumentException($"Unknown reduction kind {kind}.", nameof(kind)),
    };

    private static void SortAxis<T>(Tensor<T> t, int axis, bool descending, Interfaces.INumericOperations<T> ops)
    {
        int outer = 1, axisLen = t._shape[axis], inner = 1;
        for (int d = 0; d < axis; d++) outer *= t._shape[d];
        for (int d = axis + 1; d < t.Rank; d++) inner *= t._shape[d];

        var span = t.AsWritableSpan();
        var buf = new T[axisLen];
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                for (int a = 0; a < axisLen; a++) buf[a] = span[(o * axisLen + a) * inner + i];
                Array.Sort(buf, Comparer<T>.Create((x, y) =>
                {
                    int c = ops.LessThan(x, y) ? -1 : ops.GreaterThan(x, y) ? 1 : 0;
                    return descending ? -c : c;
                }));
                for (int a = 0; a < axisLen; a++) span[(o * axisLen + a) * inner + i] = buf[a];
            }
        }
    }
}
