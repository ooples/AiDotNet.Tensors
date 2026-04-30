// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Masked;

/// <summary>
/// Mask-aware ops over <see cref="MaskedTensor{T}"/>. Reductions skip
/// masked-out positions so a single NaN/Inf in a dense tensor doesn't
/// pollute a downstream mean — PyTorch's most-cited masked-tensor
/// motivator. Elementwise ops produce a masked output whose mask is
/// the AND of the inputs' masks.
/// </summary>
public static class MaskedOps
{
    /// <summary>Sum over valid lanes only. Returns a 1-element scalar
    /// tensor; <c>0</c> when every lane is masked out.</summary>
    public static Tensor<T> Sum<T>(MaskedTensor<T> a)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T acc = ops.Zero;
        var src = a.Values.AsSpan();
        var mask = a.PackedMask;
        for (int i = 0; i < src.Length; i++)
            if ((mask[i >> 3] & (1 << (i & 7))) != 0)
                acc = ops.Add(acc, src[i]);
        var result = new Tensor<T>(new[] { 1 });
        result.AsWritableSpan()[0] = acc;
        return result;
    }

    /// <summary>Mean over valid lanes only — divides by
    /// <see cref="MaskedTensor{T}.ValidCount"/>, not the dense element
    /// count, so masked-out entries don't pull the average down.</summary>
    public static Tensor<T> Mean<T>(MaskedTensor<T> a)
    {
        if (a.ValidCount == 0)
            throw new InvalidOperationException("Mean of an all-masked tensor is undefined.");
        var ops = MathHelper.GetNumericOperations<T>();
        var sum = Sum(a);
        T scale = ops.Divide(ops.One, ops.FromDouble(a.ValidCount));
        var result = new Tensor<T>(new[] { 1 });
        result.AsWritableSpan()[0] = ops.Multiply(sum.AsSpan()[0], scale);
        return result;
    }

    /// <summary>Population variance over valid lanes (N divisor, not
    /// N-1).</summary>
    public static Tensor<T> Var<T>(MaskedTensor<T> a)
    {
        if (a.ValidCount == 0)
            throw new InvalidOperationException("Var of an all-masked tensor is undefined.");
        var ops = MathHelper.GetNumericOperations<T>();
        var mean = Mean(a).AsSpan()[0];
        T sumSq = ops.Zero;
        var src = a.Values.AsSpan();
        var mask = a.PackedMask;
        for (int i = 0; i < src.Length; i++)
        {
            if ((mask[i >> 3] & (1 << (i & 7))) == 0) continue;
            T diff = ops.Subtract(src[i], mean);
            sumSq = ops.Add(sumSq, ops.Multiply(diff, diff));
        }
        T scale = ops.Divide(ops.One, ops.FromDouble(a.ValidCount));
        var result = new Tensor<T>(new[] { 1 });
        result.AsWritableSpan()[0] = ops.Multiply(sumSq, scale);
        return result;
    }

    /// <summary>Standard deviation = sqrt(Var).</summary>
    public static Tensor<T> Std<T>(MaskedTensor<T> a)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var v = Var(a).AsSpan()[0];
        var result = new Tensor<T>(new[] { 1 });
        result.AsWritableSpan()[0] = ops.FromDouble(Math.Sqrt(ops.ToDouble(v)));
        return result;
    }

    /// <summary>Product over valid lanes. Returns <c>1</c> when every
    /// lane is masked.</summary>
    public static Tensor<T> Prod<T>(MaskedTensor<T> a)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T acc = ops.One;
        var src = a.Values.AsSpan();
        var mask = a.PackedMask;
        for (int i = 0; i < src.Length; i++)
            if ((mask[i >> 3] & (1 << (i & 7))) != 0)
                acc = ops.Multiply(acc, src[i]);
        var result = new Tensor<T>(new[] { 1 });
        result.AsWritableSpan()[0] = acc;
        return result;
    }

    /// <summary>Min over valid lanes.</summary>
    public static Tensor<T> Min<T>(MaskedTensor<T> a)
    {
        if (a.ValidCount == 0)
            throw new InvalidOperationException("Min of an all-masked tensor is undefined.");
        var ops = MathHelper.GetNumericOperations<T>();
        var src = a.Values.AsSpan();
        var mask = a.PackedMask;
        T best = default!;
        bool seen = false;
        for (int i = 0; i < src.Length; i++)
        {
            if ((mask[i >> 3] & (1 << (i & 7))) == 0) continue;
            if (!seen) { best = src[i]; seen = true; }
            else if (ops.LessThan(src[i], best)) best = src[i];
        }
        var result = new Tensor<T>(new[] { 1 });
        result.AsWritableSpan()[0] = best;
        return result;
    }

    /// <summary>Max over valid lanes.</summary>
    public static Tensor<T> Max<T>(MaskedTensor<T> a)
    {
        if (a.ValidCount == 0)
            throw new InvalidOperationException("Max of an all-masked tensor is undefined.");
        var ops = MathHelper.GetNumericOperations<T>();
        var src = a.Values.AsSpan();
        var mask = a.PackedMask;
        T best = default!;
        bool seen = false;
        for (int i = 0; i < src.Length; i++)
        {
            if ((mask[i >> 3] & (1 << (i & 7))) == 0) continue;
            if (!seen) { best = src[i]; seen = true; }
            else if (ops.GreaterThan(src[i], best)) best = src[i];
        }
        var result = new Tensor<T>(new[] { 1 });
        result.AsWritableSpan()[0] = best;
        return result;
    }

    /// <summary>Index of the lane with the minimum value among valid
    /// lanes. Throws when every lane is masked.</summary>
    public static int ArgMin<T>(MaskedTensor<T> a)
    {
        if (a.ValidCount == 0)
            throw new InvalidOperationException("ArgMin of an all-masked tensor is undefined.");
        var ops = MathHelper.GetNumericOperations<T>();
        var src = a.Values.AsSpan();
        var mask = a.PackedMask;
        int bestIdx = -1;
        T bestVal = default!;
        for (int i = 0; i < src.Length; i++)
        {
            if ((mask[i >> 3] & (1 << (i & 7))) == 0) continue;
            if (bestIdx < 0 || ops.LessThan(src[i], bestVal)) { bestIdx = i; bestVal = src[i]; }
        }
        return bestIdx;
    }

    /// <summary>Index of the lane with the maximum value among valid
    /// lanes.</summary>
    public static int ArgMax<T>(MaskedTensor<T> a)
    {
        if (a.ValidCount == 0)
            throw new InvalidOperationException("ArgMax of an all-masked tensor is undefined.");
        var ops = MathHelper.GetNumericOperations<T>();
        var src = a.Values.AsSpan();
        var mask = a.PackedMask;
        int bestIdx = -1;
        T bestVal = default!;
        for (int i = 0; i < src.Length; i++)
        {
            if ((mask[i >> 3] & (1 << (i & 7))) == 0) continue;
            if (bestIdx < 0 || ops.GreaterThan(src[i], bestVal)) { bestIdx = i; bestVal = src[i]; }
        }
        return bestIdx;
    }

    /// <summary>Element-wise add. Output mask = AND of inputs' masks
    /// (a lane is valid iff both inputs are valid there).</summary>
    public static MaskedTensor<T> Add<T>(MaskedTensor<T> a, MaskedTensor<T> b)
    {
        EnsureSameShape(a, b);
        var ops = MathHelper.GetNumericOperations<T>();
        var values = new Tensor<T>(a.Shape);
        var src1 = a.Values.AsSpan();
        var src2 = b.Values.AsSpan();
        var dst = values.AsWritableSpan();
        var maskA = a.PackedMask;
        var maskB = b.PackedMask;
        var mask = new byte[(a.Length + 7) >> 3];
        int valid = 0;
        for (int i = 0; i < a.Length; i++)
        {
            bool ok = (maskA[i >> 3] & (1 << (i & 7))) != 0
                   && (maskB[i >> 3] & (1 << (i & 7))) != 0;
            dst[i] = ok ? ops.Add(src1[i], src2[i]) : default!;
            if (ok)
            {
                mask[i >> 3] |= (byte)(1 << (i & 7));
                valid++;
            }
        }
        return new MaskedTensor<T>(values, mask, valid);
    }

    /// <summary>Element-wise multiply — same mask semantics as
    /// <see cref="Add{T}"/>.</summary>
    public static MaskedTensor<T> Multiply<T>(MaskedTensor<T> a, MaskedTensor<T> b)
    {
        EnsureSameShape(a, b);
        var ops = MathHelper.GetNumericOperations<T>();
        var values = new Tensor<T>(a.Shape);
        var src1 = a.Values.AsSpan();
        var src2 = b.Values.AsSpan();
        var dst = values.AsWritableSpan();
        var maskA = a.PackedMask;
        var maskB = b.PackedMask;
        var mask = new byte[(a.Length + 7) >> 3];
        int valid = 0;
        for (int i = 0; i < a.Length; i++)
        {
            bool ok = (maskA[i >> 3] & (1 << (i & 7))) != 0
                   && (maskB[i >> 3] & (1 << (i & 7))) != 0;
            dst[i] = ok ? ops.Multiply(src1[i], src2[i]) : default!;
            if (ok)
            {
                mask[i >> 3] |= (byte)(1 << (i & 7));
                valid++;
            }
        }
        return new MaskedTensor<T>(values, mask, valid);
    }

    private static void EnsureSameShape<T>(MaskedTensor<T> a, MaskedTensor<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Shape mismatch: {a.Length} vs {b.Length}.");
        if (a.Rank != b.Rank)
            throw new ArgumentException($"Rank mismatch: {a.Rank} vs {b.Rank}.");
        for (int i = 0; i < a.Rank; i++)
            if (a.Shape[i] != b.Shape[i])
                throw new ArgumentException($"Shape mismatch on axis {i}: {a.Shape[i]} vs {b.Shape[i]}.");
    }
}
