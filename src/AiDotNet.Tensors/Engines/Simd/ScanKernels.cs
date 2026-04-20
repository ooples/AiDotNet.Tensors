// Copyright (c) AiDotNet. All rights reserved.
// SIMD prefix-sum / prefix-product / running-max / running-min primitives.
// Called from TensorCumSum / TensorCumProd / TensorCumMax / TensorCumMin
// on the contiguous float32 fast path.
//
// Algorithm: Sklansky scan within each 8-lane AVX2 block followed by a
// scalar carry-forward between blocks. For sum this is the classic
// shuffle-add ladder:
//   v = [a,b,c,d,e,f,g,h]
//   shift-1: [0,a,b,c,d,e,f,g]; v += shifted        -> [a, a+b, b+c, ..., g+h]
//   shift-2: [0,0,v0,v1,v2,v3,v4,v5]; v += shifted   -> prefix-of-pairs
//   shift-4: [0,0,0,0,v0,v1,v2,v3]; v += shifted    -> full prefix
// Then add the previous block's running total to every lane.

using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

internal static class ScanKernels
{
    /// <summary>
    /// In-place prefix sum of a contiguous <see cref="float"/> span.
    /// Uses an AVX2 Sklansky block scan when available, scalar otherwise.
    /// Output[i] = Σ_{j ≤ i} input[j].
    /// </summary>
    public static void PrefixSumFloat(ReadOnlySpan<float> input, Span<float> output)
    {
        int n = input.Length;
        if (output.Length < n) throw new ArgumentException("output too small");

#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && n >= 8)
        {
            PrefixSumFloatAvx2(input, output);
            return;
        }
#endif
        float acc = 0f;
        for (int i = 0; i < n; i++)
        {
            acc += input[i];
            output[i] = acc;
        }
    }

    /// <summary>
    /// In-place running-max of a contiguous <see cref="float"/> span.
    /// Output[i] = max_{j ≤ i} input[j].
    /// </summary>
    public static void RunningMaxFloat(ReadOnlySpan<float> input, Span<float> output)
    {
        int n = input.Length;
        if (output.Length < n) throw new ArgumentException("output too small");

#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && n >= 8)
        {
            RunningMaxFloatAvx2(input, output);
            return;
        }
#endif
        float acc = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            if (input[i] > acc) acc = input[i];
            output[i] = acc;
        }
    }

    /// <summary>Running-min (symmetric to RunningMax).</summary>
    public static void RunningMinFloat(ReadOnlySpan<float> input, Span<float> output)
    {
        int n = input.Length;
        if (output.Length < n) throw new ArgumentException("output too small");

#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && n >= 8)
        {
            RunningMinFloatAvx2(input, output);
            return;
        }
#endif
        float acc = float.PositiveInfinity;
        for (int i = 0; i < n; i++)
        {
            if (input[i] < acc) acc = input[i];
            output[i] = acc;
        }
    }

    /// <summary>Running-product (small-magnitude values only; fp32 underflows fast).</summary>
    public static void PrefixProductFloat(ReadOnlySpan<float> input, Span<float> output)
    {
        int n = input.Length;
        if (output.Length < n) throw new ArgumentException("output too small");

        float acc = 1f;
        for (int i = 0; i < n; i++)
        {
            acc *= input[i];
            output[i] = acc;
        }
    }

#if NET5_0_OR_GREATER
    // Sklansky prefix sum within a 256-bit (8-lane) vector.
    //
    // Given v = [a, b, c, d, e, f, g, h],  result[i] = Σ_{j<=i} v[j].
    // Three shift-and-add stages: shift by 1, shift by 2, shift by 4.
    // Uses VPERMD with an index vector + AND mask to implement "shift by N
    // lanes with zeros inserted at the low end".
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> PrefixSumVec8(Vector256<float> v)
    {
        // Stage 1: add v shifted right by 1 lane.
        var idx1 = Vector256.Create(0, 0, 1, 2, 3, 4, 5, 6);
        var mask1 = Vector256.Create(0, -1, -1, -1, -1, -1, -1, -1).AsSingle();
        var s1 = Avx.And(Avx2.PermuteVar8x32(v.AsInt32(), idx1).AsSingle(), mask1);
        v = Avx.Add(v, s1);

        // Stage 2: add v shifted right by 2 lanes.
        var idx2 = Vector256.Create(0, 0, 0, 1, 2, 3, 4, 5);
        var mask2 = Vector256.Create(0, 0, -1, -1, -1, -1, -1, -1).AsSingle();
        var s2 = Avx.And(Avx2.PermuteVar8x32(v.AsInt32(), idx2).AsSingle(), mask2);
        v = Avx.Add(v, s2);

        // Stage 3: add v shifted right by 4 lanes.
        var idx4 = Vector256.Create(0, 0, 0, 0, 0, 1, 2, 3);
        var mask4 = Vector256.Create(0, 0, 0, 0, -1, -1, -1, -1).AsSingle();
        var s4 = Avx.And(Avx2.PermuteVar8x32(v.AsInt32(), idx4).AsSingle(), mask4);
        v = Avx.Add(v, s4);

        return v;
    }

    private static unsafe void PrefixSumFloatAvx2(ReadOnlySpan<float> input, Span<float> output)
    {
        int n = input.Length;
        fixed (float* src = input)
        fixed (float* dst = output)
        {
            float carry = 0f;
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var v = Avx.LoadVector256(src + i);
                var scanned = PrefixSumVec8(v);
                var withCarry = Avx.Add(scanned, Vector256.Create(carry));
                Avx.Store(dst + i, withCarry);
                // New carry = last lane of the scanned block including carry.
                // GetElement isn't const-indexable cross-version; use store-to-lane.
                // Cheapest: read the last element of the just-written block.
                carry = dst[i + 7];
            }
            for (; i < n; i++)
            {
                carry += src[i];
                dst[i] = carry;
            }
        }
    }

    // For max / min we fall back to the scalar loop inside AVX2 — the
    // Sklansky trick works for any associative op, but we keep the
    // implementation small for the first landing and revisit if benchmarks
    // flag running-max as a hot path.
    private static void RunningMaxFloatAvx2(ReadOnlySpan<float> input, Span<float> output)
    {
        float acc = float.NegativeInfinity;
        for (int i = 0; i < input.Length; i++)
        {
            if (input[i] > acc) acc = input[i];
            output[i] = acc;
        }
    }

    private static void RunningMinFloatAvx2(ReadOnlySpan<float> input, Span<float> output)
    {
        float acc = float.PositiveInfinity;
        for (int i = 0; i < input.Length; i++)
        {
            if (input[i] < acc) acc = input[i];
            output[i] = acc;
        }
    }
#endif
}
