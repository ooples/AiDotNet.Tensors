// Copyright (c) AiDotNet. All rights reserved.
//
// int4-NATIVE VNNI GEMM (Phase A / #1622, L5a compute rung).
//
// SgemmWithInt4GroupScaled (the sibling file) keeps the activation in fp32 and only the WEIGHT
// int4 — accurate but the compute is fp32-rate. This kernel additionally quantizes the activation
// to uint8 per row and runs the matmul as a u8×s8 integer dot via VPDPBUSD (AVX-VNNI), the same
// machinery the int8 path uses, so quantized GEMM runs at int8 throughput (2-4x fp32). The weight
// stays 4-bit in memory (unpacked to int8 per element in the supplied buffer); group-quant means
// one fp32 scale per `groupSize` block, so the int32 accumulation + dequant is done PER GROUP:
//   c[i,j] = Σ_g (Σ_{p∈g} u8[i,p]·w8[j,p]  −  128·Σ_{p∈g} w8[j,p]) · actScale[i] · groupScale[g]
// The −128·Σw term removes the uint8 zero-point (u8 = s8 + 128), identical to the int8 path's
// bRowSum correction but evaluated per group. N-parallel (partitions output features), so batch=1
// saturates all cores. Both runs of a self-relative test quantize identically, so the activation
// quantization is invisible to the assertions.

using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>
    /// True when the int4-native VNNI path can run (AVX-VNNI present).
    /// Gated NET9_0_OR_GREATER (not NET8): <see cref="AvxVnni"/> is
    /// <c>[RequiresPreviewFeatures]</c> on net8.0 (CA2252) and only stable from
    /// .NET 9. Below that the dispatcher uses the fp32 weight-only fallback.
    /// </summary>
    internal static bool Int4VnniAvailable =>
#if NET9_0_OR_GREATER
        AvxVnni.IsSupported;
#else
        false;
#endif

    /// <summary>
    /// Routes an int4 group-quant weight-only GEMM to the int4-native VNNI integer kernel when
    /// AVX-VNNI is present (2-4x fp32 via u8×s8 VPDPBUSD, at the cost of int8 activation quant),
    /// otherwise to the fp32 weight-only path (exact activation, dequant int4 weight). The int4
    /// route is only reached for weights a model deliberately stored int4 (foundation-scale), so
    /// the aggressive activation quant is already opted into.
    /// </summary>
    internal static void SgemmWithInt4GroupScaledDispatch(
        float[] a, sbyte[] wData, float[] groupScales, int groupSize, float[] c, int m, int k, int n)
    {
        if (Int4VnniAvailable)
            SgemmWithInt4GroupScaledVnni(a, wData, groupScales, groupSize, c, m, k, n);
        else
            SgemmWithInt4GroupScaled(a, wData, groupScales, groupSize, c, m, k, n);
    }

    /// <summary>
    /// Activation-quantized int4 weight-only GEMM: <c>C[m,n] = A[m,k] · Wᵀ</c> with <c>W[n,k]</c>
    /// supplied as sign-extended int4 (<paramref name="wData"/>) + per-group fp32 scales
    /// (<paramref name="groupScales"/>), computed as a u8×s8 integer matmul (VPDPBUSD) with
    /// per-group dequant. Falls back to a correct scalar integer path off the VNNI fast path /
    /// when <paramref name="groupSize"/> does not tile the rows. Output is overwritten.
    /// </summary>
    internal static void SgemmWithInt4GroupScaledVnni(
        float[] a,
        sbyte[] wData, float[] groupScales, int groupSize,
        float[] c,
        int m, int k, int n)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (wData is null) throw new ArgumentNullException(nameof(wData));
        if (groupScales is null) throw new ArgumentNullException(nameof(groupScales));
        if (c is null) throw new ArgumentNullException(nameof(c));
        if (m <= 0 || n <= 0) return;
        if (k <= 0) { Array.Clear(c, 0, Math.Min(c.Length, m * n)); return; }
        if (groupSize <= 0) throw new ArgumentOutOfRangeException(nameof(groupSize));
        if (wData.Length < (long)n * k) throw new ArgumentException("wData.Length must be >= n*k", nameof(wData));
        if (a.Length < (long)m * k) throw new ArgumentException("a.Length must be >= m*k", nameof(a));
        if (c.Length < (long)m * n) throw new ArgumentException("c.Length must be >= m*n", nameof(c));

        // Quantize activations → uint8 per row (dynamic), reusing the int8 path's quantizer.
        var aU8 = new byte[m * k];
        var actScale = new float[m];
        Int8Quantizer.QuantizeActivationsPerRowToUint8(a, m, k, aU8, actScale);

        int cores = Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism);
        int numChunks = Math.Max(1, Math.Min(n, cores * 4));
        int perChunk = (n + numChunks - 1) / numChunks;
        bool vnni = Int4VnniAvailable;

        CpuParallelSettings.LightweightParallel(numChunks, chunk =>
        {
            int j0 = chunk * perChunk;
            if (j0 >= n) return;
            int j1 = Math.Min(j0 + perChunk, n);
            for (int j = j0; j < j1; j++)
                ComputeColumn(aU8, actScale, wData, groupScales, groupSize, c, m, k, n, j, vnni);
        });
    }

    private static void ComputeColumn(
        byte[] aU8, float[] actScale, sbyte[] wData, float[] groupScales, int groupSize,
        float[] c, int m, int k, int n, int j, bool vnni)
    {
        long wBase = (long)j * k;
        for (int i = 0; i < m; i++)
        {
            long aBase = (long)i * k;
            double acc = 0.0;
            int p = 0;
            while (p < k)
            {
                long flat = wBase + p;
                int g = (int)(flat / groupSize);
                // Length of the run that stays inside BOTH this row's remaining k and group g.
                int runToGroupEnd = (int)((long)(g + 1) * groupSize - flat);
                int len = Math.Min(runToGroupEnd, k - p);

                int dot = 0, wsum = 0;
                int aOff = (int)(aBase + p);
                int wOff = (int)(wBase + p);
#if NET9_0_OR_GREATER
                int t = 0;
                if (vnni && len >= 32)
                {
                    var accV = Vector256<int>.Zero;
                    var sumV = Vector256<int>.Zero;
                    var ones = Vector256.Create((byte)1);
                    for (; t + 32 <= len; t += 32)
                    {
                        var av = Vector256.LoadUnsafe(ref aU8[aOff + t]);
                        var wv = Vector256.LoadUnsafe(ref Unsafe.As<sbyte, byte>(ref wData[wOff + t])).AsSByte();
                        accV = AvxVnni.MultiplyWideningAndAdd(accV, av, wv);   // Σ u8·s8
                        sumV = AvxVnni.MultiplyWideningAndAdd(sumV, ones, wv); // Σ s8 (for zero-point)
                    }
                    dot += HorizontalSum(accV);
                    wsum += HorizontalSum(sumV);
                    for (; t < len; t++) { dot += aU8[aOff + t] * wData[wOff + t]; wsum += wData[wOff + t]; }
                }
                else
                {
                    for (; t < len; t++) { dot += aU8[aOff + t] * wData[wOff + t]; wsum += wData[wOff + t]; }
                }
#else
                for (int t = 0; t < len; t++) { dot += aU8[aOff + t] * wData[wOff + t]; wsum += wData[wOff + t]; }
#endif
                int signed = dot - 128 * wsum;          // remove uint8 zero-point (u8 = s8 + 128)
                acc += signed * (double)groupScales[g]; // per-group weight scale
                p += len;
            }
            c[(long)i * n + j] = (float)(acc * actScale[i]); // per-row activation scale
        }
    }

#if NET5_0_OR_GREATER
    private static int HorizontalSum(Vector256<int> v)
    {
        var lo = v.GetLower();
        var hi = v.GetUpper();
        var s = Sse2.Add(lo, hi);                       // 4 lanes
        s = Sse2.Add(s, Sse2.Shuffle(s, 0b_01_00_11_10));
        s = Sse2.Add(s, Sse2.Shuffle(s, 0b_00_01_00_01));
        return s.ToScalar();
    }
#endif
}
