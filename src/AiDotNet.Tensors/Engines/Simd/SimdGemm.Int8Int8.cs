using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Issue #465 — W8A8 (int8 weights × int8 activations) GEMM, true int8 compute.
///
/// <para><b>Phase 0 (de-risk).</b> This file currently holds the AVX2 int8×int8
/// micro-kernel and is exercised by a viability microbenchmark vs fp32
/// <see cref="SimdGemm"/>. The whole W8A8 program only proceeds to the full
/// fused entry point (<c>SgemmA8W8RowScaledCachedB</c>, Phase 1) if int8 actually
/// wins on AVX2-only hardware. On a VNNI-capable host the single-instruction
/// <c>VPDPBUSD</c> path (Phase 2) is expected to win clearly; on AVX2-only hosts
/// the emulation (<c>VPMADDUBSW</c>+<c>VPMADDWD</c>) carries real overhead and may
/// not beat fp32 FMA — which is exactly what Phase 0 measures.</para>
///
/// <para><b>Math.</b> For an int8 dot product <c>Σ aᵢ·bᵢ</c> with <c>a</c> the
/// activation (sbyte) and <c>b</c> the weight (sbyte), <c>VPMADDUBSW</c> needs the
/// left operand <em>unsigned</em>. We shift the activation by +128 to
/// <c>u = a + 128 ∈ [1, 255]</c> and recover the signed dot product with a
/// per-weight-row correction:
/// <code>
///   Σ aᵢ·bᵢ = Σ (uᵢ − 128)·bᵢ = (Σ uᵢ·bᵢ) − 128·(Σ bᵢ)
/// </code>
/// The first term is the <c>VPMADDUBSW</c>/<c>VPMADDWD</c> int32 accumulation; the
/// second is a per-output correction using a precomputed per-row weight sum
/// (<c>O(n)</c>, negligible vs the <c>O(m·n·k)</c> GEMM). The int32 result is
/// dequantized with <c>actScale[row] · weightScale[col]</c>.</para>
/// </summary>
internal static partial class SimdGemm
{
#if NET5_0_OR_GREATER
    /// <summary>
    /// True when the AVX2 int8×int8 emulation path can run (AVX2 required for
    /// <c>VPMADDUBSW</c>/<c>VPMADDWD</c>).
    /// </summary>
    internal static bool Int8Int8Avx2Available => Avx2.IsSupported;
#else
    internal static bool Int8Int8Avx2Available => false;
#endif

#if NET9_0_OR_GREATER
    /// <summary>
    /// True when the AVX-VNNI <c>VPDPBUSD</c> fast path can run (256-bit AVX-VNNI).
    /// VNNI accumulates the int8 dot product directly into int32 in a single
    /// instruction — no int16 intermediate, so (unlike the AVX2 emulation) it does
    /// not saturate and is both faster and more accurate.
    /// </summary>
    /// <remarks>
    /// Gated NET9_0_OR_GREATER (not NET8): <see cref="AvxVnni"/> is marked
    /// <c>[RequiresPreviewFeatures]</c> on net8.0 (CA2252) and only became a stable,
    /// non-preview intrinsic in .NET 9. On net8.0 / net6.0 / netstandard2.0 / net462
    /// this path is compiled out and the dispatcher falls back to the AVX2 emulation
    /// (net6.0+) or the scalar kernel.
    /// </remarks>
    internal static bool Int8Int8VnniAvailable => AvxVnni.IsSupported;
#else
    internal static bool Int8Int8VnniAvailable => false;
#endif

    /// <summary>
    /// Phase 0 reference/throughput kernel: C[m,n] (fp32) = dequant( Aq8 · Bq8ᵀ ).
    /// <paramref name="aU8"/> are activations already quantized to int8 then shifted
    /// to unsigned (<c>u = q + 128</c>), row-major [m, k]. <paramref name="bI8"/> are
    /// weights as sbyte, row-major [n, k] (each row a weight vector — the natural
    /// <c>VPMADDUBSW</c> read pattern). <paramref name="bRowSum"/>[j] = Σₖ bI8[j,k]
    /// (precomputed, for the −128·Σb correction). Dequant: actScale[i]·wScale[j].
    /// </summary>
    /// <remarks>
    /// This is the de-risk kernel — straightforward register-blocked accumulation,
    /// not yet the cache-tiled production driver. It is representative of int8
    /// inner-loop throughput (especially the m≈1 decode case) for the AVX2-vs-fp32
    /// viability decision.
    /// </remarks>
    internal static unsafe void MatMulInt8Int8Avx2(
        ReadOnlySpan<byte> aU8, ReadOnlySpan<sbyte> bI8,
        ReadOnlySpan<int> bRowSum,
        ReadOnlySpan<float> actScale, ReadOnlySpan<float> wScale,
        Span<float> c, int m, int k, int n)
    {
#if NET5_0_OR_GREATER
        if (!Avx2.IsSupported)
        {
            MatMulInt8Int8Scalar(aU8, bI8, bRowSum, actScale, wScale, c, m, k, n);
            return;
        }

        int kVec = k & ~31;                 // 32 int8 lanes per VPMADDUBSW
        var ones16 = Vector256.Create((short)1);

        fixed (byte* pa0 = aU8)
        fixed (sbyte* pb0 = bI8)
        fixed (float* pc0 = c)
        {
            byte* pa = pa0; sbyte* pb = pb0; float* pc = pc0;
            for (int i = 0; i < m; i++)
            {
                byte* aRow = pa + (long)i * k;
                float aScale = actScale[i];
                for (int j = 0; j < n; j++)
                {
                    sbyte* bRow = pb + (long)j * k;
                    // Four independent int32 accumulator vectors to hide the
                    // VPMADDUBSW→VPMADDWD dependency chain latency.
                    var acc0 = Vector256<int>.Zero;
                    var acc1 = Vector256<int>.Zero;
                    int t = 0;
                    for (; t <= kVec - 64; t += 64)
                    {
                        var au0 = Avx.LoadVector256(aRow + t);
                        var bw0 = Avx.LoadVector256(bRow + t);
                        var au1 = Avx.LoadVector256(aRow + t + 32);
                        var bw1 = Avx.LoadVector256(bRow + t + 32);
                        // uint8 × sbyte → int16 pairs (VPMADDUBSW), then ×1 pair-sum
                        // into int32 (VPMADDWD) to avoid int16 saturation.
                        acc0 = Avx2.Add(acc0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(au0, bw0), ones16));
                        acc1 = Avx2.Add(acc1, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(au1, bw1), ones16));
                    }
                    for (; t < kVec; t += 32)
                    {
                        var au = Avx.LoadVector256(aRow + t);
                        var bw = Avx.LoadVector256(bRow + t);
                        acc0 = Avx2.Add(acc0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(au, bw), ones16));
                    }
                    int raw = HorizontalSumInt32(Avx2.Add(acc0, acc1));
                    // Scalar tail (k % 32).
                    for (; t < k; t++) raw += aRow[t] * bRow[t];
                    // Undo the +128 activation shift, then dequantize.
                    int signed = raw - 128 * bRowSum[j];
                    pc[(long)i * n + j] = signed * (aScale * wScale[j]);
                }
            }
        }
#else
        MatMulInt8Int8Scalar(aU8, bI8, bRowSum, actScale, wScale, c, m, k, n);
#endif
    }

#if NET9_0_OR_GREATER
    /// <summary>
    /// Phase 2 VNNI fast path: same contract as <see cref="MatMulInt8Int8Avx2"/> but
    /// uses <c>VPDPBUSD</c> (<see cref="AvxVnni.MultiplyWideningAndAdd"/>) to accumulate
    /// 32 int8 products into int32 per instruction with no int16 saturation. Gated on
    /// <see cref="Int8Int8VnniAvailable"/>; falls back to AVX2 when VNNI is absent.
    /// </summary>
    internal static unsafe void MatMulInt8Int8Vnni(
        ReadOnlySpan<byte> aU8, ReadOnlySpan<sbyte> bI8,
        ReadOnlySpan<int> bRowSum,
        ReadOnlySpan<float> actScale, ReadOnlySpan<float> wScale,
        Span<float> c, int m, int k, int n)
    {
        if (!AvxVnni.IsSupported)
        {
            MatMulInt8Int8Avx2(aU8, bI8, bRowSum, actScale, wScale, c, m, k, n);
            return;
        }

        int kVec = k & ~31;                 // 32 int8 lanes per VPDPBUSD
        fixed (byte* pa0 = aU8)
        fixed (sbyte* pb0 = bI8)
        fixed (float* pc0 = c)
        {
            byte* pa = pa0; sbyte* pb = pb0; float* pc = pc0;
            for (int i = 0; i < m; i++)
            {
                byte* aRow = pa + (long)i * k;
                float aScale = actScale[i];
                for (int j = 0; j < n; j++)
                {
                    sbyte* bRow = pb + (long)j * k;
                    var acc0 = Vector256<int>.Zero;
                    var acc1 = Vector256<int>.Zero;
                    int t = 0;
                    for (; t <= kVec - 64; t += 64)
                    {
                        acc0 = AvxVnni.MultiplyWideningAndAdd(acc0, Avx.LoadVector256(aRow + t), Avx.LoadVector256(bRow + t));
                        acc1 = AvxVnni.MultiplyWideningAndAdd(acc1, Avx.LoadVector256(aRow + t + 32), Avx.LoadVector256(bRow + t + 32));
                    }
                    for (; t < kVec; t += 32)
                        acc0 = AvxVnni.MultiplyWideningAndAdd(acc0, Avx.LoadVector256(aRow + t), Avx.LoadVector256(bRow + t));
                    int raw = HorizontalSumInt32(Avx2.Add(acc0, acc1));
                    for (; t < k; t++) raw += aRow[t] * bRow[t];
                    int signed = raw - 128 * bRowSum[j];
                    pc[(long)i * n + j] = signed * (aScale * wScale[j]);
                }
            }
        }
    }
#endif

#if NET5_0_OR_GREATER
    private static int HorizontalSumInt32(Vector256<int> v)
    {
        var lo = v.GetLower();
        var hi = v.GetUpper();
        var sum128 = Sse2.Add(lo, hi);
        sum128 = Sse2.Add(sum128, Sse2.Shuffle(sum128, 0b_01_00_11_10));
        sum128 = Sse2.Add(sum128, Sse2.Shuffle(sum128, 0b_10_11_00_01));
        return sum128.GetElement(0);
    }
#endif

    // Per-weight-row sum cache (Σₖ bInt8[j,k], for the −128·Σb correction).
    // Keyed on the weight array identity: the consumer reuses the same pre-quantized
    // weight buffer across many calls, so this is computed once per weight matrix.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<sbyte[], int[]>
        s_bRowSumCache = new();

    /// <summary>
    /// W8A8 fused entry point (#465 Phase 1): C[m,n] = dequant( quant(A) · Bᵀ ).
    /// Quantizes the fp32 activations <paramref name="a"/> internally (dynamic,
    /// per-row symmetric int8 → uint8), runs the int8×int8 GEMM against the
    /// pre-quantized weights <paramref name="bInt8"/> (<c>[n, k]</c>, each row a
    /// weight vector with scale <paramref name="rowScales"/>[row]), and dequantizes
    /// the int32 accumulator with <c>actScale[row]·rowScales[col]</c>. The
    /// per-weight-row sum needed for the +128 shift correction is computed once and
    /// cached on the weight-array identity.
    /// </summary>
    /// <remarks>
    /// Routes to VNNI (<c>VPDPBUSD</c>, Phase 2) when available, else the AVX2
    /// <c>VPMADDUBSW</c> emulation, else scalar. Activation quant is <c>O(m·k)</c>
    /// (negligible vs the <c>O(m·n·k)</c> GEMM). Drop-in for the consumer's
    /// <c>Int8WeightOnlyMatMul</c> behind a W8A8 opt-in.
    /// </remarks>
    public static void SgemmA8W8RowScaledCachedB(
        ReadOnlySpan<float> a, sbyte[] bInt8, ReadOnlySpan<float> rowScales,
        Span<float> c, int m, int k, int n)
    {
        if (bInt8 is null) throw new ArgumentNullException(nameof(bInt8));
        if (m <= 0 || n <= 0) return;
        if (k <= 0) { c.Slice(0, m * n).Clear(); return; }
        if (a.Length < m * k) throw new ArgumentException("a.Length must be >= m*k", nameof(a));
        if (bInt8.Length < (long)n * k) throw new ArgumentException("bInt8.Length must be >= n*k", nameof(bInt8));
        if (rowScales.Length < n) throw new ArgumentException("rowScales.Length must be >= n", nameof(rowScales));
        if (c.Length < m * n) throw new ArgumentException("c.Length must be >= m*n", nameof(c));

        // Quantize activations → uint8 + per-row scale (dynamic, per call).
        var aU8 = new byte[m * k];
        var actScale = new float[m];
        Int8Quantizer.QuantizeActivationsPerRowToUint8(a, m, k, aU8, actScale);

        int[] bRowSum = GetOrBuildBRowSum(bInt8, n, k);

        MatMulInt8Int8Dispatch(aU8, bInt8, bRowSum, actScale, rowScales, c, m, k, n);
    }

    /// <summary>Picks the best available int8×int8 kernel: VNNI → AVX2 → scalar.</summary>
    internal static void MatMulInt8Int8Dispatch(
        ReadOnlySpan<byte> aU8, ReadOnlySpan<sbyte> bI8, ReadOnlySpan<int> bRowSum,
        ReadOnlySpan<float> actScale, ReadOnlySpan<float> wScale,
        Span<float> c, int m, int k, int n)
    {
#if NET9_0_OR_GREATER
        if (Int8Int8VnniAvailable)
        {
            MatMulInt8Int8Vnni(aU8, bI8, bRowSum, actScale, wScale, c, m, k, n);
            return;
        }
#endif
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported)
        {
            MatMulInt8Int8Avx2(aU8, bI8, bRowSum, actScale, wScale, c, m, k, n);
            return;
        }
#endif
        MatMulInt8Int8Scalar(aU8, bI8, bRowSum, actScale, wScale, c, m, k, n);
    }

    private static int[] GetOrBuildBRowSum(sbyte[] bInt8, int n, int k)
    {
        if (s_bRowSumCache.TryGetValue(bInt8, out var cached) && cached.Length == n)
            return cached;
        var rowSum = new int[n];
        for (int j = 0; j < n; j++)
        {
            int off = j * k, sum = 0;
            for (int t = 0; t < k; t++) sum += bInt8[off + t];
            rowSum[j] = sum;
        }
        // ConditionalWeakTable.AddOrUpdate is not on net471; Remove+Add is fine
        // (single-writer-per-weight-matrix in practice; a benign recompute on race).
        s_bRowSumCache.Remove(bInt8);
        s_bRowSumCache.Add(bInt8, rowSum);
        return rowSum;
    }

    /// <summary>Scalar reference (net471 / non-AVX2): same math, no intrinsics.</summary>
    internal static void MatMulInt8Int8Scalar(
        ReadOnlySpan<byte> aU8, ReadOnlySpan<sbyte> bI8,
        ReadOnlySpan<int> bRowSum,
        ReadOnlySpan<float> actScale, ReadOnlySpan<float> wScale,
        Span<float> c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            float aScale = actScale[i];
            for (int j = 0; j < n; j++)
            {
                int raw = 0;
                int aOff = i * k, bOff = j * k;
                for (int t = 0; t < k; t++) raw += aU8[aOff + t] * bI8[bOff + t];
                int signed = raw - 128 * bRowSum[j];
                c[i * n + j] = signed * (aScale * wScale[j]);
            }
        }
    }
}
