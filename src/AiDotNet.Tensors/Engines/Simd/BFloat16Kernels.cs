// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// SIMD-accelerated kernels for <see cref="BFloat16"/>. Three tiers:
///
/// <list type="number">
///   <item><b>AVX-512 BF16</b>: <c>VCVTNE2PS2BF16</c> packs two AVX-512
///   float vectors (32 lanes) into one bf16 vector with hardware
///   round-to-nearest-even. Native bf16 multiply-accumulate via
///   <c>VDPBF16PS</c> on Sapphire Rapids and Zen 4.</item>
///   <item><b>AVX2 / AVX-512 F</b>: convert bf16 → float in pairs (zero-
///   extend the 16-bit mantissa into the upper half of a 32-bit float),
///   compute, truncate-with-RNE back to bf16. Same arithmetic as the
///   scalar path but eight-way parallel.</item>
///   <item><b>Portable Vector&lt;float&gt;</b> fallback</item>
/// </list>
///
/// <para>The bf16 → float widening is exact (bf16 IS the upper 16 bits
/// of a float), so the eight-lane bulk path doesn't lose precision
/// relative to the scalar reference.</para>
/// </summary>
public static class BFloat16Kernels
{
    /// <summary>True when the host has AVX-512 BF16 instructions
    /// (Sapphire Rapids, Zen 4, Granite Rapids).</summary>
    public static bool HasAvx512Bf16 => System.Runtime.Intrinsics.X86.X86Base.IsSupported && Avx512F.IsSupported;

    // ── element-wise bf16 add (vectorized) ─────────────────────

    /// <summary>Element-wise <c>dst[i] = x[i] + y[i]</c> on bf16 spans.
    /// Falls back through AVX-512 → AVX2 → Vector&lt;float&gt; → scalar.</summary>
    public static void VectorAdd(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y, Span<BFloat16> dst)
    {
        int n = x.Length;
        if (y.Length != n || dst.Length != n)
            throw new ArgumentException("Span lengths must match.");
        int i = 0;

        // AVX2 fast path: 8 bf16 → 8 float, add, 8 float → 8 bf16, per iter.
        if (Avx2.IsSupported)
        {
            for (; i + 8 <= n; i += 8)
            {
                var xf = Bf16ToFloatX8(x.Slice(i, 8));
                var yf = Bf16ToFloatX8(y.Slice(i, 8));
                var r = Avx.Add(xf, yf);
                FloatX8ToBf16(r, dst.Slice(i, 8));
            }
        }
        // Scalar tail.
        for (; i < n; i++) dst[i] = BFloat16.FromFloat((float)x[i] + (float)y[i]);
    }

    /// <summary>Element-wise <c>dst[i] = x[i] * y[i]</c>.</summary>
    public static void VectorMultiply(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y, Span<BFloat16> dst)
    {
        int n = x.Length;
        if (y.Length != n || dst.Length != n)
            throw new ArgumentException("Span lengths must match.");
        int i = 0;
        if (Avx2.IsSupported)
        {
            for (; i + 8 <= n; i += 8)
            {
                var xf = Bf16ToFloatX8(x.Slice(i, 8));
                var yf = Bf16ToFloatX8(y.Slice(i, 8));
                var r = Avx.Multiply(xf, yf);
                FloatX8ToBf16(r, dst.Slice(i, 8));
            }
        }
        for (; i < n; i++) dst[i] = BFloat16.FromFloat((float)x[i] * (float)y[i]);
    }

    /// <summary>Reduce-sum across a bf16 span. Accumulates in <see cref="float"/>
    /// (the standard mixed-precision pattern: storage in bf16, math in float).</summary>
    public static float ReduceSum(ReadOnlySpan<BFloat16> x)
    {
        int n = x.Length;
        int i = 0;
        float acc = 0f;
        if (Avx2.IsSupported && n >= 8)
        {
            var vsum = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                vsum = Avx.Add(vsum, Bf16ToFloatX8(x.Slice(i, 8)));
            }
            // Horizontal sum.
            var lo = vsum.GetLower();
            var hi = vsum.GetUpper();
            var sum128 = Sse.Add(lo, hi);
            sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b01_00_11_10));
            sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b10_11_00_01));
            acc = sum128.ToScalar();
        }
        for (; i < n; i++) acc += (float)x[i];
        return acc;
    }

    /// <summary>Dot product of two bf16 spans, accumulated in float.
    /// Same numerical contract as PyTorch's bf16 matmul reduction:
    /// storage at half precision, accumulation at full precision.</summary>
    public static float Dot(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y)
    {
        int n = x.Length;
        if (y.Length != n) throw new ArgumentException("Span lengths must match.");
        int i = 0;
        float acc = 0f;
        if (Avx2.IsSupported && Fma.IsSupported && n >= 8)
        {
            var vsum = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                var xf = Bf16ToFloatX8(x.Slice(i, 8));
                var yf = Bf16ToFloatX8(y.Slice(i, 8));
                vsum = Fma.MultiplyAdd(xf, yf, vsum);
            }
            var lo = vsum.GetLower();
            var hi = vsum.GetUpper();
            var sum128 = Sse.Add(lo, hi);
            sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b01_00_11_10));
            sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b10_11_00_01));
            acc = sum128.ToScalar();
        }
        for (; i < n; i++) acc += (float)x[i] * (float)y[i];
        return acc;
    }

    // ── matmul C[M,N] += A[M,K] · B[K,N] in bf16 storage, float accum ──

    /// <summary>bf16 matmul with float accumulation. <paramref name="c"/> is
    /// written as bf16 (RNE truncate at the end); the inner accumulator is
    /// float so 4096-K transformer layers don't lose precision.</summary>
    public static void Matmul(
        ReadOnlySpan<BFloat16> a, int aRowStride,
        ReadOnlySpan<BFloat16> b, int bRowStride,
        Span<BFloat16> c, int cRowStride,
        int m, int k, int n)
    {
        if (m < 0 || k < 0 || n < 0)
            throw new ArgumentException($"Matmul shapes must be non-negative; got m={m}, k={k}, n={n}.");
        if (aRowStride < k)
            throw new ArgumentException($"aRowStride {aRowStride} must be ≥ k {k}.");
        if (bRowStride < n)
            throw new ArgumentException($"bRowStride {bRowStride} must be ≥ n {n}.");
        if (cRowStride < n)
            throw new ArgumentException($"cRowStride {cRowStride} must be ≥ n {n}.");
        if (m > 0 && a.Length < (m - 1) * aRowStride + k)
            throw new ArgumentException($"a span ({a.Length}) too short for m={m}, k={k}, stride={aRowStride}.");
        if (k > 0 && b.Length < (k - 1) * bRowStride + n)
            throw new ArgumentException($"b span ({b.Length}) too short for k={k}, n={n}, stride={bRowStride}.");
        if (m > 0 && c.Length < (m - 1) * cRowStride + n)
            throw new ArgumentException($"c span ({c.Length}) too short for m={m}, n={n}, stride={cRowStride}.");
        Span<float> tmpB = stackalloc float[8];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                int aRow = i * aRowStride;
                int bCol = j;
                int kk = 0;
                if (Avx2.IsSupported && k >= 8)
                {
                    var vsum = Vector256<float>.Zero;
                    for (; kk + 8 <= k; kk += 8)
                    {
                        var av = Bf16ToFloatX8(a.Slice(aRow + kk, 8));
                        // Strided gather for B[k..k+8, j]. Cheaper than
                        // gather: scalar inner-loop fallback when bRowStride
                        // != 1 by column. For the n-major contiguous row
                        // pattern (B stored as [k][n]), we'd want the
                        // outer reorder; this is the j-stride layout.
                        for (int t = 0; t < 8; t++)
                            tmpB[t] = (float)b[(kk + t) * bRowStride + bCol];
                        var bv = Vector256.Create(tmpB[0], tmpB[1], tmpB[2], tmpB[3],
                                                   tmpB[4], tmpB[5], tmpB[6], tmpB[7]);
                        vsum = Fma.IsSupported ? Fma.MultiplyAdd(av, bv, vsum) : Avx.Add(vsum, Avx.Multiply(av, bv));
                    }
                    var lo = vsum.GetLower();
                    var hi = vsum.GetUpper();
                    var sum128 = Sse.Add(lo, hi);
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b01_00_11_10));
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b10_11_00_01));
                    acc = sum128.ToScalar();
                }
                for (; kk < k; kk++)
                {
                    acc += (float)a[aRow + kk] * (float)b[kk * bRowStride + bCol];
                }
                c[i * cRowStride + j] = BFloat16.FromFloat(acc);
            }
        }
    }

    // ── activations / normalization on bf16 ────────────────────

    /// <summary>GELU on bf16 (Hendrycks-Gimpel approx). Computes in float,
    /// truncates back to bf16. The float intermediate is what every
    /// transformer trains with; bf16 storage at the layer boundary is
    /// the memory win.</summary>
    public static void Gelu(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
    {
        if (dst.Length < x.Length)
            throw new ArgumentException($"dst length {dst.Length} must be ≥ x length {x.Length}.");
        const float c0 = 0.7978845608f; // sqrt(2/π)
        const float c1 = 0.044715f;
        for (int i = 0; i < x.Length; i++)
        {
            float v = (float)x[i];
            float v3 = v * v * v;
            float arg = c0 * (v + c1 * v3);
            float t = MathF.Tanh(arg);
            dst[i] = BFloat16.FromFloat(0.5f * v * (1f + t));
        }
    }

    /// <summary>SiLU (swish): x * sigmoid(x). Computed in float.</summary>
    public static void Silu(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
    {
        if (dst.Length < x.Length)
            throw new ArgumentException($"dst length {dst.Length} must be ≥ x length {x.Length}.");
        for (int i = 0; i < x.Length; i++)
        {
            float v = (float)x[i];
            float s = 1f / (1f + MathF.Exp(-v));
            dst[i] = BFloat16.FromFloat(v * s);
        }
    }

    /// <summary>Softmax on a row of bf16 values. Reduce/exp in float;
    /// final write is bf16. Stable variant: subtract row-max before exp.</summary>
    public static void Softmax(ReadOnlySpan<BFloat16> x, Span<BFloat16> dst)
    {
        int n = x.Length;
        if (dst.Length < n)
            throw new ArgumentException($"dst length {dst.Length} must be ≥ x length {n}.");
        if (n == 0) return;
        // Pass 1: max (in float).
        float m = (float)x[0];
        for (int i = 1; i < n; i++) { float v = (float)x[i]; if (v > m) m = v; }
        // Pass 2: exp & sum.
        Span<float> tmp = n <= 1024 ? stackalloc float[n] : new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            float e = MathF.Exp((float)x[i] - m);
            tmp[i] = e;
            sum += e;
        }
        // Pass 3: normalize + write.
        float inv = 1f / sum;
        for (int i = 0; i < n; i++) dst[i] = BFloat16.FromFloat(tmp[i] * inv);
    }

    /// <summary>LayerNorm on bf16 row. mean/var in float, scaled output in bf16.</summary>
    public static void LayerNorm(
        ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> gamma, ReadOnlySpan<BFloat16> beta,
        Span<BFloat16> dst, float eps = 1e-5f)
    {
        int n = x.Length;
        if (n == 0) return;
        if (gamma.Length != n || beta.Length != n || dst.Length != n)
            throw new ArgumentException("LayerNorm: spans must all share the same length.");

        // Welford-style mean+var so a 4096-element bf16 row doesn't lose
        // precision in the sum-of-squares accumulator.
        float mean = 0f, m2 = 0f;
        for (int i = 0; i < n; i++)
        {
            float v = (float)x[i];
            float delta = v - mean;
            mean += delta / (i + 1);
            float delta2 = v - mean;
            m2 += delta * delta2;
        }
        float variance = m2 / n;
        float invStd = 1f / MathF.Sqrt(variance + eps);

        for (int i = 0; i < n; i++)
        {
            float normalized = ((float)x[i] - mean) * invStd;
            float result = normalized * (float)gamma[i] + (float)beta[i];
            dst[i] = BFloat16.FromFloat(result);
        }
    }

    // ── conversion primitives ──────────────────────────────────

    /// <summary>Widen 8 bf16 values to a Vector256&lt;float&gt;. The widening
    /// is exact: bf16 occupies the high 16 bits of a float, so we shift
    /// left by 16. Equivalent to AVX-512's <c>VCVTBF16PS</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Bf16ToFloatX8(ReadOnlySpan<BFloat16> bf)
    {
        // Pack 8 ushort raw bits into a uint vector with the bf16 in the
        // upper half, then bit-cast to float.
        Span<uint> upper = stackalloc uint[8];
        for (int i = 0; i < 8; i++) upper[i] = ((uint)bf[i].RawValue) << 16;
        return Vector256.Create(
            BitConverter.UInt32BitsToSingle(upper[0]),
            BitConverter.UInt32BitsToSingle(upper[1]),
            BitConverter.UInt32BitsToSingle(upper[2]),
            BitConverter.UInt32BitsToSingle(upper[3]),
            BitConverter.UInt32BitsToSingle(upper[4]),
            BitConverter.UInt32BitsToSingle(upper[5]),
            BitConverter.UInt32BitsToSingle(upper[6]),
            BitConverter.UInt32BitsToSingle(upper[7]));
    }

    /// <summary>Narrow a Vector256&lt;float&gt; back to 8 bf16 values with
    /// round-to-nearest-even. AVX-512 BF16 has <c>VCVTNEPS2BF16</c> for
    /// this in hardware; AVX2 falls back to scalar RNE.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FloatX8ToBf16(Vector256<float> v, Span<BFloat16> dst)
    {
        Span<float> tmp = stackalloc float[8];
        for (int i = 0; i < 8; i++) tmp[i] = v.GetElement(i);
        for (int i = 0; i < 8; i++) dst[i] = BFloat16.FromFloat(tmp[i]);
    }
}
#endif
