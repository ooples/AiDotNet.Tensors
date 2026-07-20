using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Block-Q8_0 matrix multiply — the quantized GEMM that matches llama.cpp / ggml
/// (<c>ggml_vec_dot_q8_0_q8_0</c>). Weights stay in Q8_0 (int8 + one fp32 scale per
/// 32-element block); the fp32 activation is quantized to Q8_0 rows on the fly and the
/// dot is done as int8×int8 → int32, scaled by the product of the two block scales.
///
/// <para>Layout (matches GGUF Q8_0 for a linear weight stored [out, in] = [N, K]):
/// <list type="bullet">
///   <item><b>weightQs</b>: <c>sbyte[N*K]</c>, row-major; each weight row n is K/32
///   contiguous blocks of 32 int8 values along the contraction dim K.</item>
///   <item><b>weightScales</b>: <c>float[N*(K/32)]</c>, one scale per weight block.</item>
///   <item><b>activations</b>: <c>float[M*K]</c>, row-major; quantized internally.</item>
///   <item><b>output</b>: <c>float[M*N]</c>; <c>C[m,n] = Σ_k act[m,k]·W[n,k]</c>.</item>
/// </list></para>
///
/// <para>K must be a multiple of 32 (the Q8_0 block size), which holds for every
/// transformer hidden/FFN dimension. The AVX2 path uses the exact ggml pipeline
/// (<c>_mm256_sign_epi8</c> + <c>_mm256_maddubs_epi16</c> + <c>_mm256_madd_epi16</c>);
/// net471 (no <c>System.Runtime.Intrinsics</c>) uses the scalar reference, which is the
/// bit-for-bit formula ggml's scalar fallback uses.</para>
/// </summary>
public static class Q8BlockGemm
{
    /// <summary>Q8_0 block size (values per scale).</summary>
    public const int QK = 32;

    /// <summary>True when K is a valid Q8_0 contraction dimension (multiple of 32).</summary>
    public static bool IsSupportedK(int k) => k > 0 && (k % QK) == 0;

    /// <summary>
    /// Quantizes a row-major [rows, K] float matrix into Q8_0 blocks: <paramref name="dstQs"/>
    /// (int8, same length) and <paramref name="dstScales"/> (float, rows*K/32). Group size 32,
    /// symmetric, round-half-away-from-zero — identical to GGUF Q8_0 / <c>quantize_row_q8_0</c>.
    /// </summary>
    public static void QuantizeRows(ReadOnlySpan<float> src, int rows, int k, Span<sbyte> dstQs, Span<float> dstScales)
    {
        if (!IsSupportedK(k)) throw new ArgumentException($"K ({k}) must be a positive multiple of {QK}.", nameof(k));
        if (src.Length != rows * k) throw new ArgumentException($"src length {src.Length} != rows*k {rows * k}.");
        if (dstQs.Length != rows * k) throw new ArgumentException($"dstQs length {dstQs.Length} != rows*k {rows * k}.");
        int bpr = k / QK;
        if (dstScales.Length != rows * bpr) throw new ArgumentException($"dstScales length {dstScales.Length} != rows*(k/32) {rows * bpr}.");

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < bpr; b++)
            {
                int off = r * k + b * QK;
                float maxAbs = 0f;
                for (int j = 0; j < QK; j++)
                {
                    float a = Math.Abs(src[off + j]);
                    if (a > maxAbs) maxAbs = a;
                }
                float scale = maxAbs / 127f;
                if (scale == 0f) scale = 1f;
                float inv = 1f / scale;
                dstScales[r * bpr + b] = scale;
                for (int j = 0; j < QK; j++)
                {
                    int q = (int)Math.Round(src[off + j] * inv, MidpointRounding.AwayFromZero);
                    if (q < -127) q = -127;
                    if (q > 127) q = 127;
                    dstQs[off + j] = (sbyte)q;
                }
            }
        }
    }

    /// <summary>
    /// C[M,N] = activations[M,K] · Wᵀ where W is Q8_0 blocks [N,K]. Activation is quantized to
    /// Q8_0 internally. Parallelized over M rows.
    /// </summary>
    public static void MatMul(
        ReadOnlySpan<float> activations,
        ReadOnlySpan<sbyte> weightQs,
        ReadOnlySpan<float> weightScales,
        Span<float> output,
        int m, int k, int n)
    {
        if (!IsSupportedK(k)) throw new ArgumentException($"K ({k}) must be a positive multiple of {QK}.", nameof(k));
        if (m < 0 || n < 0) throw new ArgumentException($"shapes must be non-negative; got m={m}, n={n}.");
        if (activations.Length != m * k) throw new ArgumentException($"activations length {activations.Length} != m*k {m * k}.");
        if (weightQs.Length != n * k) throw new ArgumentException($"weightQs length {weightQs.Length} != n*k {n * k}.");
        int bpr = k / QK;
        if (weightScales.Length != n * bpr) throw new ArgumentException($"weightScales length {weightScales.Length} != n*(k/32) {n * bpr}.");
        if (output.Length != m * n) throw new ArgumentException($"output length {output.Length} != m*n {m * n}.");
        if (m == 0 || n == 0) { output.Clear(); return; }

        // Quantize all activation rows up front (small vs the N-wide inner loop; reused across all n).
        var actQs = new sbyte[m * k];
        var actScales = new float[m * bpr];
        QuantizeRows(activations, m, k, actQs, actScales);

        // Copy to arrays the parallel body can pin (spans can't cross the lambda boundary).
        var wQs = weightQs.ToArray();
        var wSc = weightScales.ToArray();
        var outArr = new float[m * n];

        AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(
            0, m, (long)m * n * k, i => RowProduct(actQs, actScales, wQs, wSc, outArr, i, k, n, bpr));

        outArr.CopyTo(output);
    }

    private static void RowProduct(
        sbyte[] actQs, float[] actScales, sbyte[] wQs, float[] wSc, float[] outArr,
        int i, int k, int n, int bpr)
    {
        for (int j = 0; j < n; j++)
        {
            float acc = 0f;
            int aRow = i * k, aScaleRow = i * bpr;
            int wRow = j * k, wScaleRow = j * bpr;
            for (int b = 0; b < bpr; b++)
            {
                int di = DotBlock(actQs, aRow + b * QK, wQs, wRow + b * QK);
                acc += di * actScales[aScaleRow + b] * wSc[wScaleRow + b];
            }
            outArr[i * n + j] = acc;
        }
    }

#if NET5_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe int DotBlock(sbyte[] a, int aOff, sbyte[] b, int bOff)
    {
        if (Avx2.IsSupported)
        {
            fixed (sbyte* ap = a, bp = b)
            {
                var x = Avx.LoadVector256(ap + aOff);
                var y = Avx.LoadVector256(bp + bOff);
                var ax = Avx2.Sign(x, x).AsByte();               // |a| as unsigned magnitude
                var sy = Avx2.Sign(y, x);                        // apply a's sign to b
                var dot16 = Avx2.MultiplyAddAdjacent(ax, sy);    // maddubs -> int16 pairs
                var dot32 = Avx2.MultiplyAddAdjacent(dot16, Vector256.Create((short)1)); // madd -> int32
                var lo = dot32.GetLower();
                var hi = dot32.GetUpper();
                var s = Sse2.Add(lo, hi);
                s = Sse2.Add(s, Sse2.Shuffle(s, 0x4E));
                s = Sse2.Add(s, Sse2.Shuffle(s, 0xB1));
                return s.ToScalar();
            }
        }
        return DotBlockScalar(a, aOff, b, bOff);
    }
#else
    private static int DotBlock(sbyte[] a, int aOff, sbyte[] b, int bOff) => DotBlockScalar(a, aOff, b, bOff);
#endif

    private static int DotBlockScalar(sbyte[] a, int aOff, sbyte[] b, int bOff)
    {
        int sum = 0;
        for (int j = 0; j < QK; j++) sum += a[aOff + j] * b[bOff + j];
        return sum;
    }
}
