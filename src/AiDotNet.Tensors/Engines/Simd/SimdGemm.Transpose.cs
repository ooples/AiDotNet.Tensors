using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>
    /// dst[dOff + c * dstStride + r] = src[sOff + r * srcStride + c] for r &lt; <paramref name="rows"/>,
    /// c &lt; <paramref name="cols"/>: a strided out-of-place transpose, 8x8 tiles through AVX shuffles where available.
    /// Pure data movement, so the result is bit-identical to the scalar copy. The small-K GEMM routes transpose an
    /// operand per call (dW's split-k chunks, dX's Wᵀ) and the attention kernels a key/value tile per head; as scalar
    /// strided copies those were ~30% of the serial attention forward at head dim 16 (Ryzen 7 4800H).
    /// </summary>
    internal static void Transpose(
        float[] src, int sOff, int srcStride, float[] dst, int dOff, int dstStride, int rows, int cols) =>
        Transpose(src, sOff, srcStride, dst, dOff, dstStride, rows, cols, scale: 1f, scaled: false);

    /// <summary>
    /// <see cref="Transpose(float[], int, int, float[], int, int, int, int)"/> with every element multiplied by
    /// <paramref name="scale"/> on the way (one float multiply per element, in every path): the attention kernels fold
    /// the softmax scale into the key tile they transpose anyway, instead of a separate scaled copy of the query rows.
    /// </summary>
    internal static void TransposeScaled(
        float[] src, int sOff, int srcStride, float[] dst, int dOff, int dstStride, int rows, int cols, float scale) =>
        Transpose(src, sOff, srcStride, dst, dOff, dstStride, rows, cols, scale, scaled: true);

    private static unsafe void Transpose(
        float[] src, int sOff, int srcStride, float[] dst, int dOff, int dstStride, int rows, int cols,
        float scale, bool scaled)
    {
        if (rows <= 0 || cols <= 0) return;
        // The tiles below address raw pointers, so the whole extent is checked once, for every path alike.
        if (sOff < 0 || dOff < 0
            || (long)sOff + (long)(rows - 1) * srcStride + cols > src.Length
            || (long)dOff + (long)(cols - 1) * dstStride + rows > dst.Length)
            throw new ArgumentOutOfRangeException(nameof(rows), "Transpose extent exceeds an operand.");
        int r8 = 0, c8 = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && rows >= 8 && cols >= 8)
        {
            r8 = rows & ~7;
            c8 = cols & ~7;
            fixed (float* ps = src, pd = dst)
            {
                for (int r = 0; r < r8; r += 8)
                    for (int c = 0; c < c8; c += 8)
                        Transpose8x8Avx(ps + sOff + (long)r * srcStride + c, srcStride, pd + dOff + (long)c * dstStride + r, dstStride,
                            scale, scaled);
            }
        }
#endif
        // Remainders: the columns right of the last full tile for the tiled rows, then every column of the rest.
        for (int r = 0; r < r8; r++)
        {
            int s = sOff + r * srcStride;
            for (int c = c8; c < cols; c++) dst[dOff + c * dstStride + r] = scaled ? src[s + c] * scale : src[s + c];
        }
        for (int r = r8; r < rows; r++)
        {
            int s = sOff + r * srcStride;
            for (int c = 0; c < cols; c++) dst[dOff + c * dstStride + r] = scaled ? src[s + c] * scale : src[s + c];
        }
    }

#if NET5_0_OR_GREATER
    /// <summary>dst[k * dstStride + i] = src[i * srcStride + k] (times <paramref name="scale"/> when
    /// <paramref name="scaled"/>) for an 8x8 tile (unpack / shuffle / lane permute).</summary>
    private static unsafe void Transpose8x8Avx(float* src, int srcStride, float* dst, int dstStride, float scale, bool scaled)
    {
        var r0 = Avx.LoadVector256(src); var r1 = Avx.LoadVector256(src + srcStride);
        var r2 = Avx.LoadVector256(src + 2 * srcStride); var r3 = Avx.LoadVector256(src + 3 * srcStride);
        var r4 = Avx.LoadVector256(src + 4 * srcStride); var r5 = Avx.LoadVector256(src + 5 * srcStride);
        var r6 = Avx.LoadVector256(src + 6 * srcStride); var r7 = Avx.LoadVector256(src + 7 * srcStride);
        if (scaled)
        {
            var vs = System.Runtime.Intrinsics.Vector256.Create(scale);
            r0 = Avx.Multiply(r0, vs); r1 = Avx.Multiply(r1, vs); r2 = Avx.Multiply(r2, vs); r3 = Avx.Multiply(r3, vs);
            r4 = Avx.Multiply(r4, vs); r5 = Avx.Multiply(r5, vs); r6 = Avx.Multiply(r6, vs); r7 = Avx.Multiply(r7, vs);
        }
        var t0 = Avx.UnpackLow(r0, r1); var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3); var t3 = Avx.UnpackHigh(r2, r3);
        var t4 = Avx.UnpackLow(r4, r5); var t5 = Avx.UnpackHigh(r4, r5);
        var t6 = Avx.UnpackLow(r6, r7); var t7 = Avx.UnpackHigh(r6, r7);
        var u0 = Avx.Shuffle(t0, t2, 0x44); var u1 = Avx.Shuffle(t0, t2, 0xEE);
        var u2 = Avx.Shuffle(t1, t3, 0x44); var u3 = Avx.Shuffle(t1, t3, 0xEE);
        var u4 = Avx.Shuffle(t4, t6, 0x44); var u5 = Avx.Shuffle(t4, t6, 0xEE);
        var u6 = Avx.Shuffle(t5, t7, 0x44); var u7 = Avx.Shuffle(t5, t7, 0xEE);
        Avx.Store(dst, Avx.Permute2x128(u0, u4, 0x20));
        Avx.Store(dst + dstStride, Avx.Permute2x128(u1, u5, 0x20));
        Avx.Store(dst + 2 * dstStride, Avx.Permute2x128(u2, u6, 0x20));
        Avx.Store(dst + 3 * dstStride, Avx.Permute2x128(u3, u7, 0x20));
        Avx.Store(dst + 4 * dstStride, Avx.Permute2x128(u0, u4, 0x31));
        Avx.Store(dst + 5 * dstStride, Avx.Permute2x128(u1, u5, 0x31));
        Avx.Store(dst + 6 * dstStride, Avx.Permute2x128(u2, u6, 0x31));
        Avx.Store(dst + 7 * dstStride, Avx.Permute2x128(u3, u7, 0x31));
    }
#endif
}
