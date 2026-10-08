using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Out-of-place 2D transpose (<c>dst[c, r] = src[r, c]</c>) of row-major buffers, using AVX register
/// transposes (4×4 for double, 8×8 for float) inside cache-sized tiles, parallelized over tile rows.
/// </summary>
/// <remarks>
/// An element-by-element transpose reads or writes with a stride of one full row, so every element
/// lands on a different cache line. Transposing a 4×4 (double) or 8×8 (float) block in registers
/// turns both sides into full-vector loads and stores, and the tiling keeps the source and
/// destination tiles resident in cache while a tile is processed.
/// </remarks>
internal static class SimdTranspose
{
    // 64×64 doubles is 32 KB per tile, so the source and destination tiles fit in L2 together.
    private const int Tile = 64;

    // Below this many elements a parallel dispatch costs more than it saves.
    private const long ParallelThreshold = 1L << 16;

    [MethodImpl(Hot)]
    public static unsafe void Transpose(ReadOnlySpan<double> src, Span<double> dst, int rows, int cols)
    {
        ValidateLengths(src.Length, dst.Length, rows, cols);
        fixed (double* s = src, d = dst)
        {
            nint sp = (nint)s, dp = (nint)d;
            int dstRowTiles = (cols + Tile - 1) / Tile;
            if ((long)rows * cols < ParallelThreshold || dstRowTiles == 1)
            {
                for (int t = 0; t < dstRowTiles; t++)
                    TransposeDstRowTile((double*)sp, (double*)dp, rows, cols, t * Tile);
                return;
            }

            CpuParallelSettings.ParallelForOrSerial(0, dstRowTiles, (long)rows * cols,
                t => TransposeDstRowTile((double*)sp, (double*)dp, rows, cols, t * Tile),
                deterministicSafe: true);
        }
    }

    [MethodImpl(Hot)]
    public static unsafe void Transpose(ReadOnlySpan<float> src, Span<float> dst, int rows, int cols)
    {
        ValidateLengths(src.Length, dst.Length, rows, cols);
        fixed (float* s = src, d = dst)
        {
            nint sp = (nint)s, dp = (nint)d;
            int dstRowTiles = (cols + Tile - 1) / Tile;
            if ((long)rows * cols < ParallelThreshold || dstRowTiles == 1)
            {
                for (int t = 0; t < dstRowTiles; t++)
                    TransposeDstRowTile((float*)sp, (float*)dp, rows, cols, t * Tile);
                return;
            }

            CpuParallelSettings.ParallelForOrSerial(0, dstRowTiles, (long)rows * cols,
                t => TransposeDstRowTile((float*)sp, (float*)dp, rows, cols, t * Tile),
                deterministicSafe: true);
        }
    }

    private static void ValidateLengths(int srcLength, int dstLength, int rows, int cols)
    {
        long total = (long)rows * cols;
        if (rows < 0 || cols < 0 || srcLength < total || dstLength < total)
            throw new ArgumentException("Source and destination must each hold at least rows * cols elements.");
    }

    /// <summary>
    /// Writes destination rows <c>[j0, j0 + Tile)</c> (source columns) in full. Each parallel worker
    /// therefore owns a contiguous slice of the destination, so the first-touch page faults on a
    /// freshly allocated result are taken on disjoint pages rather than by every worker on every page.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void TransposeDstRowTile(double* src, double* dst, int rows, int cols, int j0)
    {
        int jEnd = Math.Min(j0 + Tile, cols);
        for (int i0 = 0; i0 < rows; i0 += Tile)
        {
            int iEnd = Math.Min(i0 + Tile, rows);
            int i = i0;
            if (Avx.IsSupported)
            {
                for (; i + 4 <= iEnd; i += 4)
                {
                    int j = j0;
                    for (; j + 4 <= jEnd; j += 4)
                        Transpose4x4(src + (long)i * cols + j, cols, dst + (long)j * rows + i, rows);
                    for (; j < jEnd; j++)
                        for (int ii = i; ii < i + 4; ii++)
                            dst[(long)j * rows + ii] = src[(long)ii * cols + j];
                }
            }

            for (; i < iEnd; i++)
                for (int j = j0; j < jEnd; j++)
                    dst[(long)j * rows + i] = src[(long)i * cols + j];
        }
    }

    /// <summary>
    /// Writes destination rows <c>[j0, j0 + Tile)</c> (source columns) in full. Each parallel worker
    /// therefore owns a contiguous slice of the destination, so the first-touch page faults on a
    /// freshly allocated result are taken on disjoint pages rather than by every worker on every page.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void TransposeDstRowTile(float* src, float* dst, int rows, int cols, int j0)
    {
        int jEnd = Math.Min(j0 + Tile, cols);
        for (int i0 = 0; i0 < rows; i0 += Tile)
        {
            int iEnd = Math.Min(i0 + Tile, rows);
            int i = i0;
            if (Avx.IsSupported)
            {
                for (; i + 8 <= iEnd; i += 8)
                {
                    int j = j0;
                    for (; j + 8 <= jEnd; j += 8)
                        Transpose8x8(src + (long)i * cols + j, cols, dst + (long)j * rows + i, rows);
                    for (; j < jEnd; j++)
                        for (int ii = i; ii < i + 8; ii++)
                            dst[(long)j * rows + ii] = src[(long)ii * cols + j];
                }
            }

            for (; i < iEnd; i++)
                for (int j = j0; j < jEnd; j++)
                    dst[(long)j * rows + i] = src[(long)i * cols + j];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Transpose4x4(double* s, int sStride, double* d, int dStride)
    {
        var r0 = Avx.LoadVector256(s);
        var r1 = Avx.LoadVector256(s + sStride);
        var r2 = Avx.LoadVector256(s + 2L * sStride);
        var r3 = Avx.LoadVector256(s + 3L * sStride);

        // Per 128-bit lane: t0 = [r0[0] r1[0] | r0[2] r1[2]], t1 = [r0[1] r1[1] | r0[3] r1[3]].
        var t0 = Avx.UnpackLow(r0, r1);
        var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3);
        var t3 = Avx.UnpackHigh(r2, r3);

        // Joining low lanes gives columns 0/1, high lanes give columns 2/3.
        Avx.Store(d, Avx.Permute2x128(t0, t2, 0x20));
        Avx.Store(d + dStride, Avx.Permute2x128(t1, t3, 0x20));
        Avx.Store(d + 2L * dStride, Avx.Permute2x128(t0, t2, 0x31));
        Avx.Store(d + 3L * dStride, Avx.Permute2x128(t1, t3, 0x31));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Transpose8x8(float* s, int sStride, float* d, int dStride)
    {
        var r0 = Avx.LoadVector256(s);
        var r1 = Avx.LoadVector256(s + sStride);
        var r2 = Avx.LoadVector256(s + 2L * sStride);
        var r3 = Avx.LoadVector256(s + 3L * sStride);
        var r4 = Avx.LoadVector256(s + 4L * sStride);
        var r5 = Avx.LoadVector256(s + 5L * sStride);
        var r6 = Avx.LoadVector256(s + 6L * sStride);
        var r7 = Avx.LoadVector256(s + 7L * sStride);

        // Interleave row pairs: t0 = [a0 b0 a1 b1 | a4 b4 a5 b5], t1 = [a2 b2 a3 b3 | a6 b6 a7 b7].
        var t0 = Avx.UnpackLow(r0, r1);
        var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3);
        var t3 = Avx.UnpackHigh(r2, r3);
        var t4 = Avx.UnpackLow(r4, r5);
        var t5 = Avx.UnpackHigh(r4, r5);
        var t6 = Avx.UnpackLow(r6, r7);
        var t7 = Avx.UnpackHigh(r6, r7);

        // Gather four rows per column: u0 = [a0 b0 c0 d0 | a4 b4 c4 d4], u1 = [a1 b1 c1 d1 | a5 b5 c5 d5], ...
        var u0 = Avx.Shuffle(t0, t2, 0x44);
        var u1 = Avx.Shuffle(t0, t2, 0xEE);
        var u2 = Avx.Shuffle(t1, t3, 0x44);
        var u3 = Avx.Shuffle(t1, t3, 0xEE);
        var u4 = Avx.Shuffle(t4, t6, 0x44);
        var u5 = Avx.Shuffle(t4, t6, 0xEE);
        var u6 = Avx.Shuffle(t5, t7, 0x44);
        var u7 = Avx.Shuffle(t5, t7, 0xEE);

        // Join the top-four and bottom-four rows: low lanes are columns 0-3, high lanes columns 4-7.
        Avx.Store(d, Avx.Permute2x128(u0, u4, 0x20));
        Avx.Store(d + dStride, Avx.Permute2x128(u1, u5, 0x20));
        Avx.Store(d + 2L * dStride, Avx.Permute2x128(u2, u6, 0x20));
        Avx.Store(d + 3L * dStride, Avx.Permute2x128(u3, u7, 0x20));
        Avx.Store(d + 4L * dStride, Avx.Permute2x128(u0, u4, 0x31));
        Avx.Store(d + 5L * dStride, Avx.Permute2x128(u1, u5, 0x31));
        Avx.Store(d + 6L * dStride, Avx.Permute2x128(u2, u6, 0x31));
        Avx.Store(d + 7L * dStride, Avx.Permute2x128(u3, u7, 0x31));
    }
}
#endif
