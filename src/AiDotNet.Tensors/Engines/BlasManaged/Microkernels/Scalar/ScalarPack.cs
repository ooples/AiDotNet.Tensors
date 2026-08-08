using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Scalar packing routines for A and B sides of a GEMM. Output vpanel layout
/// matches what <see cref="ScalarFp64_4x4"/> and <see cref="ScalarFp32_4x4"/>
/// read. Used at runtime on net471 and on any host without AVX2 support;
/// AVX2/AVX-512/Neon paths replace these with SIMD-vectorized equivalents in
/// later phases.
/// </summary>
internal static class ScalarPack
{
    /// <summary>
    /// Pack a logical Mc-row × Kc-col panel of A into BLIS vpanel layout
    /// <c>[Mc/Mr, Kc, Mr]</c> — linearized as
    /// <c>packed[stripe * Kc * Mr + k * Mr + row]</c>.
    ///
    /// <para>
    /// When <paramref name="transA"/> is <c>false</c>, A is stored row-major
    /// <c>[M, K]</c> with leading dimension <paramref name="lda"/>: the pack
    /// routine reads <c>a[logicalRow * lda + k]</c>.
    /// </para>
    /// <para>
    /// When <paramref name="transA"/> is <c>true</c>, A is stored row-major
    /// <c>[K, M]</c> with leading dimension <paramref name="lda"/>: the pack
    /// routine reads <c>a[k * lda + logicalRow]</c>. The transposition is
    /// absorbed by the pack — the microkernel reads packed-A as if A had
    /// never been transposed.
    /// </para>
    ///
    /// <para>
    /// This implementation handles <c>mc</c> exactly divisible by <c>mr</c>.
    /// Tail handling for <c>mc % mr != 0</c> is added in Phase G.
    /// </para>
    /// </summary>
    /// <param name="a">Source A buffer, length ≥ lda × (transA ? K : M).</param>
    /// <param name="lda">Leading dimension of A.</param>
    /// <param name="transA">True if A is stored as A^T (logical [M, K] view from [K, M] memory).</param>
    /// <param name="packed">Destination vpanel buffer, length ≥ mc × kc.</param>
    /// <param name="mc">Rows of A to pack (must be ≤ Mc panel size, exactly divisible by mr).</param>
    /// <param name="kc">Cols of A to pack (one Kc block).</param>
    /// <param name="mr">Microkernel row-tile width (e.g., 4 for ScalarFp64_4x4).</param>
    public static void PackA<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        Span<T> packed, int mc, int kc, int mr) where T : unmanaged
    {
        int numStripes = mc / mr;
        for (int stripe = 0; stripe < numStripes; stripe++)
        {
            int packedOff = stripe * kc * mr;
            int baseRow = stripe * mr;

            if (transA)
            {
                // A is [K, M], so consecutive logical ROWS are adjacent in memory — and the vpanel wants
                // them adjacent too. The whole inner row loop is therefore a CONTIGUOUS block move, which
                // CopyTo performs with the platform's vectorized memmove instead of mr scalar loads and
                // stores. Bit-exact by construction: packing moves values and never arithmetic on them.
                for (int k = 0; k < kc; k++)
                    a.Slice((k * lda) + baseRow, mr).CopyTo(packed.Slice(packedOff + (k * mr), mr));
            }
            else
            {
                // A is [M, K]: consecutive logical rows are lda apart, so this is a strided gather and
                // cannot become a block move. Left scalar deliberately — a register transpose would help
                // here and is a separate, riskier change.
                for (int k = 0; k < kc; k++)
                {
                    for (int row = 0; row < mr; row++)
                        packed[packedOff + (k * mr) + row] = a[((baseRow + row) * lda) + k];
                }
            }
        }
    }

    /// <summary>
    /// Pack a logical Kc-row × Nc-col panel of B into BLIS stripe layout
    /// <c>[ceil(Nc/Nr), Kc, Nr]</c> — linearized as
    /// <c>packed[stripe * Kc * Nr + k * Nr + col]</c>.
    ///
    /// <para>
    /// When <paramref name="transB"/> is <c>false</c>, B is stored row-major
    /// <c>[K, N]</c> with leading dimension <paramref name="ldb"/>: the pack
    /// routine reads <c>b[k * ldb + logicalCol]</c>.
    /// </para>
    /// <para>
    /// When <paramref name="transB"/> is <c>true</c>, B is stored row-major
    /// <c>[N, K]</c> with leading dimension <paramref name="ldb"/>: the pack
    /// routine reads <c>b[logicalCol * ldb + k]</c>. The transposition is
    /// absorbed by the pack — the microkernel reads packed-B as if B had
    /// never been transposed.
    /// </para>
    ///
    /// <para>
    /// When <c>nc % nr != 0</c>, the last (partial) stripe is packed with
    /// zero-padding in the unused lane positions so the packed buffer always
    /// uses full <c>Nr</c>-wide stripes. The caller must allocate
    /// <c>ceil(nc / nr) * nr * kc</c> elements in <paramref name="packed"/>.
    /// </para>
    /// </summary>
    /// <param name="b">Source B buffer, length ≥ ldb × (transB ? N : K).</param>
    /// <param name="ldb">Leading dimension of B.</param>
    /// <param name="transB">True if B is stored as B^T (logical [K, N] view from [N, K] memory).</param>
    /// <param name="packed">Destination stripe buffer, length ≥ ceil(nc / nr) × nr × kc.</param>
    /// <param name="nc">Cols of B to pack (must be ≤ Nc panel size).</param>
    /// <param name="kc">Rows of B to pack (one Kc block).</param>
    /// <param name="nr">Microkernel column-tile width (e.g., 4 for ScalarFp64_4x4).</param>
    public static void PackB<T>(
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> packed, int nc, int kc, int nr) where T : unmanaged
    {
        int numFullStripes = nc / nr;
        int tailCols = nc % nr;         // 0 when nc is divisible by nr

        // Pack full Nr-wide stripes.
        for (int stripe = 0; stripe < numFullStripes; stripe++)
        {
            int packedOff = stripe * kc * nr;
            int baseCol = stripe * nr;

            if (!transB)
            {
                // B is [K, N], so a stripe's nr columns are ADJACENT in memory and the packed stripe wants
                // them adjacent as well. This is the common case (untransposed B) and the entire inner
                // column loop collapses to a contiguous block move, which CopyTo does with a vectorized
                // memmove. Bit-exact: packing only moves values.
                for (int k = 0; k < kc; k++)
                    b.Slice((k * ldb) + baseCol, nr).CopyTo(packed.Slice(packedOff + (k * nr), nr));
            }
            else
            {
                // B is [N, K]: a stripe's columns are ldb apart, a strided gather. Left scalar.
                for (int k = 0; k < kc; k++)
                {
                    for (int col = 0; col < nr; col++)
                        packed[packedOff + (k * nr) + col] = b[((baseCol + col) * ldb) + k];
                }
            }
        }

        // Pack the partial tail stripe with zero-padding (Task G2).
        // The packed layout always uses Nr-wide rows so tail kernels can read
        // packedB[k * nr + col] safely for col in [0, effectiveNr).
        if (tailCols > 0)
        {
            int tailStripe = numFullStripes;
            int tailPackedOff = tailStripe * kc * nr;
            int tailBaseCol = tailStripe * nr;

            for (int k = 0; k < kc; k++)
            {
                var destination = packed.Slice(tailPackedOff + (k * nr), nr);

                if (!transB)
                {
                    // Contiguous for the real columns, then the padding lanes are CLEARED rather than left
                    // as whatever the buffer held. That zero-fill is load bearing, not tidiness: the packed
                    // layout is always nr wide so a tail kernel reads all nr lanes, and pooled buffers are
                    // not zeroed on rent — stale values in the padding would be multiplied into C.
                    b.Slice((k * ldb) + tailBaseCol, tailCols).CopyTo(destination);
                    destination.Slice(tailCols).Clear();
                }
                else
                {
                    for (int col = 0; col < tailCols; col++)
                        destination[col] = b[((tailBaseCol + col) * ldb) + k];
                    destination.Slice(tailCols).Clear();
                }
            }
        }
    }
}
