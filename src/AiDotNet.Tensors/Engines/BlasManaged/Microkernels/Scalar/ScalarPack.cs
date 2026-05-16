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
            for (int k = 0; k < kc; k++)
            {
                for (int row = 0; row < mr; row++)
                {
                    int logicalRow = stripe * mr + row;
                    T value = transA
                        ? a[k * lda + logicalRow]            // A stored [K, M], read M-stride
                        : a[logicalRow * lda + k];           // A stored [M, K], read K-stride
                    packed[packedOff + k * mr + row] = value;
                }
            }
        }
    }
}
