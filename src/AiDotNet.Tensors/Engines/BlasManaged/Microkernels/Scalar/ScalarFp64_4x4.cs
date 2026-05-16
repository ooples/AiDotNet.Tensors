using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Scalar reference microkernel: 4×4 output tile, FP64.
/// Reads packed-A in [Mr=4, Kc] layout (K-contiguous within each row) and
/// packed-B in [Kc, Nr=4] layout (col-contiguous within each k). Accumulates
/// over the K-loop and writes C[0..4, 0..4] += packedA · packedB. Caller is
/// responsible for zero-initializing C before the first kernel call.
///
/// This kernel is the ground truth — AVX2, AVX-512, and Neon microkernels
/// assert their output against this scalar reference in unit tests. Used at
/// runtime on net471 and on any host without AVX2 support.
/// </summary>
internal static class ScalarFp64_4x4
{
    /// <summary>The row-tile width of this microkernel (output rows per invocation).</summary>
    public const int Mr = 4;
    /// <summary>The column-tile width of this microkernel (output cols per invocation).</summary>
    public const int Nr = 4;

    /// <summary>
    /// Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile, summing over
    /// kc K-steps. C must be pre-zeroed if a fresh result is desired; otherwise
    /// values accumulate into existing C entries.
    /// </summary>
    /// <param name="packedA">Packed-A stripe, layout [Mr × Kc] row-major.</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr] row-major.</param>
    /// <param name="c">Output buffer; the kernel reads + writes the C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C (cols of the full C matrix, ≥ Nr).</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static void Run(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc)
    {
        // Hold all 16 C-tile cells in registers across the K-loop. The JIT will
        // keep them on the stack on net471 (no Vector<T>); on net10.0 the
        // RyuJIT's register allocator may promote them. Either way, this is
        // the ground-truth reference and not a perf-critical path.
        double c00 = c[0 * ldc + 0], c01 = c[0 * ldc + 1], c02 = c[0 * ldc + 2], c03 = c[0 * ldc + 3];
        double c10 = c[1 * ldc + 0], c11 = c[1 * ldc + 1], c12 = c[1 * ldc + 2], c13 = c[1 * ldc + 3];
        double c20 = c[2 * ldc + 0], c21 = c[2 * ldc + 1], c22 = c[2 * ldc + 2], c23 = c[2 * ldc + 3];
        double c30 = c[3 * ldc + 0], c31 = c[3 * ldc + 1], c32 = c[3 * ldc + 2], c33 = c[3 * ldc + 3];

        for (int k = 0; k < kc; k++)
        {
            double a0 = packedA[0 * kc + k];
            double a1 = packedA[1 * kc + k];
            double a2 = packedA[2 * kc + k];
            double a3 = packedA[3 * kc + k];

            double b0 = packedB[k * Nr + 0];
            double b1 = packedB[k * Nr + 1];
            double b2 = packedB[k * Nr + 2];
            double b3 = packedB[k * Nr + 3];

            c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
            c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
            c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
            c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;
        }

        c[0 * ldc + 0] = c00; c[0 * ldc + 1] = c01; c[0 * ldc + 2] = c02; c[0 * ldc + 3] = c03;
        c[1 * ldc + 0] = c10; c[1 * ldc + 1] = c11; c[1 * ldc + 2] = c12; c[1 * ldc + 3] = c13;
        c[2 * ldc + 0] = c20; c[2 * ldc + 1] = c21; c[2 * ldc + 2] = c22; c[2 * ldc + 3] = c23;
        c[3 * ldc + 0] = c30; c[3 * ldc + 1] = c31; c[3 * ldc + 2] = c32; c[3 * ldc + 3] = c33;
    }
}
