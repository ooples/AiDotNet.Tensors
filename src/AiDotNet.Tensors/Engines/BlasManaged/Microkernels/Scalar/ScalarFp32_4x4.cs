using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Scalar reference microkernel: 4×4 output tile, FP32.
/// Reads packed-A in [Kc × Mr=4] vpanel layout (Mr-contiguous within each k-slice;
/// SIMD-friendly multi-row loads per K-step) and packed-B in [Kc × Nr=4] layout
/// (Nr-contiguous within each k-slice). Accumulates over the K-loop and writes
/// C[0..Mr, 0..Nr] += packedA · packedB. Caller is responsible for zero-initializing
/// C before the first kernel call.
///
/// This kernel is the FP32 ground truth — AVX2, AVX-512, and Neon FP32 microkernels
/// assert their output against this scalar reference in unit tests. Used at runtime
/// on net471 and on any host without AVX2 support.
/// </summary>
internal static class ScalarFp32_4x4
{
    /// <summary>The row-tile width of this microkernel (output rows per invocation).</summary>
    internal const int Mr = 4;
    /// <summary>The column-tile width of this microkernel (output cols per invocation).</summary>
    internal const int Nr = 4;

    /// <summary>
    /// Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile, summing over kc K-steps.
    /// C must be pre-zeroed if a fresh result is desired; otherwise values accumulate into
    /// existing C entries. When kc is 0 the kernel reads + writes C unchanged (no-op).
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr] row-major (Mr-contiguous within each k).</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr] row-major (Nr-contiguous within each k).</param>
    /// <param name="c">Output buffer; the kernel reads + writes the C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C (cols of the full C matrix, ≥ Nr).</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc)
    {
        // Accumulate in FP64 to keep summation error O(eps_fp64 * K) ≈ 1e-14
        // for K=64 — well below FP32 epsilon. Without this, sequential FP32
        // accumulation drifts by O(eps_fp32 * K) ≈ 8e-6 relative which can
        // exceed parity-check tolerances against OpenBLAS's tree-reduced
        // SIMD path even when both are "correct". Cast back to FP32 only at
        // write-back. This is the production fix for the routing-shim drift
        // failure surfaced on net471 (no AVX2 intrinsics → scalar path).
        double c00 = c[0 * ldc + 0], c01 = c[0 * ldc + 1], c02 = c[0 * ldc + 2], c03 = c[0 * ldc + 3];
        double c10 = c[1 * ldc + 0], c11 = c[1 * ldc + 1], c12 = c[1 * ldc + 2], c13 = c[1 * ldc + 3];
        double c20 = c[2 * ldc + 0], c21 = c[2 * ldc + 1], c22 = c[2 * ldc + 2], c23 = c[2 * ldc + 3];
        double c30 = c[3 * ldc + 0], c31 = c[3 * ldc + 1], c32 = c[3 * ldc + 2], c33 = c[3 * ldc + 3];

        for (int k = 0; k < kc; k++)
        {
            double a0 = packedA[k * Mr + 0];
            double a1 = packedA[k * Mr + 1];
            double a2 = packedA[k * Mr + 2];
            double a3 = packedA[k * Mr + 3];

            double b0 = packedB[k * Nr + 0];
            double b1 = packedB[k * Nr + 1];
            double b2 = packedB[k * Nr + 2];
            double b3 = packedB[k * Nr + 3];

            c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
            c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
            c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
            c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;
        }

        c[0 * ldc + 0] = (float)c00; c[0 * ldc + 1] = (float)c01; c[0 * ldc + 2] = (float)c02; c[0 * ldc + 3] = (float)c03;
        c[1 * ldc + 0] = (float)c10; c[1 * ldc + 1] = (float)c11; c[1 * ldc + 2] = (float)c12; c[1 * ldc + 3] = (float)c13;
        c[2 * ldc + 0] = (float)c20; c[2 * ldc + 1] = (float)c21; c[2 * ldc + 2] = (float)c22; c[2 * ldc + 3] = (float)c23;
        c[3 * ldc + 0] = (float)c30; c[3 * ldc + 1] = (float)c31; c[3 * ldc + 2] = (float)c32; c[3 * ldc + 3] = (float)c33;
    }

    /// <summary>
    /// Variant of <see cref="Run"/> that reads B directly from caller-supplied
    /// memory with stride <paramref name="ldb"/> instead of from a packed B
    /// stripe. Used by the PackAOnly strategy when packing B is not worthwhile.
    ///
    /// This variant supports only transB=false: B must be laid out as [K, N]
    /// row-major. For transB=true B, callers must pre-transpose or use a
    /// different strategy. The caller passes a slice of B positioned at the
    /// (pc, jc) corner of the current panel; the kernel reads
    /// b[k * ldb + col] for col in [0, Nr).
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr] row-major.</param>
    /// <param name="b">Source B buffer. Caller passes a slice positioned at the (pc, jc) corner of the current panel.</param>
    /// <param name="ldb">Leading dimension of B (cols of full B for transB=false).</param>
    /// <param name="c">Output buffer; reads + writes C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static void RunStridedB(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> b,
        int ldb,
        Span<float> c,
        int ldc,
        int kc)
    {
        // Same FP64-accumulator strategy as Run — see that method's comment.
        double c00 = c[0 * ldc + 0], c01 = c[0 * ldc + 1], c02 = c[0 * ldc + 2], c03 = c[0 * ldc + 3];
        double c10 = c[1 * ldc + 0], c11 = c[1 * ldc + 1], c12 = c[1 * ldc + 2], c13 = c[1 * ldc + 3];
        double c20 = c[2 * ldc + 0], c21 = c[2 * ldc + 1], c22 = c[2 * ldc + 2], c23 = c[2 * ldc + 3];
        double c30 = c[3 * ldc + 0], c31 = c[3 * ldc + 1], c32 = c[3 * ldc + 2], c33 = c[3 * ldc + 3];

        for (int k = 0; k < kc; k++)
        {
            double a0 = packedA[k * Mr + 0];
            double a1 = packedA[k * Mr + 1];
            double a2 = packedA[k * Mr + 2];
            double a3 = packedA[k * Mr + 3];

            double b0 = b[k * ldb + 0];
            double b1 = b[k * ldb + 1];
            double b2 = b[k * ldb + 2];
            double b3 = b[k * ldb + 3];

            c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
            c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
            c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
            c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;
        }

        c[0 * ldc + 0] = (float)c00; c[0 * ldc + 1] = (float)c01; c[0 * ldc + 2] = (float)c02; c[0 * ldc + 3] = (float)c03;
        c[1 * ldc + 0] = (float)c10; c[1 * ldc + 1] = (float)c11; c[1 * ldc + 2] = (float)c12; c[1 * ldc + 3] = (float)c13;
        c[2 * ldc + 0] = (float)c20; c[2 * ldc + 1] = (float)c21; c[2 * ldc + 2] = (float)c22; c[2 * ldc + 3] = (float)c23;
        c[3 * ldc + 0] = (float)c30; c[3 * ldc + 1] = (float)c31; c[3 * ldc + 2] = (float)c32; c[3 * ldc + 3] = (float)c33;
    }
}
