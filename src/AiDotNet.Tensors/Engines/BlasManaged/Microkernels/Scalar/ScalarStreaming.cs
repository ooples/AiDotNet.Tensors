using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Streaming scalar microkernel for FP64 and FP32. Reads A and B directly
/// in caller-supplied strides — no pack step. Used by
/// <see cref="StreamingStrategy"/> when K is small (typically &lt;32) and
/// pack overhead would exceed the GEMM compute time.
///
/// <para>
/// Handles all 4 (transA, transB) combinations via index branching. For
/// scalar code the branch cost is negligible; SIMD phases will split into
/// 4 distinct kernels where the load patterns differ materially.
/// </para>
/// </summary>
internal static class ScalarStreaming
{
    /// <summary>
    /// Compute C += op(A) · op(B) directly without packing. C is read-modify-write.
    /// </summary>
    public static void RunFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = c[i * ldc + j];
                for (int kk = 0; kk < k; kk++)
                {
                    double aval = transA ? a[kk * lda + i] : a[i * lda + kk];
                    double bval = transB ? b[j * ldb + kk] : b[kk * ldb + j];
                    sum += aval * bval;
                }
                c[i * ldc + j] = sum;
            }
        }
    }

    /// <summary>
    /// FP32 mirror of <see cref="RunFp64"/>. Accumulates in FP64 internally so
    /// the bit-exact contract with <see cref="ScalarFp32_4x4"/>'s output is
    /// preserved (both microkernels must produce the same C for the
    /// TinyShapeBypassTest "bypass matches forced full path" guarantee).
    /// FP64 internal accumulators also keep summation error to O(eps_fp64 · K),
    /// well below the routing-shim test's correctness bound.
    /// </summary>
    public static void RunFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc,
        int m, int n, int k)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = c[i * ldc + j];
                for (int kk = 0; kk < k; kk++)
                {
                    double aval = transA ? a[kk * lda + i] : a[i * lda + kk];
                    double bval = transB ? b[j * ldb + kk] : b[kk * ldb + j];
                    sum += aval * bval;
                }
                c[i * ldc + j] = (float)sum;
            }
        }
    }
}
