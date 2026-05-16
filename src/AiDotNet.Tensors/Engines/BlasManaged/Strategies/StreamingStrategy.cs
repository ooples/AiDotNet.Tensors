using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Streaming strategy — no packing. Dispatches the scalar streaming microkernel
/// directly over the full (M, N, K) shape. Used by <see cref="BlasManaged.Gemm{T}"/>
/// for small K (typically &lt; 32) where the pack cost in <see cref="PackBothStrategy"/>
/// or <see cref="PackAOnlyStrategy"/> would exceed the GEMM compute time.
///
/// <para>
/// Phase B uses a single scalar microkernel parameterized by trans flags.
/// AVX2/AVX-512/Neon phases may split this into 4 trans-specialized kernels
/// where SIMD load patterns differ.
/// </para>
/// </summary>
internal static class StreamingStrategy
{
    /// <summary>
    /// Compute C += op(A) · op(B) with no packing. C is read-modify-write
    /// (caller is responsible for zeroing C before the first call).
    /// </summary>
    /// <typeparam name="T">Element type. Must be float or double.</typeparam>
    public static void Run<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k) where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            ScalarStreaming.RunFp64(
                MemoryMarshal.Cast<T, double>(a), lda, transA,
                MemoryMarshal.Cast<T, double>(b), ldb, transB,
                MemoryMarshal.Cast<T, double>(c), ldc,
                m, n, k);
        }
        else if (typeof(T) == typeof(float))
        {
            ScalarStreaming.RunFp32(
                MemoryMarshal.Cast<T, float>(a), lda, transA,
                MemoryMarshal.Cast<T, float>(b), ldb, transB,
                MemoryMarshal.Cast<T, float>(c), ldc,
                m, n, k);
        }
        else
        {
            throw new NotSupportedException($"StreamingStrategy does not support T={typeof(T).Name}.");
        }
    }
}
