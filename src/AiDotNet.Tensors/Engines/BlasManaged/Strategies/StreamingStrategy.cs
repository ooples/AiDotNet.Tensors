using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Streaming strategy — no packing. Dispatches the streaming microkernel
/// directly over the full (M, N, K) shape. Used by <see cref="BlasManaged.Gemm{T}"/>
/// for small K (typically &lt; 32) where the pack cost in <see cref="PackBothStrategy"/>
/// or <see cref="PackAOnlyStrategy"/> would exceed the GEMM compute time.
///
/// <para>
/// Routes to <see cref="Avx512Streaming"/> when AVX-512 is available,
/// then <see cref="Avx2Streaming"/> when AVX2 + FMA are available,
/// then <see cref="NeonStreaming"/> on ARM64 hosts,
/// otherwise falls back to the scalar reference kernel.
/// </para>
///
/// <para>
/// <b>Sub-issue B (#370) task B.2:</b> when <see cref="AxisSelector"/> picks
/// <see cref="ParallelismAxis.N"/>, the dispatcher partitions N across threads
/// and each thread runs the streaming kernel on its own column slice of C.
/// Disjoint writes — no synchronization needed; bit-exact across thread counts.
/// </para>
/// </summary>
internal static class StreamingStrategy
{
    /// <summary>
    /// Microkernel column-tile width used for AxisSelector threshold computation.
    /// Conservative value matching the AVX2 streaming kernel's vector width
    /// (8 floats / 4 doubles in a 256-bit register).
    /// </summary>
    private const int StreamingNr = 8;
    private const int StreamingMr = 8;

    /// <summary>
    /// Compute C += op(A) · op(B) with no packing. C is read-modify-write
    /// (caller is responsible for zeroing C before the first call).
    /// </summary>
    public static void Run<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        int procs = options.NumThreads > 0 ? options.NumThreads : Environment.ProcessorCount;
        // -1 from caller = force single-thread (deterministic regression-test path).
        if (options.NumThreads < 0) procs = 1;
        bool isDeterministic = BlasProvider.IsDeterministicMode;

        var axis = AxisSelector.Select(m, n, k, StreamingMr, StreamingNr, procs, isDeterministic);

        if (axis == ParallelismAxis.N && n >= procs * StreamingNr * 2)
        {
            RunNParallel(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, procs);
            return;
        }

        RunSerial(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
    }

    /// <summary>
    /// Partition N across <paramref name="procs"/> threads. Each thread writes
    /// a disjoint column slice of C, so no synchronization is needed and the
    /// output is bit-exact identical to the serial result.
    /// </summary>
    private static void RunNParallel<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        int procs) where T : unmanaged
    {
        // Pin a, b, c so worker threads can capture raw pointers across the
        // parallel boundary (Span<T> can't cross the Parallel.For lambda).
        unsafe
        {
            fixed (T* aPtr = a)
            fixed (T* bPtr = b)
            fixed (T* cPtr = c)
            {
                // Capture locals so the lambda can close over them without
                // accessing the ReadOnlySpan/Span (illegal across the lambda).
                T* aLocal = aPtr;
                T* bLocal = bPtr;
                T* cLocal = cPtr;
                int aLen = a.Length, bLen = b.Length, cLen = c.Length;

                int procsLocal = procs;
                int nLocal = n;
                int mLocal = m;
                int kLocal = k;
                int ldaLocal = lda, ldbLocal = ldb, ldcLocal = ldc;
                bool taLocal = transA, tbLocal = transB;

                Parallel.For(0, procsLocal, p =>
                {
                    int nStart = (int)(((long)p * nLocal) / procsLocal);
                    int nEnd = (int)(((long)(p + 1) * nLocal) / procsLocal);
                    int nChunk = nEnd - nStart;
                    if (nChunk <= 0) return;

                    // B slice along N: depends on transB.
                    //   transB=false: B[K, N] row-major. Column nStart starts at b[nStart];
                    //                 stride between rows is ldb (unchanged).
                    //   transB=true:  B[N, K] row-major. Row nStart starts at b[nStart*ldb];
                    //                 stride between rows is ldb (unchanged); kernel sees
                    //                 a [nChunk, K] sub-block.
                    int bOffset = tbLocal ? nStart * ldbLocal : nStart;
                    int bSliceLen = bLen - bOffset;

                    // C slice along N: row-major, column nStart starts at c[nStart],
                    // row stride is ldc (unchanged).
                    int cOffset = nStart;
                    int cSliceLen = cLen - cOffset;

                    var aSpan = new ReadOnlySpan<T>(aLocal, aLen);
                    var bSpan = new ReadOnlySpan<T>(bLocal + bOffset, bSliceLen);
                    var cSpan = new Span<T>(cLocal + cOffset, cSliceLen);

                    RunSerial(aSpan, ldaLocal, taLocal,
                              bSpan, ldbLocal, tbLocal,
                              cSpan, ldcLocal,
                              mLocal, nChunk, kLocal);
                });
            }
        }
    }

    /// <summary>
    /// Serial microkernel dispatch: AVX-512 → AVX2 → Neon → scalar.
    /// </summary>
    private static void RunSerial<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k) where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            if (Avx512Streaming.IsSupported)
            {
                Avx512Streaming.RunFp64(
                    MemoryMarshal.Cast<T, double>(a), lda, transA,
                    MemoryMarshal.Cast<T, double>(b), ldb, transB,
                    MemoryMarshal.Cast<T, double>(c), ldc,
                    m, n, k);
                return;
            }
            if (Avx2Streaming.IsSupported)
            {
                Avx2Streaming.RunFp64(
                    MemoryMarshal.Cast<T, double>(a), lda, transA,
                    MemoryMarshal.Cast<T, double>(b), ldb, transB,
                    MemoryMarshal.Cast<T, double>(c), ldc,
                    m, n, k);
                return;
            }
            if (NeonStreaming.IsSupported)
            {
                NeonStreaming.RunFp64(
                    MemoryMarshal.Cast<T, double>(a), lda, transA,
                    MemoryMarshal.Cast<T, double>(b), ldb, transB,
                    MemoryMarshal.Cast<T, double>(c), ldc,
                    m, n, k);
                return;
            }
            ScalarStreaming.RunFp64(
                MemoryMarshal.Cast<T, double>(a), lda, transA,
                MemoryMarshal.Cast<T, double>(b), ldb, transB,
                MemoryMarshal.Cast<T, double>(c), ldc,
                m, n, k);
            return;
        }
        if (typeof(T) == typeof(float))
        {
            if (Avx512Streaming.IsSupported)
            {
                Avx512Streaming.RunFp32(
                    MemoryMarshal.Cast<T, float>(a), lda, transA,
                    MemoryMarshal.Cast<T, float>(b), ldb, transB,
                    MemoryMarshal.Cast<T, float>(c), ldc,
                    m, n, k);
                return;
            }
            if (Avx2Streaming.IsSupported)
            {
                Avx2Streaming.RunFp32(
                    MemoryMarshal.Cast<T, float>(a), lda, transA,
                    MemoryMarshal.Cast<T, float>(b), ldb, transB,
                    MemoryMarshal.Cast<T, float>(c), ldc,
                    m, n, k);
                return;
            }
            if (NeonStreaming.IsSupported)
            {
                NeonStreaming.RunFp32(
                    MemoryMarshal.Cast<T, float>(a), lda, transA,
                    MemoryMarshal.Cast<T, float>(b), ldb, transB,
                    MemoryMarshal.Cast<T, float>(c), ldc,
                    m, n, k);
                return;
            }
            ScalarStreaming.RunFp32(
                MemoryMarshal.Cast<T, float>(a), lda, transA,
                MemoryMarshal.Cast<T, float>(b), ldb, transB,
                MemoryMarshal.Cast<T, float>(c), ldc,
                m, n, k);
            return;
        }
        throw new NotSupportedException($"StreamingStrategy does not support T={typeof(T).Name}.");
    }
}
