using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged.Pool;
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
        // Determinism comes from either the global BlasProvider switch or the per-call
        // BlasOptions.Mode. Any source asking for Deterministic wins (OR semantics).
        bool isDeterministic = BlasProvider.IsDeterministicMode || options.Mode == BlasMode.Deterministic;

        var axis = AxisSelector.Select(m, n, k, StreamingMr, StreamingNr, procs, isDeterministic);

        // K-axis (Fast mode only): tall-K shape where M and N are too small for
        // M-axis or N-axis splits. AxisSelector already gates K-axis on
        // !isDeterministic, but we double-check here so a later AxisSelector
        // refinement that ignores the determinism flag can't accidentally enable
        // K-axis under Deterministic mode.
        if (axis == ParallelismAxis.K && !isDeterministic && procs > 1)
        {
            RunKParallel(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, procs);
            return;
        }

        if (axis == ParallelismAxis.N && n >= procs * StreamingNr * 2)
        {
            RunNParallel(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, procs);
            return;
        }

        // Sub-G #375 note: 64×64×64 routes here (AxisSelector picks MN_2D
        // because neither M nor N reaches the procs×{mr,nr}×2 gate at
        // procs=16). An M-axis parallel path was prototyped — it correctly
        // partitions M across threads with disjoint C row-slices — but at
        // 64³ the per-thread work (M/procs)·N·K = 32K FMAs ≈ 2 µs at AVX2
        // peak is dwarfed by .NET ThreadPool dispatch overhead (~5-10 µs
        // per task). Net: M-axis parallel was 4 GFLOPS, the same as
        // serial, on this shape. The 10× gap to OpenBLAS (~30 GFLOPS,
        // observed) is OpenBLAS using a pre-warmed persistent worker pool
        // with µs-level dispatch — closing it requires a custom Streaming-
        // microkernel thread pool, not a TPL Parallel.For wrapper. Tracked
        // as a follow-up; for now the kernel-level gap on tiny cubes
        // remains the binding Sub-G blocker on this shape family.

        RunSerial(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
    }

    /// <summary>
    /// K-axis split: partition K across <paramref name="procs"/> threads. Each
    /// thread accumulates its partial C[M,N] over its K-slice; partials are
    /// reduced in fixed pairwise order. Non-associative — Fast mode only.
    /// </summary>
    private static void RunKParallel<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        int procs) where T : unmanaged
    {
        int totalElems = m * n;
        if (typeof(T) == typeof(double))
        {
            var partials = new Memory<double>[procs];
            for (int p = 0; p < procs; p++) partials[p] = new double[totalElems];

            unsafe
            {
                fixed (T* aPtr = a) fixed (T* bPtr = b)
                {
                    T* aLocal = aPtr;
                    T* bLocal = bPtr;
                    int aLen = a.Length, bLen = b.Length;
                    int ldaLocal = lda, ldbLocal = ldb;
                    int mLocal = m, nLocal = n;
                    bool taLocal = transA, tbLocal = transB;
                    int procsLocal = procs;

                    StreamingWorkerPool.Dispatch(procsLocal, p =>
                    {
                        var (kStart, kLen) = KAxisDriver.GetThreadRange(k, procsLocal, p);
                        if (kLen <= 0) return;

                        int aOffset = taLocal ? kStart * ldaLocal : kStart;
                        int bOffset = tbLocal ? kStart : kStart * ldbLocal;

                        var aSpan = new ReadOnlySpan<T>(aLocal + aOffset, aLen - aOffset);
                        var bSpan = new ReadOnlySpan<T>(bLocal + bOffset, bLen - bOffset);
                        var partialSpan = partials[p].Span;
                        // Cast partial buffer (double) to T span.
                        Span<T> cTyped = MemoryMarshal.Cast<double, T>(partialSpan);
                        RunSerial(aSpan, ldaLocal, taLocal,
                                  bSpan, ldbLocal, tbLocal,
                                  cTyped, nLocal,
                                  mLocal, nLocal, kLen);
                    });
                }
            }

            ReductionTree.ReducePairwiseFp64(partials, totalElems);
            // Copy reduced partials[0] into caller's C (row-major M×N with stride ldc).
            var src = partials[0].Span;
            var cDouble = MemoryMarshal.Cast<T, double>(c);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    cDouble[i * ldc + j] = src[i * n + j];
        }
        else if (typeof(T) == typeof(float))
        {
            var partials = new Memory<float>[procs];
            for (int p = 0; p < procs; p++) partials[p] = new float[totalElems];

            unsafe
            {
                fixed (T* aPtr = a) fixed (T* bPtr = b)
                {
                    T* aLocal = aPtr;
                    T* bLocal = bPtr;
                    int aLen = a.Length, bLen = b.Length;
                    int ldaLocal = lda, ldbLocal = ldb;
                    int mLocal = m, nLocal = n;
                    bool taLocal = transA, tbLocal = transB;
                    int procsLocal = procs;

                    StreamingWorkerPool.Dispatch(procsLocal, p =>
                    {
                        var (kStart, kLen) = KAxisDriver.GetThreadRange(k, procsLocal, p);
                        if (kLen <= 0) return;

                        int aOffset = taLocal ? kStart * ldaLocal : kStart;
                        int bOffset = tbLocal ? kStart : kStart * ldbLocal;

                        var aSpan = new ReadOnlySpan<T>(aLocal + aOffset, aLen - aOffset);
                        var bSpan = new ReadOnlySpan<T>(bLocal + bOffset, bLen - bOffset);
                        var partialSpan = partials[p].Span;
                        Span<T> cTyped = MemoryMarshal.Cast<float, T>(partialSpan);
                        RunSerial(aSpan, ldaLocal, taLocal,
                                  bSpan, ldbLocal, tbLocal,
                                  cTyped, nLocal,
                                  mLocal, nLocal, kLen);
                    });
                }
            }

            ReductionTree.ReducePairwiseFp32(partials, totalElems);
            var src = partials[0].Span;
            var cFloat = MemoryMarshal.Cast<T, float>(c);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    cFloat[i * ldc + j] = src[i * n + j];
        }
        else
        {
            throw new NotSupportedException($"StreamingStrategy K-axis does not support T={typeof(T).Name}.");
        }
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

                StreamingWorkerPool.Dispatch(procsLocal, p =>
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
