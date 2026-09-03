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
    /// Scheduling grain used for AxisSelector threshold computation. This is deliberately
    /// independent of the active kernel's SIMD width: the scheduler requires at least
    /// sixteen columns per worker, while N-axis partition boundaries use the exact kernel
    /// tile width returned by <see cref="GetColumnTileWidth{T}"/>.
    /// </summary>
    private const int StreamingSchedulingNr = 8;
    private const int StreamingMr = 8;
    private const int NParallelismThresholdMultiplier = 2;
    private const int ScalarColumnTileWidth = 1;

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

        var axis = SelectParallelismAxis(m, n, k, procs, isDeterministic);

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

        if (axis == ParallelismAxis.N &&
            (long)n >= (long)procs * StreamingSchedulingNr * NParallelismThresholdMultiplier)
        {
            int columnTileWidth = GetColumnTileWidth<T>(transB);
            RunNParallel(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, procs, columnTileWidth);
            return;
        }

        // Sub-G #375 (Layer B): M-axis + MN_2D fallback through the
        // persistent worker pool. AxisSelector picks MN_2D for shapes where
        // neither M nor N reaches the procs×{mr,nr}×2 gate (e.g. 64×64×64
        // FP64 at procs=16). The MN_2D path falls through to M-axis since
        // C[m,n] row-slicing has disjoint writes and matches the
        // RunNParallel column-disjoint contract on the other axis.
        // The persistent pool's sub-µs dispatch latency makes this win
        // even at tiny shapes (the earlier TPL-Parallel.For prototype
        // was net-zero on 64³ because dispatch overhead ≈ compute).
        if ((axis == ParallelismAxis.M || axis == ParallelismAxis.MN_2D)
            && procs > 1
            && m >= StreamingMr
            && (long)m * n * k >= AxisSelector.ParallelWorkThreshold)
        {
            int effectiveProcs = Math.Min(procs, m / StreamingMr);
            if (effectiveProcs >= 2)
            {
                RunMParallel(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, effectiveProcs);
                return;
            }
        }

        RunSerial(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
    }

    internal static ParallelismAxis SelectParallelismAxis(
        int m, int n, int k, int procs, bool isDeterministic) =>
        AxisSelector.Select(m, n, k, StreamingMr, StreamingSchedulingNr, procs, isDeterministic);

    /// <summary>
    /// Partition M across <paramref name="procs"/> threads. Each thread
    /// writes a disjoint row slice of C — bit-exact across thread counts.
    /// Mirror of <see cref="RunNParallel{T}"/> on the M axis; used for
    /// shapes where N is too small for N-axis splitting (e.g., 64×64×64
    /// cubes where N=64 doesn't reach procs×nr×2=256 at procs=16).
    /// </summary>
    private static void RunMParallel<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        int procs) where T : unmanaged
    {
        unsafe
        {
            fixed (T* aPtr = a)
            fixed (T* bPtr = b)
            fixed (T* cPtr = c)
            {
                T* aLocal = aPtr;
                T* bLocal = bPtr;
                T* cLocal = cPtr;
                int aLen = a.Length, bLen = b.Length, cLen = c.Length;
                int procsLocal = procs;
                int mLocal = m;
                int nLocal = n;
                int kLocal = k;
                int ldaLocal = lda, ldbLocal = ldb, ldcLocal = ldc;
                bool taLocal = transA, tbLocal = transB;

                PersistentParallelExecutor.Instance.Execute(procsLocal, p =>
                {
                    int mStart = (int)(((long)p * mLocal) / procsLocal);
                    int mEnd = (int)(((long)(p + 1) * mLocal) / procsLocal);
                    int mChunk = mEnd - mStart;
                    if (mChunk <= 0) return;

                    // A slice along M: depends on transA.
                    //   transA=false: A[M, K] row-major. Row mStart starts at
                    //                 a[mStart*lda]; stride is lda. Kernel sees [mChunk, K].
                    //   transA=true:  A[K, M] row-major. Column mStart starts at a[mStart];
                    //                 stride between rows is lda. Kernel sees a [K, mChunk] panel.
                    int aOffset = taLocal ? mStart : mStart * ldaLocal;
                    int aSliceLen = aLen - aOffset;

                    // C slice along M: row-major, row mStart starts at c[mStart*ldc].
                    int cOffset = mStart * ldcLocal;
                    int cSliceLen = cLen - cOffset;

                    var aSpan = new ReadOnlySpan<T>(aLocal + aOffset, aSliceLen);
                    var bSpan = new ReadOnlySpan<T>(bLocal, bLen);
                    var cSpan = new Span<T>(cLocal + cOffset, cSliceLen);

                    RunSerial(aSpan, ldaLocal, taLocal,
                              bSpan, ldbLocal, tbLocal,
                              cSpan, ldcLocal,
                              mChunk, nLocal, kLocal);
                });
            }
        }
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

                    PersistentParallelExecutor.Instance.Execute(procsLocal, p =>
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

                    PersistentParallelExecutor.Instance.Execute(procsLocal, p =>
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
        int procs,
        int columnTileWidth) where T : unmanaged
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

                PersistentParallelExecutor.Instance.Execute(procsLocal, p =>
                {
                    var (nStart, nEnd) = GetNPartitionRange(
                        nLocal, procsLocal, p, columnTileWidth);
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
    /// Returns the active streaming kernel's smallest SIMD column tile. A transposed B
    /// is processed one output column at a time, even when its K reduction is vectorized.
    /// </summary>
    internal static int GetColumnTileWidth<T>(bool transB) where T : unmanaged
    {
        if (transB)
        {
            return ScalarColumnTileWidth;
        }

        if (typeof(T) == typeof(double))
        {
            if (Avx512Streaming.IsSupported) return Avx512Streaming.Fp64ColumnTileWidth;
            if (Avx2Streaming.IsSupported) return Avx2Streaming.Fp64ColumnTileWidth;
            if (NeonStreaming.IsSupported) return NeonStreaming.Fp64ColumnTileWidth;
            if (PortableSimdStreaming.IsSupported) return PortableSimdStreaming.Fp64ColumnTileWidth;
            return ScalarColumnTileWidth;
        }

        if (typeof(T) == typeof(float))
        {
            if (Avx512Streaming.IsSupported) return Avx512Streaming.Fp32ColumnTileWidth;
            if (Avx2Streaming.IsSupported) return Avx2Streaming.Fp32ColumnTileWidth;
            if (NeonStreaming.IsSupported) return NeonStreaming.Fp32ColumnTileWidth;
            return ScalarColumnTileWidth;
        }

        throw new NotSupportedException($"StreamingStrategy does not support T={typeof(T).Name}.");
    }

    /// <summary>
    /// Partitions complete kernel tiles across workers. Every boundary except the final
    /// matrix tail is tile-aligned, so a parallel slice follows the same SIMD path as the
    /// corresponding columns in serial execution.
    /// </summary>
    internal static (int Start, int End) GetNPartitionRange(
        int n, int procs, int partition, int columnTileWidth)
    {
        int fullColumnTiles = n / columnTileWidth;
        int firstTile = (int)(((long)partition * fullColumnTiles) / procs);
        int tileEnd = (int)(((long)(partition + 1) * fullColumnTiles) / procs);
        int start = firstTile * columnTileWidth;
        int end = partition == procs - 1 ? n : tileEnd * columnTileWidth;
        return (start, end);
    }

    /// <summary>
    /// Serial microkernel dispatch: AVX-512 → AVX2 → Neon → BCL <see cref="System.Numerics.Vector{T}"/>
    /// (FP64) → scalar.
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
            // BCL Vector<T> tier. Reached on TFMs without System.Runtime.Intrinsics (net471,
            // where the AVX/Neon IsSupported above are compile-time false) and on any host whose
            // intrinsic sets are unavailable. Bit-identical to ScalarStreaming.RunFp64, and it
            // internally defers to it for transB / sub-lane n.
            if (PortableSimdStreaming.IsSupported)
            {
                PortableSimdStreaming.RunFp64(
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
