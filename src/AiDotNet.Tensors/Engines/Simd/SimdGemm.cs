using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.CpuJit;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// High-performance General Matrix Multiply (GEMM) using BLIS/GotoBLAS tiled architecture.
/// C[m,n] += A[m,k] * B[k,n] with FMA micro-kernel, panel packing, and cache-level blocking.
/// </summary>
internal static class SimdGemm
{
    // Cache blocking parameters (tuned for typical L1=32KB, L2=256KB, L3=8MB)
    // Iter 2: Mc lowered 256→128 for more row blocks (better parallel utilization).
    // Iter 3 (reverted): Mc=64 regressed across the board — too small for cache reuse.
    // Iter 8 (reverted): Kc=256 was tried to fit per-tile working set in L2, but the
    // extra pc iterations (4 instead of 2) doubled the number of parallel barriers and
    // regressed 512² by 1.5x. The L2-fit argument was wrong because packed A gets
    // evicted by packed B during the inner loop anyway. Kept at 512.
    private const int Mc = 128;  // Panel height for A (fits in L2)
    private const int Kc = 512;  // Panel depth (fits in L1)
    private const int Nc = 4096; // Panel width for B (fits in L3)

    // Micro-kernel register block: 6 rows x 16 columns
    // 6 rows * 16 cols = 96 floats = 12 Vector256<float> accumulators
    private const int Mr = 6;
    private const int Nr = 16;

    // A/B test toggle: set to false to force sequential SgemmTiled for baseline
    // comparisons. Defaults to true so multi-core systems get parallel execution.
    // Intended for benchmark A/B iteration, not production config.
    internal static bool UseParallelGemm = true;

    // Minimum problem size (m*n*k, count as flops/2) to enable parallel dispatch.
    // Iter 2 (2026-04-11): raised 4M → 20M after measuring that 256² (16.8M) regressed
    // with parallel on — thread-pool overhead dominates at that size. 512² (134M) and
    // 1024² (1B) are the real winners and comfortably clear the new threshold.
    private const long ParallelWorkThreshold = 20L * 1024 * 1024;

    /// <summary>
    /// Computes C = A * B where A is [m,k], B is [k,n], C is [m,n].
    /// All matrices are in row-major order. C is cleared before computation.
    /// </summary>
    [MethodImpl(Hot)]
    public static void Sgemm(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
        c.Clear();
        SgemmAdd(a, b, c, m, k, n);
    }

    /// <summary>
    /// Computes C = op(A) * op(B) with optional transpose on either operand.
    /// op(X) = X when transX=false, op(X) = X^T when transX=true.
    /// lda/ldb are the leading dimensions (row strides) of the source storage.
    /// This enables zero-copy matmul on transposed stride-based views.
    /// </summary>
    [MethodImpl(Hot)]
    public static void Sgemm(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n)
    {
        c.Clear();
        SgemmAdd(a, lda, transA, b, ldb, transB, c, m, k, n);
    }

    /// <summary>
    /// Computes C = beta*C + A*B. When beta=0, overwrites C (clears first).
    /// When beta=1, accumulates into C. Matches BLAS sgemm semantics.
    /// </summary>
    [MethodImpl(Hot)]
    public static void Sgemm(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n,
        float beta)
    {
        if (beta == 0f)
            c.Clear();
        else if (beta != 1f)
        {
            for (int i = 0; i < c.Length; i++)
                c[i] *= beta;
        }
        SgemmAdd(a, b, c, m, k, n);
    }

    /// <summary>
    /// Computes C += A * B (accumulates into C without clearing).
    /// </summary>
    [MethodImpl(Hot)]
    public static void SgemmAdd(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
        SgemmAdd(a, k, false, b, n, false, c, m, k, n);
    }

    /// <summary>
    /// Computes C += op(A) * op(B) with stride and transpose support.
    /// </summary>
    [MethodImpl(Hot)]
    public static void SgemmAdd(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n)
    {
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && m >= Mr && n >= Nr)
        {
            SgemmTiled(a, lda, transA, b, ldb, transB, c, m, k, n);
            return;
        }
#endif
        SgemmScalar(a, lda, transA, b, ldb, transB, c, m, k, n);
    }

    /// <summary>
    /// Scalar GEMM fallback with stride/transpose support.
    /// </summary>
    [MethodImpl(Hot)]
    private static void SgemmScalar(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            int cRowBase = i * n;
            for (int p = 0; p < k; p++)
            {
                // op(A)[i,p]: if transA, read A[p,i] = a[p*lda+i]; else A[i,p] = a[i*lda+p]
                float aip = transA ? a[p * lda + i] : a[i * lda + p];
                for (int j = 0; j < n; j++)
                {
                    // op(B)[p,j]: if transB, read B[j,p] = b[j*ldb+p]; else B[p,j] = b[p*ldb+j]
                    float bpj = transB ? b[j * ldb + p] : b[p * ldb + j];
#if NET5_0_OR_GREATER
                    c[cRowBase + j] = MathF.FusedMultiplyAdd(aip, bpj, c[cRowBase + j]);
#else
                    c[cRowBase + j] += aip * bpj;
#endif
                }
            }
        }
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Tiled GEMM with panel packing and FMA micro-kernel.
    /// Follows BLIS architecture: loop order is jc -> pc -> ic -> jr -> ir.
    /// Supports parallelism over both M and N dimensions for different matrix shapes.
    /// </summary>
    /// <summary>
    /// Small-M GEMM: for M &lt;= 64, skip packing and compute directly.
    /// Each row of C is computed as a dot product of a row of A with columns of B,
    /// using SIMD to process 8 columns of B at a time.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SgemmSmallM(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n)
    {
        fixed (float* pA = a, pB = b, pC = c)
        {
            for (int i = 0; i < m; i++)
            {
                int cRow = i * n;
                // Process N in chunks of 8 (AVX2 width)
                int j = 0;
                for (; j + 8 <= n; j += 8)
                {
                    var acc = Vector256<float>.Zero;
                    for (int p = 0; p < k; p++)
                    {
                        float aVal = transA ? pA[p * lda + i] : pA[i * lda + p];
                        var aVec = Vector256.Create(aVal);
                        int bIdx = transB ? (j * ldb + p) : (p * ldb + j);
                        // For transB, elements are strided — need gather or scalar
                        if (!transB)
                        {
                            var bVec = Avx.LoadVector256(pB + bIdx);
                            acc = Fma.MultiplyAdd(aVec, bVec, acc);
                        }
                        else
                        {
                            // Gather from transposed B — each element at stride ldb
                            var bVec = Vector256.Create(
                                pB[j * ldb + p], pB[(j + 1) * ldb + p], pB[(j + 2) * ldb + p], pB[(j + 3) * ldb + p],
                                pB[(j + 4) * ldb + p], pB[(j + 5) * ldb + p], pB[(j + 6) * ldb + p], pB[(j + 7) * ldb + p]);
                            acc = Fma.MultiplyAdd(aVec, bVec, acc);
                        }
                    }
                    Avx.Store(pC + cRow + j, acc);
                }
                // Scalar tail — use FMA to match SIMD rounding behavior
                for (; j < n; j++)
                {
                    float sum = 0;
                    for (int p = 0; p < k; p++)
                    {
                        float aVal = transA ? pA[p * lda + i] : pA[i * lda + p];
                        float bVal = transB ? pB[j * ldb + p] : pB[p * ldb + j];
                        sum = MathF.FusedMultiplyAdd(aVal, bVal, sum);
                    }
                    pC[cRow + j] = sum;
                }
            }
        }
    }

    [MethodImpl(Hot)]
    private static void SgemmTiled(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n)
    {
        // Decide parallel vs sequential up front. Parallel dispatches either:
        //   - 1D parallel (SgemmTiledParallelM): row blocks only, when nc is too small
        //     to justify col-sub parallelism OR when numRowBlocks already matches the
        //     target core count.
        //   - 2D parallel (SgemmTiledParallel2D): (row block × col sub) grid, when
        //     either M or N alone doesn't saturate available cores. Handles row-heavy
        //     problems (1024²) by splitting columns too, AND col-heavy problems
        //     (LM-head [64, 128]x[128, 50257]) by parallelizing the 1 row block across
        //     many col subs.
        int maxThreads = Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        int numRowBlocks = (m + Mc - 1) / Mc;
        bool canParallelize = UseParallelGemm
            && maxThreads > 1
            && numRowBlocks >= 1
            && !transA && !transB  // Parallel path uses the no-transpose Pack overloads
            && (long)m * k * n >= ParallelWorkThreshold;

        // Round up to micro-tile dimensions to avoid buffer overruns in PackA/PackB padding
        int mcRounded = ((Mc + Mr - 1) / Mr) * Mr;
        int ncRounded = ((Nc + Nr - 1) / Nr) * Nr;
        int packedASize = mcRounded * Kc;
        int packedBSize = Kc * ncRounded;
        float[] packedABuf = ArrayPool<float>.Shared.Rent(packedASize);
        float[] packedBBuf = ArrayPool<float>.Shared.Rent(packedBSize);

        try
        {
            for (int jc = 0; jc < n; jc += Nc)
            {
                int nc = Math.Min(Nc, n - jc);

                for (int pc = 0; pc < k; pc += Kc)
                {
                    int kc = Math.Min(Kc, k - pc);

                    // Decide per-tile parallel dispatch. numColSubs is adaptive to
                    // logical core count so numRowBlocks * numColSubs ≈ maxThreads.
                    // Iter 5: logical cores, not physical — SMT siblings help because
                    // blocked GEMM is load-port-limited not FMA-port-limited.
                    // Iter 7: allow numRowBlocks=1 through the 2D path so col-heavy
                    // problems like LM-head [64,128]x[128,50257] get parallelism.
                    int desiredColSubs = canParallelize ? Math.Max(1, maxThreads / numRowBlocks) : 1;
                    int maxColSubs = Math.Max(1, nc / (Nr * 4));
                    int numColSubs = Math.Min(desiredColSubs, maxColSubs);
                    int totalTiles = numRowBlocks * numColSubs;

                    bool used2D = canParallelize && numColSubs >= 2 && totalTiles >= 2;
                    bool used1D = canParallelize && !used2D && numRowBlocks >= 2;

                    if (used2D)
                    {
                        SgemmTiledParallel2D(
                            a, b, c,
                            m, k, n,
                            jc, nc, pc, kc,
                            numRowBlocks, numColSubs);
                    }
                    else if (used1D)
                    {
                        // 1D parallel-M: each worker owns a disjoint row block with its
                        // own packed-A, B is packed once and shared read-only. Output
                        // row ranges are disjoint so no synchronization on C is needed.
                        SgemmTiledParallelM(
                            a, b, c,
                            m, k, n,
                            jc, nc, pc, kc,
                            numRowBlocks, packedBBuf);
                    }
                    else
                    {
                        // Sequential path (original)
                        PackA(a, packedABuf, lda, transA, ic: 0, mc: Math.Min(Mc, m), pc, kc);
                        PackB(b, packedBBuf, ldb, transB, pc, kc, jc, nc);

                        for (int ic = 0; ic < m; ic += Mc)
                        {
                            int mc = Math.Min(Mc, m - ic);
                            if (ic > 0) // first block already packed above
                                PackA(a, packedABuf, lda, transA, ic, mc, pc, kc);
                            MacroKernel(packedABuf, packedBBuf, c, mc, nc, kc, n, ic, jc);
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packedABuf);
            ArrayPool<float>.Shared.Return(packedBBuf);
        }
    }

    /// <summary>
    /// N-dimension parallel GEMM: splits columns across workers.
    /// Ideal for small M, large N (e.g. Conv2D im2col where M=outChannels, N=outputSize).
    /// Each worker packs its own B slice and processes all M rows for that column range.
    /// Uses unsafe pinned pointers to avoid array copies for closure capture.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SgemmTiledParallelN(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n,
        int jc, int nc, int pc, int kc,
        int numNrBlocks, int maxThreads,
        float[] packedABuf)
    {
        int numWorkers = Math.Min(maxThreads, numNrBlocks);
        int nrPerWorker = (numNrBlocks + numWorkers - 1) / numWorkers;

        // Pre-pack A (shared across all workers, m is small so only 1 Mc block)
        int firstMc = Math.Min(Mc, m);
        PackA(a, packedABuf, k, 0, firstMc, pc, kc);

        // Pre-pack B slices for each worker
        var packedBSlices = new float[numWorkers][];
        var sliceNcs = new int[numWorkers];
        var sliceJStarts = new int[numWorkers];
        int actualWorkers = numWorkers;

        for (int w = 0; w < numWorkers; w++)
        {
            int nrStart = w * nrPerWorker;
            if (nrStart >= numNrBlocks) { actualWorkers = w; break; }
            int nrEnd = Math.Min(nrStart + nrPerWorker, numNrBlocks);
            int jStart = nrStart * Nr;
            int jEnd = Math.Min(nrEnd * Nr, nc);
            int localNc = jEnd - jStart;
            sliceNcs[w] = localNc;
            sliceJStarts[w] = jStart;

            // Round up to Nr boundary since PackB pads remaining columns
            int packedNc = ((localNc + Nr - 1) / Nr) * Nr;
            packedBSlices[w] = ArrayPool<float>.Shared.Rent(kc * packedNc);
            PackB(b, packedBSlices[w], n, pc, kc, jc + jStart, localNc);
        }

        // Pin A (only needed if m > Mc for additional blocks)
        float[]? aArr = null;
        GCHandle aHandle = default;
        float* aPtr = null;
        if (m > Mc)
        {
            aArr = ArrayPool<float>.Shared.Rent(a.Length);
            a.CopyTo(aArr);
            aHandle = GCHandle.Alloc(aArr, GCHandleType.Pinned);
            aPtr = (float*)aHandle.AddrOfPinnedObject();
        }

        try
        {
            // Capture locals for closure
            var localPackedABuf = packedABuf;
            var localPackedBSlices = packedBSlices;
            var localSliceNcs = sliceNcs;
            var localSliceJStarts = sliceJStarts;
            var localAPtr = aPtr;
            int localALen = a.Length;
            int localM = m, localK = k, localN = n;
            int localJc = jc, localPc = pc, localKc = kc;
            int localFirstMc = firstMc;
            int cLen = c.Length;

            // Pin C for direct pointer access (workers write to non-overlapping columns)
            fixed (float* cPtr = c)
            {
                var localCPtr = cPtr;
                var localCLen = cLen;

                Helpers.CpuParallelSettings.LightweightParallel(actualWorkers, workerId =>
                {
                    int workerNc = localSliceNcs[workerId];
                    int jStart = localSliceJStarts[workerId];
                    var cSpan = new Span<float>(localCPtr, localCLen);

                    // First Mc block: use shared pre-packed A
                    MacroKernel(localPackedABuf, localPackedBSlices[workerId],
                        cSpan, localFirstMc, workerNc, localKc, localN, 0, localJc + jStart);

                    // Additional Mc blocks (if m > Mc)
                    for (int ic = Mc; ic < localM; ic += Mc)
                    {
                        int mc = Math.Min(Mc, localM - ic);
                        // Round up to Mr boundary since PackA pads remaining rows
                        int packedMc = ((mc + Mr - 1) / Mr) * Mr;
                        float[] workerPackedA = ArrayPool<float>.Shared.Rent(packedMc * localKc);
                        try
                        {
                            var aSpan = new ReadOnlySpan<float>(localAPtr, localALen);
                            PackA(aSpan, workerPackedA, localK, ic, mc, localPc, localKc);
                            MacroKernel(workerPackedA, localPackedBSlices[workerId],
                                cSpan, mc, workerNc, localKc, localN, ic, localJc + jStart);
                        }
                        finally
                        {
                            ArrayPool<float>.Shared.Return(workerPackedA);
                        }
                    }
                });
            }
        }
        finally
        {
            if (aHandle.IsAllocated) aHandle.Free();
            if (aArr is not null) ArrayPool<float>.Shared.Return(aArr);
            for (int w = 0; w < actualWorkers; w++)
            {
                ArrayPool<float>.Shared.Return(packedBSlices[w]);
            }
        }
    }

    /// <summary>
    /// M-dimension parallel GEMM: splits rows across workers.
    /// Ideal for tall matrices (m >= 512). Uses pinned pointers to avoid array copies.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SgemmTiledParallelM(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n,
        int jc, int nc, int pc, int kc,
        int numRowBlocks, float[] packedBBuf)
    {
        // Pack B once (shared across all M workers, read-only)
        PackB(b, packedBBuf, n, pc, kc, jc, nc);

        // Pin A for closure capture (workers read non-overlapping rows)
        float[] aArr = ArrayPool<float>.Shared.Rent(a.Length);
        a.CopyTo(aArr);
        var aHandle = GCHandle.Alloc(aArr, GCHandleType.Pinned);

        try
        {
            var localPackedBBuf = packedBBuf;
            int localM = m, localK = k, localN = n;
            int localJc = jc, localNc = nc, localPc = pc, localKc = kc;
            float* localAPtr = (float*)aHandle.AddrOfPinnedObject();
            int localALen = a.Length;
            int cLen = c.Length;

            // Pin C for direct pointer access (workers write to non-overlapping rows)
            fixed (float* cPtr = c)
            {
                var localCPtr = cPtr;
                var localCLen = cLen;

                Helpers.CpuParallelSettings.LightweightParallel(numRowBlocks, iiBlock =>
                {
                    int ic = iiBlock * Mc;
                    int mc = Math.Min(Mc, localM - ic);
                    // Round up to Mr boundary since PackA pads remaining rows
                    int packedMc = ((mc + Mr - 1) / Mr) * Mr;
                    float[] localPackedA = ArrayPool<float>.Shared.Rent(packedMc * localKc);
                    try
                    {
                        var aSpan = new ReadOnlySpan<float>(localAPtr, localALen);
                        var cSpan = new Span<float>(localCPtr, localCLen);
                        PackA(aSpan, localPackedA, localK, ic, mc, localPc, localKc);
                        MacroKernel(localPackedA, localPackedBBuf, cSpan, mc, localNc, localKc, localN, ic, localJc);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(localPackedA);
                    }
                });
            }
        }
        finally
        {
            aHandle.Free();
            ArrayPool<float>.Shared.Return(aArr);
        }
    }

    /// <summary>
    /// 2D parallel GEMM: dispatches a grid of (ic_block × jc_sub) tiles in parallel.
    /// Iter 4 (2026-04-11): breaks past the numRowBlocks parallelism ceiling. On a
    /// 16-core machine at 1024² with Mc=128, the 1D M-parallel path was limited to
    /// 8 workers; this variant adds col-sub parallelism to fill the remaining cores.
    ///
    /// Layout:
    ///   - Each row block r has its own packed-A buffer, shared across all col subs
    ///     (so PackA runs numRowBlocks times total, not numRowBlocks * numColSubs).
    ///   - Each col sub cs has its own packed-B buffer, shared across all row blocks
    ///     (so PackB runs numColSubs times, not numRowBlocks * numColSubs).
    ///   - Compute dispatches numRowBlocks * numColSubs tiles in parallel, each
    ///     reading its row's packed-A and its col-sub's packed-B, writing to a
    ///     disjoint (mc x subNc) region of C.
    ///
    /// Determinism: output regions are disjoint per tile, inner pc loop is still
    /// sequential, and the micro-kernel's FMA order is fixed. Results are bit-exact
    /// identical to the sequential and 1D-parallel paths.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SgemmTiledParallel2D(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n,
        int jc, int nc, int pc, int kc,
        int numRowBlocks, int numColSubBlocks)
    {
        // colSubSize: number of B columns per sub-block, rounded down to a multiple
        // of Nr so MacroKernel's Nr panel loop stays clean. The last sub absorbs the
        // remainder. If nc < numColSubBlocks*Nr, fall back to the 1D parallel path.
        int colSubSize = (nc / numColSubBlocks / Nr) * Nr;
        if (colSubSize < Nr)
        {
            // Degenerate — not enough cols per sub. Caller should have checked,
            // but we guard here to avoid a zero-sized sub.
            colSubSize = nc;
            numColSubBlocks = 1;
        }

        int mcRounded = ((Mc + Mr - 1) / Mr) * Mr;
        int packedASizePerRow = mcRounded * Kc;
        int colSubRounded = ((colSubSize + Nr - 1) / Nr) * Nr;
        int lastColSubWidth = nc - (numColSubBlocks - 1) * colSubSize;
        int lastColSubRounded = ((lastColSubWidth + Nr - 1) / Nr) * Nr;
        int packedBSizePerSub = Kc * Math.Max(colSubRounded, lastColSubRounded);

        var packedABufs = new float[numRowBlocks][];
        var packedBBufs = new float[numColSubBlocks][];
        for (int r = 0; r < numRowBlocks; r++)
            packedABufs[r] = ArrayPool<float>.Shared.Rent(packedASizePerRow);
        for (int cs = 0; cs < numColSubBlocks; cs++)
            packedBBufs[cs] = ArrayPool<float>.Shared.Rent(packedBSizePerSub);

        try
        {
            var localPackedABufs = packedABufs;
            var localPackedBBufs = packedBBufs;
            int localM = m, localK = k, localN = n;
            int localJc = jc, localNc = nc, localPc = pc, localKc = kc;
            int localColSubSize = colSubSize;
            int localNumColSubs = numColSubBlocks;
            int localMc = Mc;

            // Pin A, B, C directly via the input spans — no extra copy. The fixed
            // block outlives every LightweightParallel call below (synchronous
            // dispatch waits for all workers before returning), so the pointers
            // stay valid throughout. Iter 6: previous version copied A and B into
            // rented arrays for pinning, which added 8MB of serial copy at 1024².
            fixed (float* aPtr0 = a)
            fixed (float* bPtr0 = b)
            fixed (float* cPtr0 = c)
            {
                float* localAPtr = aPtr0;
                int localALen = a.Length;
                float* localBPtr = bPtr0;
                int localBLen = b.Length;
                float* localCPtr = cPtr0;
                int localCLen = c.Length;

                // Iter 12: fuse phases 1 (pack A) and 2 (pack B) into a single parallel
                // dispatch to save one barrier per (jc, pc) iteration. Tasks 0..numRowBlocks-1
                // pack A, tasks numRowBlocks..numRowBlocks+numColSubs-1 pack B. All
                // independent output buffers — no contention, all can run concurrently.
                int numPackTasks = numRowBlocks + numColSubBlocks;
                int localNumRowBlocks = numRowBlocks;
                Helpers.CpuParallelSettings.LightweightParallel(numPackTasks, taskId =>
                {
                    if (taskId < localNumRowBlocks)
                    {
                        // Pack A for row block r = taskId
                        int r = taskId;
                        int ic = r * localMc;
                        int mcLocal = Math.Min(localMc, localM - ic);
                        if (mcLocal > 0)
                        {
                            var aSpan = new ReadOnlySpan<float>(localAPtr, localALen);
                            PackA(aSpan, localPackedABufs[r], localK, ic, mcLocal, localPc, localKc);
                        }
                    }
                    else
                    {
                        // Pack B for col sub cs = taskId - numRowBlocks
                        int cs = taskId - localNumRowBlocks;
                        int jStart = cs * localColSubSize;
                        int subNc = (cs == localNumColSubs - 1) ? (localNc - jStart) : localColSubSize;
                        if (subNc > 0)
                        {
                            var bSpan = new ReadOnlySpan<float>(localBPtr, localBLen);
                            PackB(bSpan, localPackedBBufs[cs], localN, localPc, localKc, localJc + jStart, subNc);
                        }
                    }
                });

                // Phase 2 (was 3): parallel compute for all (ic, jc_sub) tiles.
                int totalTiles = numRowBlocks * numColSubBlocks;
                Helpers.CpuParallelSettings.LightweightParallel(totalTiles, tileId =>
                {
                    int r = tileId / localNumColSubs;
                    int cs = tileId % localNumColSubs;
                    int ic = r * localMc;
                    int mcLocal = Math.Min(localMc, localM - ic);
                    int jStart = cs * localColSubSize;
                    int subNc = (cs == localNumColSubs - 1) ? (localNc - jStart) : localColSubSize;
                    if (mcLocal > 0 && subNc > 0)
                    {
                        var cSpan = new Span<float>(localCPtr, localCLen);
                        MacroKernel(
                            localPackedABufs[r], localPackedBBufs[cs],
                            cSpan, mcLocal, subNc, localKc, localN,
                            ic, localJc + jStart);
                    }
                });
            }
        }
        finally
        {
            for (int r = 0; r < numRowBlocks; r++)
                ArrayPool<float>.Shared.Return(packedABufs[r]);
            for (int cs = 0; cs < numColSubBlocks; cs++)
                ArrayPool<float>.Shared.Return(packedBBufs[cs]);
        }
    }

    /// <summary>
    /// Pack op(A)[ic:ic+mc, pc:pc+kc] into row-panel format for sequential access in micro-kernel.
    /// When transA=false: reads A[row, col] = a[row*lda + col] (row-major).
    /// When transA=true:  reads A^T[row, col] = a[col*lda + row] (transposed).
    /// Layout: groups of Mr rows, each stored as Mr x kc contiguous block.
    /// Iter 14: non-transpose full-Mr panel path uses direct pointer arithmetic with
    /// 6 hoisted row pointers and inner-loop unroll-by-4. Eliminates the JIT's bounds
    /// checks and repeated index calculations, cutting pack A time substantially.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackA(ReadOnlySpan<float> a, float[] packed, int lda, bool transA, int ic, int mc, int pc, int kc)
    {
        int pos = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        // Fast path: non-transpose, full Mr-row panels. Hoist 6 row pointers out
        // of the p-loop, unroll p by 4.
        if (!transA)
        {
            fixed (float* aPtr = a)
            fixed (float* packedPtr = packed)
            {
                for (; i + Mr <= mc; i += Mr)
                {
                    float* row0 = aPtr + (ic + i + 0) * lda + pc;
                    float* row1 = aPtr + (ic + i + 1) * lda + pc;
                    float* row2 = aPtr + (ic + i + 2) * lda + pc;
                    float* row3 = aPtr + (ic + i + 3) * lda + pc;
                    float* row4 = aPtr + (ic + i + 4) * lda + pc;
                    float* row5 = aPtr + (ic + i + 5) * lda + pc;
                    float* pp = packedPtr + pos;

                    int p = 0;
                    for (; p + 4 <= kc; p += 4)
                    {
                        pp[0]  = row0[0]; pp[1]  = row1[0]; pp[2]  = row2[0]; pp[3]  = row3[0]; pp[4]  = row4[0]; pp[5]  = row5[0];
                        pp[6]  = row0[1]; pp[7]  = row1[1]; pp[8]  = row2[1]; pp[9]  = row3[1]; pp[10] = row4[1]; pp[11] = row5[1];
                        pp[12] = row0[2]; pp[13] = row1[2]; pp[14] = row2[2]; pp[15] = row3[2]; pp[16] = row4[2]; pp[17] = row5[2];
                        pp[18] = row0[3]; pp[19] = row1[3]; pp[20] = row2[3]; pp[21] = row3[3]; pp[22] = row4[3]; pp[23] = row5[3];
                        pp += 24;
                        row0 += 4; row1 += 4; row2 += 4; row3 += 4; row4 += 4; row5 += 4;
                    }
                    for (; p < kc; p++)
                    {
                        pp[0] = row0[0]; pp[1] = row1[0]; pp[2] = row2[0]; pp[3] = row3[0]; pp[4] = row4[0]; pp[5] = row5[0];
                        pp += 6;
                        row0++; row1++; row2++; row3++; row4++; row5++;
                    }
                    pos += Mr * kc;
                }
            }
        }
        else
#endif
        {
            // Scalar fallback (transpose path or older TFMs)
            for (; i + Mr <= mc; i += Mr)
            {
                for (int p = 0; p < kc; p++)
                {
                    for (int ii = 0; ii < Mr; ii++)
                    {
                        int row = ic + i + ii;
                        int col = pc + p;
                        packed[pos++] = transA ? a[col * lda + row] : a[row * lda + col];
                    }
                }
            }
        }

        // Remaining rows (less than Mr)
        int remaining = mc - i;
        if (remaining > 0)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int ii = 0; ii < remaining; ii++)
                {
                    int row = ic + i + ii;
                    int col = pc + p;
                    packed[pos++] = transA ? a[col * lda + row] : a[row * lda + col];
                }
                for (int ii = remaining; ii < Mr; ii++)
                {
                    packed[pos++] = 0;
                }
            }
        }
    }

    /// <summary>
    /// Backward-compatible PackA without transpose (used by parallel paths).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PackA(ReadOnlySpan<float> a, float[] packed, int lda, int ic, int mc, int pc, int kc)
        => PackA(a, packed, lda, false, ic, mc, pc, kc);

    /// <summary>
    /// Pack op(B)[pc:pc+kc, jc:jc+nc] into column-panel format.
    /// When transB=false: reads B[row, col] = b[row*ldb + col] (row-major).
    /// When transB=true:  reads B^T[row, col] = b[col*ldb + row] (transposed).
    /// Layout: groups of Nr columns, each stored as kc x Nr contiguous block.
    /// Iter 11: non-transpose full-Nr path uses 2x Vector256 loads/stores per k
    /// iteration — reads 16 contiguous floats from a B row and writes them to
    /// the packed buffer as two 256-bit aligned writes. ~8x faster than the
    /// scalar fallback on cached data.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackB(ReadOnlySpan<float> b, float[] packed, int ldb, bool transB, int pc, int kc, int jc, int nc)
    {
        int pos = 0;
        int j = 0;

#if NET5_0_OR_GREATER
        // Fast path: non-transpose, full Nr-column panels with SIMD copy.
        // Nr=16 floats per row → 2 Vector256<float> = 64 bytes per iter.
        if (!transB && Avx.IsSupported)
        {
            fixed (float* pBBase = b)
            fixed (float* pPackedBase = packed)
            {
                for (; j + Nr <= nc; j += Nr)
                {
                    float* pPacked = pPackedBase + pos;
                    float* pBCol = pBBase + pc * ldb + (jc + j);
                    for (int p = 0; p < kc; p++)
                    {
                        var v0 = Avx.LoadVector256(pBCol);
                        var v1 = Avx.LoadVector256(pBCol + 8);
                        Avx.Store(pPacked, v0);
                        Avx.Store(pPacked + 8, v1);
                        pBCol += ldb;
                        pPacked += Nr;
                    }
                    pos += kc * Nr;
                }
            }
        }
#endif

        // Remaining full panels (transpose path or older TFMs) — scalar loop.
        for (; j + Nr <= nc; j += Nr)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int jj = 0; jj < Nr; jj++)
                {
                    int row = pc + p;
                    int col = jc + j + jj;
                    packed[pos++] = transB ? b[col * ldb + row] : b[row * ldb + col];
                }
            }
        }

        // Remaining columns (less than Nr)
        int remaining = nc - j;
        if (remaining > 0)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int jj = 0; jj < remaining; jj++)
                {
                    int row = pc + p;
                    int col = jc + j + jj;
                    packed[pos++] = transB ? b[col * ldb + row] : b[row * ldb + col];
                }
                for (int jj = remaining; jj < Nr; jj++)
                {
                    packed[pos++] = 0;
                }
            }
        }
    }

    /// <summary>
    /// Backward-compatible PackB without transpose (used by parallel paths).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PackB(ReadOnlySpan<float> b, float[] packed, int ldb, int pc, int kc, int jc, int nc)
        => PackB(b, packed, ldb, false, pc, kc, jc, nc);

    /// <summary>
    /// Macro-kernel: iterate over packed panels with Mr x Nr micro-kernel tiles.
    /// Uses JIT-compiled micro-kernel when available for guaranteed optimal register allocation.
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void MacroKernel(
        float[] packedA,
        float[] packedB,
        Span<float> c,
        int mc, int nc, int kc,
        int ldc, int icOffset, int jcOffset)
    {
        int nrBlocks = (nc + Nr - 1) / Nr;
        int mrBlocks = (mc + Mr - 1) / Mr;

        // Try JIT micro-kernel: bakes ldc as immediate, guarantees 12 YMM accumulators in registers
        CpuJitKernels.GemmMicroKernel? jitKernel =
            CpuJitSelfTest.IsVerified ? CpuJitKernels.GetGemmMicroKernel(kc, ldc) : null;

        // Iter 10 (reverted): tried pinning packedA/B/C once around the whole micro-
        // kernel loop to eliminate per-call fixed overhead. It regressed ~14% at 512².
        // The JIT was already eliding the inner-loop fixed statements, and the outer
        // fixed created a longer-lived GC pin that seemed to hurt. Reverted.
        for (int jr = 0; jr < nrBlocks; jr++)
        {
            int jLocal = jr * Nr;
            int nc_actual = Math.Min(Nr, nc - jLocal);
            int bPanelOffset = jr * kc * Nr;

            for (int ir = 0; ir < mrBlocks; ir++)
            {
                int iLocal = ir * Mr;
                int mc_actual = Math.Min(Mr, mc - iLocal);
                int aPanelOffset = ir * kc * Mr;

                if (mc_actual == Mr && nc_actual == Nr)
                {
                    if (jitKernel is not null)
                    {
                        // JIT micro-kernel: pass pointers directly
                        fixed (float* pA = &packedA[aPanelOffset])
                        fixed (float* pB = &packedB[bPanelOffset])
                        fixed (float* pC = c)
                        {
                            int cOffset = (icOffset + iLocal) * ldc + (jcOffset + jLocal);
                            jitKernel(pA, pB, pC + cOffset, kc);
                        }
                    }
                    else
                    {
                        // C# intrinsics micro-kernel fallback
                        MicroKernel6x16(
                            packedA, aPanelOffset,
                            packedB, bPanelOffset,
                            c, ldc,
                            icOffset + iLocal, jcOffset + jLocal,
                            kc);
                    }
                }
                else
                {
                    // Edge case: partial tile
                    MicroKernelScalar(
                        packedA, aPanelOffset,
                        packedB, bPanelOffset,
                        c, ldc,
                        icOffset + iLocal, jcOffset + jLocal,
                        kc, mc_actual, nc_actual);
                }
            }
        }
    }

    /// <summary>
    /// 6x16 FMA micro-kernel: computes a 6-row x 16-column tile of C.
    /// Uses 12 Vector256 accumulators (6 rows x 2 vectors of 8 floats = 16 columns).
    /// Inner loop over K dimension broadcasts A elements and FMA with B row.
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void MicroKernel6x16(
        float[] packedA, int aOffset,
        float[] packedB, int bOffset,
        Span<float> c, int ldc,
        int cRow, int cCol,
        int kc)
    {
        // 12 accumulators: 6 rows x 2 Vector256 (16 columns)
        var c00 = Vector256<float>.Zero; var c01 = Vector256<float>.Zero;
        var c10 = Vector256<float>.Zero; var c11 = Vector256<float>.Zero;
        var c20 = Vector256<float>.Zero; var c21 = Vector256<float>.Zero;
        var c30 = Vector256<float>.Zero; var c31 = Vector256<float>.Zero;
        var c40 = Vector256<float>.Zero; var c41 = Vector256<float>.Zero;
        var c50 = Vector256<float>.Zero; var c51 = Vector256<float>.Zero;

        ref float aRef = ref MemoryMarshal.GetArrayDataReference(packedA);
        ref float bRef = ref MemoryMarshal.GetArrayDataReference(packedB);

        // Prefetch distance: load B/A cache lines PrefetchDistance iterations ahead
        // so they arrive in L1 just before they're consumed. Zen 2 L2→L1 latency is
        // ~12 cycles and each k iteration is ~4 cycles (load-port limited), so 4
        // iterations ahead covers the gap. Overrun past kc doesn't cause issues —
        // prefetch instructions are hints and out-of-bounds prefetches are ignored.
        const int PrefetchDistance = 8;
        int prefetchLimit = kc - PrefetchDistance;

        for (int p = 0; p < kc; p++)
        {
            // Prefetch B for the p+PrefetchDistance iteration (both halves of the
            // Nr=16 row, which spans 2 cache lines on 64-byte lines / 16 floats).
            if (p < prefetchLimit)
            {
                int prefBIdx = bOffset + (p + PrefetchDistance) * Nr;
                Sse.Prefetch0(
                    (void*)Unsafe.AsPointer(ref Unsafe.Add(ref bRef, prefBIdx)));
                // Prefetch A's 6 floats for the p+PrefetchDistance iteration
                // (fits in one cache line since Mr=6 floats = 24 bytes).
                int prefAIdx = aOffset + (p + PrefetchDistance) * Mr;
                Sse.Prefetch0(
                    (void*)Unsafe.AsPointer(ref Unsafe.Add(ref aRef, prefAIdx)));
            }

            // Load B row (Nr=16 = 2 vectors of 8). B regs are live across all 6 A
            // broadcasts, so they stay resident throughout the inner sequence.
            int bIdx = bOffset + p * Nr;
            var b0 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bIdx)));
            var b1 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bIdx + 8)));

            // A broadcasts: load each element right before its 2 FMAs and let the
            // register die immediately. This keeps only one A reg live at a time,
            // bringing total live ymm regs to 12 acc + 2 B + 1 A = 15, fitting in
            // AVX2's 16 ymm without spills. Prior version loaded all 6 A elements
            // up front, forcing the JIT to keep 20 regs live and spill to stack.
            int aIdx = aOffset + p * Mr;
            var a = Vector256.Create(Unsafe.Add(ref aRef, aIdx));
            c00 = Fma.MultiplyAdd(a, b0, c00); c01 = Fma.MultiplyAdd(a, b1, c01);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 1));
            c10 = Fma.MultiplyAdd(a, b0, c10); c11 = Fma.MultiplyAdd(a, b1, c11);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 2));
            c20 = Fma.MultiplyAdd(a, b0, c20); c21 = Fma.MultiplyAdd(a, b1, c21);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 3));
            c30 = Fma.MultiplyAdd(a, b0, c30); c31 = Fma.MultiplyAdd(a, b1, c31);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 4));
            c40 = Fma.MultiplyAdd(a, b0, c40); c41 = Fma.MultiplyAdd(a, b1, c41);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 5));
            c50 = Fma.MultiplyAdd(a, b0, c50); c51 = Fma.MultiplyAdd(a, b1, c51);
        }

        // Store results back to C (accumulate)
        ref float cRef = ref MemoryMarshal.GetReference(c);
        StoreAccumRow(ref cRef, cRow, cCol, ldc, c00, c01);
        StoreAccumRow(ref cRef, cRow + 1, cCol, ldc, c10, c11);
        StoreAccumRow(ref cRef, cRow + 2, cCol, ldc, c20, c21);
        StoreAccumRow(ref cRef, cRow + 3, cCol, ldc, c30, c31);
        StoreAccumRow(ref cRef, cRow + 4, cCol, ldc, c40, c41);
        StoreAccumRow(ref cRef, cRow + 5, cCol, ldc, c50, c51);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreAccumRow(
        ref float cRef, int row, int col, int ldc,
        Vector256<float> v0, Vector256<float> v1)
    {
        int offset = row * ldc + col;
        ref float target = ref Unsafe.Add(ref cRef, offset);

        var existing0 = Unsafe.ReadUnaligned<Vector256<float>>(
            ref Unsafe.As<float, byte>(ref target));
        var existing1 = Unsafe.ReadUnaligned<Vector256<float>>(
            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref target, 8)));

        Unsafe.WriteUnaligned(
            ref Unsafe.As<float, byte>(ref target),
            Avx.Add(existing0, v0));
        Unsafe.WriteUnaligned(
            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref target, 8)),
            Avx.Add(existing1, v1));
    }

    /// <summary>
    /// Scalar micro-kernel for edge cases where tile is smaller than Mr x Nr.
    /// </summary>
    [MethodImpl(HotInline)]
    private static void MicroKernelScalar(
        float[] packedA, int aOffset,
        float[] packedB, int bOffset,
        Span<float> c, int ldc,
        int cRow, int cCol,
        int kc, int mr, int nr)
    {
        for (int p = 0; p < kc; p++)
        {
            for (int i = 0; i < mr; i++)
            {
                float aVal = packedA[aOffset + p * Mr + i];
                int cIdx = (cRow + i) * ldc + cCol;
                int bIdx = bOffset + p * Nr;
                for (int j = 0; j < nr; j++)
                {
#if NET5_0_OR_GREATER
                    c[cIdx + j] = MathF.FusedMultiplyAdd(aVal, packedB[bIdx + j], c[cIdx + j]);
#else
                    c[cIdx + j] += aVal * packedB[bIdx + j];
#endif
                }
            }
        }
    }
#endif
}
