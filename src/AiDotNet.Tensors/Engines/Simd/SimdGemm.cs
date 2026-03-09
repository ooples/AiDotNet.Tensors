using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
    private const int Mc = 256;  // Panel height for A (fits in L2)
    private const int Kc = 512;  // Panel depth (fits in L1)
    private const int Nc = 4096; // Panel width for B (fits in L3)

    // Micro-kernel register block: 6 rows x 16 columns
    // 6 rows * 16 cols = 96 floats = 12 Vector256<float> accumulators
    private const int Mr = 6;
    private const int Nr = 16;

    /// <summary>
    /// Computes C = A * B where A is [m,k], B is [k,n], C is [m,n].
    /// All matrices are in row-major order. C is cleared before computation.
    /// </summary>
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
    /// Computes C += A * B (accumulates into C without clearing).
    /// </summary>
    public static void SgemmAdd(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && m >= Mr && n >= Nr)
        {
            SgemmTiled(a, b, c, m, k, n);
            return;
        }
#endif
        SgemmScalar(a, b, c, m, k, n);
    }

    /// <summary>
    /// Scalar GEMM fallback for platforms without AVX2/FMA.
    /// </summary>
    private static void SgemmScalar(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
        for (int i = 0; i < m; i++)
        {
            int aRowBase = i * k;
            int cRowBase = i * n;
            for (int p = 0; p < k; p++)
            {
                float aip = a[aRowBase + p];
                int bRowBase = p * n;
                for (int j = 0; j < n; j++)
                {
                    c[cRowBase + j] += aip * b[bRowBase + j];
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
    private static void SgemmTiled(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
        int packedASize = Mc * Kc;
        int packedBSize = Kc * Nc;
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

                    int numRowBlocks = (m + Mc - 1) / Mc;
                    int numNrBlocks = (nc + Nr - 1) / Nr;
                    int maxThreads = Environment.ProcessorCount;

                    // N-parallel: preferred when N is wide enough (uses more workers than M-parallel)
                    bool useParallelN = numNrBlocks >= 4 && maxThreads > 1;
                    // M-parallel: fallback for very tall, narrow matrices where N-parallel isn't viable
                    bool useParallelM = !useParallelN && numRowBlocks >= 2 && maxThreads > 1;

                    if (useParallelN)
                    {
                        SgemmTiledParallelN(a, b, c, m, k, n, jc, nc, pc, kc,
                            numNrBlocks, maxThreads, packedABuf);
                    }
                    else if (useParallelM)
                    {
                        SgemmTiledParallelM(a, b, c, m, k, n, jc, nc, pc, kc,
                            numRowBlocks, packedBBuf);
                    }
                    else
                    {
                        // Sequential path
                        PackB(b, packedBBuf, n, pc, kc, jc, nc);

                        for (int ic = 0; ic < m; ic += Mc)
                        {
                            int mc = Math.Min(Mc, m - ic);
                            PackA(a, packedABuf, k, ic, mc, pc, kc);
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
    /// </summary>
    private static void SgemmTiledParallelN(
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
        // For m > Mc each worker will re-pack A for additional blocks
        int firstMc = Math.Min(Mc, m);
        PackA(a, packedABuf, k, 0, firstMc, pc, kc);

        // Pre-pack B slices for each worker (sequential, uses B span directly)
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

            packedBSlices[w] = ArrayPool<float>.Shared.Rent(kc * localNc);
            PackB(b, packedBSlices[w], n, pc, kc, jc + jStart, localNc);
        }

        // Copy C to array for closure capture (workers write to non-overlapping columns)
        int cLen = c.Length;
        float[] cArr = ArrayPool<float>.Shared.Rent(cLen);
        c.CopyTo(cArr.AsSpan(0, cLen));

        // Copy A to array for closure capture (only if m > Mc, otherwise packedABuf is shared read-only)
        float[]? aArr = null;
        if (m > Mc)
        {
            aArr = ArrayPool<float>.Shared.Rent(a.Length);
            a.CopyTo(aArr);
        }

        try
        {
            // Capture locals for closure
            var localPackedABuf = packedABuf;
            var localPackedBSlices = packedBSlices;
            var localSliceNcs = sliceNcs;
            var localSliceJStarts = sliceJStarts;
            var localAArr = aArr;
            int localM = m, localK = k, localN = n;
            int localJc = jc, localPc = pc, localKc = kc;
            int localFirstMc = firstMc;

            Helpers.CpuParallelSettings.LightweightParallel(actualWorkers, workerId =>
            {
                int localNc = localSliceNcs[workerId];
                int jStart = localSliceJStarts[workerId];

                // First Mc block: use shared pre-packed A
                MacroKernel(localPackedABuf, localPackedBSlices[workerId],
                    cArr.AsSpan(), localFirstMc, localNc, localKc, localN, 0, localJc + jStart);

                // Additional Mc blocks (if m > Mc)
                for (int ic = Mc; ic < localM; ic += Mc)
                {
                    int mc = Math.Min(Mc, localM - ic);
                    float[] workerPackedA = ArrayPool<float>.Shared.Rent(mc * localKc);
                    try
                    {
                        PackA(localAArr.AsSpan(), workerPackedA, localK, ic, mc, localPc, localKc);
                        MacroKernel(workerPackedA, localPackedBSlices[workerId],
                            cArr.AsSpan(), mc, localNc, localKc, localN, ic, localJc + jStart);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(workerPackedA);
                    }
                }
            });

            cArr.AsSpan(0, cLen).CopyTo(c);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(cArr);
            if (aArr is not null) ArrayPool<float>.Shared.Return(aArr);
            for (int w = 0; w < actualWorkers; w++)
            {
                ArrayPool<float>.Shared.Return(packedBSlices[w]);
            }
        }
    }

    /// <summary>
    /// M-dimension parallel GEMM: splits rows across workers.
    /// Ideal for tall matrices (m >= 512). Uses ArrayPool and LightweightParallel
    /// for reduced allocation and dispatch overhead.
    /// </summary>
    private static void SgemmTiledParallelM(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n,
        int jc, int nc, int pc, int kc,
        int numRowBlocks, float[] packedBBuf)
    {
        // Pack B once (shared across all M workers, read-only)
        PackB(b, packedBBuf, n, pc, kc, jc, nc);

        // Copy A and C to arrays for closure capture
        int aLen = a.Length;
        int cLen = c.Length;
        float[] aArr = ArrayPool<float>.Shared.Rent(aLen);
        float[] cArr = ArrayPool<float>.Shared.Rent(cLen);
        a.CopyTo(aArr);
        c.CopyTo(cArr.AsSpan(0, cLen));

        try
        {
            var localPackedBBuf = packedBBuf;
            int localM = m, localK = k, localN = n;
            int localJc = jc, localNc = nc, localPc = pc, localKc = kc;

            Helpers.CpuParallelSettings.LightweightParallel(numRowBlocks, iiBlock =>
            {
                int ic = iiBlock * Mc;
                int mc = Math.Min(Mc, localM - ic);

                float[] localPackedA = ArrayPool<float>.Shared.Rent(mc * localKc);
                try
                {
                    PackA(aArr.AsSpan(), localPackedA, localK, ic, mc, localPc, localKc);
                    MacroKernel(localPackedA, localPackedBBuf, cArr.AsSpan(), mc, localNc, localKc, localN, ic, localJc);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(localPackedA);
                }
            });

            cArr.AsSpan(0, cLen).CopyTo(c);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(aArr);
            ArrayPool<float>.Shared.Return(cArr);
        }
    }

    /// <summary>
    /// Pack A[ic:ic+mc, pc:pc+kc] into row-panel format for sequential access in micro-kernel.
    /// Layout: groups of Mr rows, each stored as Mr x kc contiguous block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PackA(ReadOnlySpan<float> a, float[] packed, int lda, int ic, int mc, int pc, int kc)
    {
        int pos = 0;
        int i = 0;

        // Full Mr-row panels
        for (; i + Mr <= mc; i += Mr)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int ii = 0; ii < Mr; ii++)
                {
                    packed[pos++] = a[(ic + i + ii) * lda + pc + p];
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
                    packed[pos++] = a[(ic + i + ii) * lda + pc + p];
                }
                // Pad with zeros
                for (int ii = remaining; ii < Mr; ii++)
                {
                    packed[pos++] = 0;
                }
            }
        }
    }

    /// <summary>
    /// Pack B[pc:pc+kc, jc:jc+nc] into column-panel format.
    /// Layout: groups of Nr columns, each stored as kc x Nr contiguous block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PackB(ReadOnlySpan<float> b, float[] packed, int ldb, int pc, int kc, int jc, int nc)
    {
        int pos = 0;
        int j = 0;

        // Full Nr-column panels
        for (; j + Nr <= nc; j += Nr)
        {
            for (int p = 0; p < kc; p++)
            {
                int bRow = (pc + p) * ldb + jc + j;
                for (int jj = 0; jj < Nr; jj++)
                {
                    packed[pos++] = b[bRow + jj];
                }
            }
        }

        // Remaining columns (less than Nr)
        int remaining = nc - j;
        if (remaining > 0)
        {
            for (int p = 0; p < kc; p++)
            {
                int bRow = (pc + p) * ldb + jc + j;
                for (int jj = 0; jj < remaining; jj++)
                {
                    packed[pos++] = b[bRow + jj];
                }
                for (int jj = remaining; jj < Nr; jj++)
                {
                    packed[pos++] = 0;
                }
            }
        }
    }

    /// <summary>
    /// Macro-kernel: iterate over packed panels with Mr x Nr micro-kernel tiles.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MacroKernel(
        float[] packedA,
        float[] packedB,
        Span<float> c,
        int mc, int nc, int kc,
        int ldc, int icOffset, int jcOffset)
    {
        int nrBlocks = (nc + Nr - 1) / Nr;
        int mrBlocks = (mc + Mr - 1) / Mr;

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
                    // Full micro-kernel
                    MicroKernel6x16(
                        packedA, aPanelOffset,
                        packedB, bPanelOffset,
                        c, ldc,
                        icOffset + iLocal, jcOffset + jLocal,
                        kc);
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
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MicroKernel6x16(
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

        for (int p = 0; p < kc; p++)
        {
            // Load B row (Nr=16 = 2 vectors of 8)
            int bIdx = bOffset + p * Nr;
            var b0 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bIdx)));
            var b1 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bIdx + 8)));

            // Load A column (Mr=6 values)
            int aIdx = aOffset + p * Mr;
            var a0 = Vector256.Create(Unsafe.Add(ref aRef, aIdx));
            var a1 = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 1));
            var a2 = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 2));
            var a3 = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 3));
            var a4 = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 4));
            var a5 = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 5));

            // FMA: C[i,j] += A[i,p] * B[p,j]
            c00 = Fma.MultiplyAdd(a0, b0, c00); c01 = Fma.MultiplyAdd(a0, b1, c01);
            c10 = Fma.MultiplyAdd(a1, b0, c10); c11 = Fma.MultiplyAdd(a1, b1, c11);
            c20 = Fma.MultiplyAdd(a2, b0, c20); c21 = Fma.MultiplyAdd(a2, b1, c21);
            c30 = Fma.MultiplyAdd(a3, b0, c30); c31 = Fma.MultiplyAdd(a3, b1, c31);
            c40 = Fma.MultiplyAdd(a4, b0, c40); c41 = Fma.MultiplyAdd(a4, b1, c41);
            c50 = Fma.MultiplyAdd(a5, b0, c50); c51 = Fma.MultiplyAdd(a5, b1, c51);
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
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
                    c[cIdx + j] += aVal * packedB[bIdx + j];
                }
            }
        }
    }
#endif
}
