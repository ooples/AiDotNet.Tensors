using System;
using System.Buffers;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>
    /// Fused weight-only GEMM for fp16-resident inference: <c>C[m,n] = A[m,k] · B[k,n]</c> where B is a
    /// <see cref="Half"/> weight in <c>[k, n]</c> row-major. On net5.0+ the fp16→fp32 conversion happens
    /// INSIDE the GEMM's own B-packing step (<see cref="PackBFromHalf"/>), one cache-resident Kc×Nc panel
    /// at a time — so the full fp32 weight is never materialized (eliminates the cold-RAM upcast that
    /// dominated the fp16-resident forward and halves the weight bytes read). Structure mirrors the tuned
    /// fp32 <see cref="SgemmTiled"/>/<see cref="SgemmTiledParallelN"/>: A is packed once per tile and shared;
    /// each worker converts+packs only its disjoint B column-slice (so the conversion is parallel too) and
    /// runs the existing fp32 <see cref="MacroKernel"/>. Bit-equivalent (up to fp32 non-associativity) to
    /// upcast-then-Sgemm. Inference-only; no autodiff. (net471 has no intrinsic tiled kernel — falls back
    /// to a whole-weight upcast + Sgemm, correct but not memory-optimal; net471 is not the perf target.)
    /// </summary>
    public static void SgemmFp16WeightB(
        ReadOnlySpan<float> a,
        ReadOnlySpan<Half> b,
        Span<float> c,
        int m, int k, int n)
    {
        if (m <= 0 || n <= 0 || k <= 0) { c.Clear(); return; }
        if ((long)b.Length < (long)k * n)
            throw new ArgumentException($"b.Length ({b.Length}) must be >= k*n ({(long)k * n}).", nameof(b));
        if (a.Length < (long)m * k)
            throw new ArgumentException($"a.Length ({a.Length}) must be >= m*k ({(long)m * k}).", nameof(a));
        if (c.Length < (long)m * n)
            throw new ArgumentException($"c.Length ({c.Length}) must be >= m*n ({(long)m * n}).", nameof(c));

#if NET5_0_OR_GREATER
        // MacroKernel accumulates (C += panel) across the K-panel (pc) loop, so C starts cleared.
        c.Clear();

        int Mc = ChooseAdaptiveMc(m, k, n);
        int mcRounded = ((Mc + Mr - 1) / Mr) * Mr;
        int ncRounded = ((Nc + Nr - 1) / Nr) * Nr;
        int maxThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        bool parallel = UseParallelGemm && maxThreads > 1 && (long)m * k * n >= ParallelWorkThreshold;

        float[] packedABuf = ArrayPool<float>.Shared.Rent(mcRounded * Kc);
        float[]? seqPackedB = parallel ? null : ArrayPool<float>.Shared.Rent(Kc * ncRounded);
        try
        {
            for (int jc = 0; jc < n; jc += Nc)
            {
                int nc = Math.Min(Nc, n - jc);
                for (int pc = 0; pc < k; pc += Kc)
                {
                    int kc = Math.Min(Kc, k - pc);
                    int numNrBlocks = (nc + Nr - 1) / Nr;

                    if (parallel && numNrBlocks >= 2)
                    {
                        Fp16TileParallelN(a, b, c, m, k, n, jc, nc, pc, kc, numNrBlocks, maxThreads, Mc, packedABuf);
                    }
                    else
                    {
                        var pb = seqPackedB ?? throw new InvalidOperationException("sequential B buffer not allocated");
                        PackBFromHalf(b, pb, n, pc, kc, jc, nc);
                        for (int ic = 0; ic < m; ic += Mc)
                        {
                            int mc = Math.Min(Mc, m - ic);
                            PackA(a, packedABuf, k, ic, mc, pc, kc);
                            MacroKernel(packedABuf, pb, c, mc, nc, kc, n, ic, jc);
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packedABuf);
            if (seqPackedB is not null) ArrayPool<float>.Shared.Return(seqPackedB);
        }
#else
        // net471 fallback: no intrinsic tiled kernel — upcast the whole weight then Sgemm.
        float[] bf = ArrayPool<float>.Shared.Rent(k * n);
        try
        {
            for (int i = 0; i < k * n; i++) bf[i] = (float)b[i];
            Sgemm(a, new ReadOnlySpan<float>(bf, 0, k * n), c, m, k, n, 0f);
        }
        finally { ArrayPool<float>.Shared.Return(bf); }
#endif
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// N-parallel fused fp16 tile: packs A once (shared) for this (jc,pc) tile, then each worker
    /// converts+packs its own disjoint B column-slice from Half (<see cref="PackBFromHalf"/>) and runs
    /// <see cref="MacroKernel"/> over its columns. Mirrors <see cref="SgemmTiledParallelN"/> but moves the
    /// (now conversion-bearing) B-pack INSIDE the worker so the fp16→fp32 decode is parallelized.
    /// </summary>
    private static unsafe void Fp16TileParallelN(
        ReadOnlySpan<float> a, ReadOnlySpan<Half> b, Span<float> c,
        int m, int k, int n,
        int jc, int nc, int pc, int kc,
        int numNrBlocks, int maxThreads, int Mc, float[] packedABuf)
    {
        int numWorkers = Math.Min(maxThreads, numNrBlocks);
        int nrPerWorker = (numNrBlocks + numWorkers - 1) / numWorkers;

        int firstMc = Math.Min(Mc, m);
        PackA(a, packedABuf, k, 0, firstMc, pc, kc);   // shared A (first Mc block) — fp32 copy, cheap

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
            int packedNc = ((localNc + Nr - 1) / Nr) * Nr;
            packedBSlices[w] = ArrayPool<float>.Shared.Rent(kc * packedNc);
        }

        // Pin A only when extra Mc blocks need per-worker packing (m > Mc).
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
            var localPackedABuf = packedABuf;
            var localPackedBSlices = packedBSlices;
            var localSliceNcs = sliceNcs;
            var localSliceJStarts = sliceJStarts;
            float* localAPtr = aPtr;
            int localALen = a.Length;
            int localM = m, localK = k, localN = n;
            int localJc = jc, localPc = pc, localKc = kc, localFirstMc = firstMc;

            fixed (Half* bPtr = b)
            fixed (float* cPtr = c)
            {
                IntPtr ipB = (IntPtr)bPtr, ipC = (IntPtr)cPtr;
                int bLen = b.Length, cLen = c.Length;

                CpuParallelSettings.LightweightParallel(actualWorkers, workerId =>
                {
                    int workerNc = localSliceNcs[workerId];
                    int jStart = localSliceJStarts[workerId];
                    var bSpan = new ReadOnlySpan<Half>((Half*)ipB, bLen);
                    var cSpan = new Span<float>((float*)ipC, cLen);
                    float[] myB = localPackedBSlices[workerId];

                    // Convert+pack THIS worker's B column-slice from Half (parallel conversion).
                    PackBFromHalf(bSpan, myB, localN, localPc, localKc, localJc + jStart, workerNc);

                    // First Mc block uses the shared pre-packed A.
                    MacroKernel(localPackedABuf, myB, cSpan, localFirstMc, workerNc, localKc, localN, 0, localJc + jStart);

                    // Additional Mc blocks (m > Mc): each worker packs its own A panel.
                    for (int ic = Mc; ic < localM; ic += Mc)
                    {
                        int mc = Math.Min(Mc, localM - ic);
                        int packedMc = ((mc + Mr - 1) / Mr) * Mr;
                        float[] workerPackedA = ArrayPool<float>.Shared.Rent(packedMc * localKc);
                        try
                        {
                            var aSpan = new ReadOnlySpan<float>(localAPtr, localALen);
                            PackA(aSpan, workerPackedA, localK, ic, mc, localPc, localKc);
                            MacroKernel(workerPackedA, myB, cSpan, mc, workerNc, localKc, localN, ic, localJc + jStart);
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
                ArrayPool<float>.Shared.Return(packedBSlices[w]);
        }
    }

    /// <summary>
    /// Packs the panel <c>B[pc:pc+kc, jc:jc+nc]</c> of a <c>[k, n]</c> row-major <see cref="Half"/> weight
    /// into the fp32 tile layout <see cref="MacroKernel"/> expects — converting fp16→fp32 during the copy.
    /// Mirrors <see cref="PackB"/>'s no-transpose layout exactly; the only change is the Half source and
    /// the inline SIMD decode (<see cref="SimdKernels.Fp16To32Vec8"/>) for the full Nr-column rows.
    /// </summary>
    private static unsafe void PackBFromHalf(ReadOnlySpan<Half> b, float[] packed, int ldb, int pc, int kc, int jc, int nc)
    {
        int pos = 0;
        int j = 0;

#if NET8_0_OR_GREATER
        if (SimdKernels.Fp16Vec8Supported)
        {
            fixed (Half* pBBase = b)
            fixed (float* pPackedBase = packed)
            {
                ushort* usBase = (ushort*)pBBase;
                for (; j + Nr <= nc; j += Nr)
                {
                    float* pPacked = pPackedBase + pos;
                    ushort* pBCol = usBase + (long)pc * ldb + (jc + j);
                    for (int p = 0; p < kc; p++)
                    {
                        SimdKernels.Fp16To32Vec8(pBCol, pPacked);          // columns 0..7
                        SimdKernels.Fp16To32Vec8(pBCol + 8, pPacked + 8);  // columns 8..15 (Nr == 16)
                        pBCol += ldb;
                        pPacked += Nr;
                    }
                    pos += kc * Nr;
                }
            }
        }
#endif

        // Remaining full Nr-column panels (no AVX2) — scalar convert.
        for (; j + Nr <= nc; j += Nr)
        {
            for (int p = 0; p < kc; p++)
            {
                int row = pc + p;
                for (int jj = 0; jj < Nr; jj++)
                    packed[pos++] = (float)b[row * ldb + (jc + j + jj)];
            }
        }

        // Trailing columns (< Nr): convert present ones, zero-pad the rest (matches PackB).
        int remaining = nc - j;
        if (remaining > 0)
        {
            for (int p = 0; p < kc; p++)
            {
                int row = pc + p;
                for (int jj = 0; jj < remaining; jj++)
                    packed[pos++] = (float)b[row * ldb + (jc + j + jj)];
                for (int jj = remaining; jj < Nr; jj++)
                    packed[pos++] = 0f;
            }
        }
    }
#endif
}
