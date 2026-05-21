// Copyright (c) AiDotNet. All rights reserved.
// Dgemm (double-precision) SIMD kernel — FP64 companion to SimdGemm (FP32).
//
// Stage 1 of the docs/mkl-replacement/PLAN.md FP64 closure: ports the FP32
// SimdGemm primitives to FP64 with Mr×Nr=4×8 geometry (Vector256<double> =
// 4 lanes vs 8 for float, so Mr halves; Nr stays at 8 doubles = 2 Vector256
// blocks). Mc/Kc/Nc halve from the FP32 constants because FP64 is 2× bytes
// per element — preserving the same L2/L3 occupancy fractions that iter
// 31/33's adaptive Mc tuning was calibrated against.
//
// Foundation (this commit):
//   - Adaptive Mc (port of ChooseAdaptiveMc with halved constants)
//   - PackADouble / PackBDouble (port of PackA / PackB, full-Mr/Nr fast path)
//   - DgemmMicroKernel4x8 (port of DirectKernel6x16, 8 YMM accumulators)
//   - DgemmMacroKernel + DgemmTiledSequential (port of SgemmTiled core)
//   - Dgemm entry dispatches to packed-tiled when m/n/k justify packing
//     overhead, otherwise to the existing block-tile path below.
//
// Subsequent commits on this branch will add (per the plan):
//   - Parallel-M / Parallel-N / Parallel-2D dispatch + heuristic
//   - DgemmDirect small-matmul no-packing path (Stage 2)
//   - Pre-packed B cache (Stage 3)
//   - Masked edge kernels (DirectKernelMxNMasked equivalent)
//   - AVX-512 8×16 microkernel companion (Stage 7)

using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    // ─── FP64 layout constants ────────────────────────────────────────────
    //
    // Register block: 4 rows × 8 cols = 32 doubles = 8 Vector256<double>
    // accumulators. With 16 YMM registers total, 8 accumulators + 2 B-loads +
    // 1 A-broadcast = 11 live YMM, well within budget. (The FP32 6×16 path
    // uses 12 accumulators; 4×8 leaves more register headroom to interleave
    // K-loop broadcasts.)
    private const int DMr = 4;
    private const int DNr = 8;

    // Mc/Kc/Nc halved from FP32's 128/512/4096 because each double is 2×
    // the bytes of a float. Preserves the same L2/L3 occupancy fractions
    // the iter-31/33 adaptive-Mc tuning landed.
    private const int DSmallMc = 64;
    private const int DLargeMc = 96;
    private const int DKc = 256;
    private const int DNc = 2048;
    private const long DAdaptiveMcWorkThreshold = 1_500_000_000L; // 1.5G FMAs (half of FP32's 3G)
    private const long DSquareMediumWorkLow  = 32_000_000L;
    private const long DSquareMediumWorkHigh = 750_000_000L;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ChooseAdaptiveMcDouble(int m, int k, int n)
    {
        long work = (long)m * k * n;
        if (work >= DAdaptiveMcWorkThreshold) return DLargeMc;
        if (m == n && n == k && work >= DSquareMediumWorkLow && work <= DSquareMediumWorkHigh)
            return DLargeMc;
        return DSmallMc;
    }

    /// <summary>
    /// Gate: should this (m, k, n) take the packed-tiled FP64 path, or the
    /// existing inline 64-block path? Packed has setup overhead (PackA +
    /// PackB allocate + populate) that only amortizes above ~1M FMAs and
    /// when m, n are large enough to actually need multiple Mr/Nr tiles.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool DgemmShouldUsePackedTiled(int m, int k, int n)
    {
        long work = (long)m * k * n;
        if (work < 1_000_000L) return false;          // setup overhead dominates
        if (m < DMr || n < DNr) return false;         // can't fill even one tile
        if (k < 8) return false;                      // K-loop micro-kernel doesn't amortize
        return true;
    }

#if NET5_0_OR_GREATER
    // ─── Microkernel: 4 rows × 8 cols register-blocked Vector256<double> FMA ──
    //
    // Mirrors FP32 DirectKernel6x16 (lines ~1820-1895 of SimdGemm.cs) with
    // half the rows and half the per-row YMM count. 8 accumulators total:
    //   c00 c01   ← row 0, cols 0-3 and 4-7
    //   c10 c11   ← row 1
    //   c20 c21   ← row 2
    //   c30 c31   ← row 3
    //
    // The packed-A layout is [Mr-stride × kc] from PackADouble: at K-step p
    // the 4 row-elements live at packedA[p*Mr + 0..3] (interleaved across
    // rows). The packed-B layout is [kc × Nr-stride]: at K-step p the 8
    // col-elements live at packedB[p*Nr + 0..7] (contiguous within the
    // 8-col tile).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DgemmMicroKernel4x8(
        double* pPackedA, double* pPackedB,
        double* pC, int ldc,
        int kc)
    {
        var c00 = Vector256<double>.Zero; var c01 = Vector256<double>.Zero;
        var c10 = Vector256<double>.Zero; var c11 = Vector256<double>.Zero;
        var c20 = Vector256<double>.Zero; var c21 = Vector256<double>.Zero;
        var c30 = Vector256<double>.Zero; var c31 = Vector256<double>.Zero;

        // K-loop: per p, broadcast A[row, p] for each of 4 rows, then FMA
        // against the two 4-wide B tiles. Interleaved A-broadcast / FMA
        // keeps each A live for only 2 FMAs, recycling YMM registers.
        for (int p = 0; p < kc; p++)
        {
            var b0 = Avx.LoadVector256(pPackedB);       // cols 0-3
            var b1 = Avx.LoadVector256(pPackedB + 4);   // cols 4-7

            var a0 = Vector256.Create(pPackedA[0]);
            c00 = Fma.MultiplyAdd(a0, b0, c00); c01 = Fma.MultiplyAdd(a0, b1, c01);

            var a1 = Vector256.Create(pPackedA[1]);
            c10 = Fma.MultiplyAdd(a1, b0, c10); c11 = Fma.MultiplyAdd(a1, b1, c11);

            var a2 = Vector256.Create(pPackedA[2]);
            c20 = Fma.MultiplyAdd(a2, b0, c20); c21 = Fma.MultiplyAdd(a2, b1, c21);

            var a3 = Vector256.Create(pPackedA[3]);
            c30 = Fma.MultiplyAdd(a3, b0, c30); c31 = Fma.MultiplyAdd(a3, b1, c31);

            pPackedA += DMr;
            pPackedB += DNr;
        }

        // Accumulate into C (load-add-store). Caller cleared C via Dgemm.
        Avx.Store(pC + 0, Avx.Add(c00, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 4, Avx.Add(c01, Avx.LoadVector256(pC + 4)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c10, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 4, Avx.Add(c11, Avx.LoadVector256(pC + 4)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c20, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 4, Avx.Add(c21, Avx.LoadVector256(pC + 4)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c30, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 4, Avx.Add(c31, Avx.LoadVector256(pC + 4)));
    }

    // ─── Packing: A panel (mc × kc) → row-stride-Mr interleaved layout ───
    //
    // FP64 mirror of FP32 PackA. For full-Mr rows packs 4-element broadcasts
    // contiguously: per K-step p, packed[p*Mr + r] = a[ic+r, pc+p]. The
    // microkernel reads each K iter as a contiguous Mr-element load.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackADouble(
        ReadOnlySpan<double> a, double[] packed, int lda,
        int ic, int mc, int pc, int kc)
    {
        int pos = 0;
        int i = 0;
        fixed (double* aPtr = a)
        fixed (double* packedPtr = packed)
        {
            for (; i + DMr <= mc; i += DMr)
            {
                double* row0 = aPtr + (long)(ic + i + 0) * lda + pc;
                double* row1 = aPtr + (long)(ic + i + 1) * lda + pc;
                double* row2 = aPtr + (long)(ic + i + 2) * lda + pc;
                double* row3 = aPtr + (long)(ic + i + 3) * lda + pc;
                double* pp = packedPtr + pos;
                for (int p = 0; p < kc; p++)
                {
                    pp[0] = row0[0]; pp[1] = row1[0]; pp[2] = row2[0]; pp[3] = row3[0];
                    pp += DMr;
                    row0++; row1++; row2++; row3++;
                }
                pos += DMr * kc;
            }
        }

        // Edge rows (less than Mr remaining): pad with zeros so the
        // microkernel can still process the full Mr panel; the caller
        // discards the resulting rows past mc.
        int remaining = mc - i;
        if (remaining > 0)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int ii = 0; ii < remaining; ii++)
                    packed[pos++] = a[(ic + i + ii) * lda + (pc + p)];
                for (int ii = remaining; ii < DMr; ii++)
                    packed[pos++] = 0.0;
            }
        }
    }

    // ─── Packing: B panel (kc × nc) → col-stride-Nr layout ───
    //
    // FP64 mirror of FP32 PackB. For full-Nr columns: per K-step p, packed
    // contains 8 contiguous doubles = the 8 column-elements at B[pc+p, jc..jc+7].
    // The microkernel reads each K iter as two Vector256<double> loads.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackBDouble(
        ReadOnlySpan<double> b, double[] packed, int ldb,
        int pc, int kc, int jc, int nc)
    {
        int pos = 0;
        int j = 0;
        fixed (double* bPtr = b)
        fixed (double* packedPtr = packed)
        {
            for (; j + DNr <= nc; j += DNr)
            {
                double* pp = packedPtr + pos;
                for (int p = 0; p < kc; p++)
                {
                    double* bRow = bPtr + (long)(pc + p) * ldb + (jc + j);
                    // 8 doubles = two Vector256<double>
                    Avx.Store(pp + 0, Avx.LoadVector256(bRow + 0));
                    Avx.Store(pp + 4, Avx.LoadVector256(bRow + 4));
                    pp += DNr;
                }
                pos += DNr * kc;
            }
        }

        // Edge cols (less than Nr remaining): pad with zeros.
        int remaining = nc - j;
        if (remaining > 0)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int jj = 0; jj < remaining; jj++)
                    packed[pos++] = b[(pc + p) * ldb + (jc + j + jj)];
                for (int jj = remaining; jj < DNr; jj++)
                    packed[pos++] = 0.0;
            }
        }
    }

    /// <summary>
    /// Macro kernel: iterate over Mr × Nr microkernel tiles within the
    /// (mc × kc × nc) packed panel. C destination is the [ic:ic+mc, jc:jc+nc]
    /// slice of the global C, accessed via raw pointer + ldc.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DgemmMacroKernel(
        double* packedA, double* packedB,
        double* cBase, int ldc,
        int mc, int nc, int kc)
    {
        // Full Mr × Nr tiles. Edge handling (mc % Mr != 0 or nc % Nr != 0)
        // walks scalar fallback — minor cost since edges are 1-Mr-1 rows /
        // 1-Nr-1 cols on shapes large enough to take the packed path.
        int mFull = (mc / DMr) * DMr;
        int nFull = (nc / DNr) * DNr;

        for (int i = 0; i < mFull; i += DMr)
        {
            double* aPanel = packedA + (long)i * kc;
            for (int j = 0; j < nFull; j += DNr)
            {
                double* bPanel = packedB + (long)j * kc;
                double* cTile = cBase + (long)i * ldc + j;
                DgemmMicroKernel4x8(aPanel, bPanel, cTile, ldc, kc);
            }
            // Right-edge cols [nFull, nc): scalar fallback into the same
            // packed-A view (just walk K via the packed layout).
            for (int j = nFull; j < nc; j++)
            {
                for (int ii = 0; ii < DMr; ii++)
                {
                    double sum = 0.0;
                    double* aRow = aPanel + ii;
                    double* bCol = packedB + (long)(j / DNr * DNr) * kc + (j % DNr);
                    // For the edge column case (j >= nFull = nc - nc%Nr),
                    // the packed-B tile that contains column j may be a
                    // partial tile that PackBDouble padded with zeros, so
                    // reading packed[p*Nr + (j % Nr)] still yields valid
                    // contribution (zero contribution from padded lanes).
                    for (int p = 0; p < kc; p++)
                    {
                        sum += aRow[p * DMr] * bCol[p * DNr];
                    }
                    cTile_Add(cBase, ldc, i + ii, j, sum);
                }
            }
        }
        // Bottom edge rows [mFull, mc): scalar over all cols, walking the
        // packed-A edge buffer that PackADouble zero-padded.
        for (int i = mFull; i < mc; i++)
        {
            int paddedRowBase = mFull * kc + ((i - mFull));
            for (int j = 0; j < nc; j++)
            {
                double sum = 0.0;
                for (int p = 0; p < kc; p++)
                {
                    double aval = packedA[paddedRowBase + p * DMr];
                    double bval = packedB[(long)(j / DNr * DNr) * kc + p * DNr + (j % DNr)];
                    sum += aval * bval;
                }
                cTile_Add(cBase, ldc, i, j, sum);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void cTile_Add(double* cBase, int ldc, int i, int j, double sum)
    {
        cBase[(long)i * ldc + j] += sum;
    }

    /// <summary>
    /// Sequential packed-tiled DGEMM: jc → pc → ic → macro kernel. Mirrors
    /// the FP32 SgemmTiled outer loop (lines ~2253-2360) with FP64 panel
    /// sizes. C is assumed already zeroed by the caller (Dgemm clears).
    /// Uses ArrayPool for the packed scratch buffers — eliminates per-call
    /// GC pressure on training-loop hot paths.
    /// </summary>
    private static unsafe void DgemmTiledSequential(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c,
        int m, int k, int n)
    {
        int Mc = ChooseAdaptiveMcDouble(m, k, n);
        var pool = System.Buffers.ArrayPool<double>.Shared;
        var packedA = pool.Rent(Mc * DKc);
        var packedB = pool.Rent(DKc * DNc);
        try
        {
            fixed (double* aPtr = a)
            fixed (double* bPtr = b)
            fixed (double* cPtr = c)
            {
                for (int jc = 0; jc < n; jc += DNc)
                {
                    int nc = Math.Min(DNc, n - jc);
                    for (int pc = 0; pc < k; pc += DKc)
                    {
                        int kc = Math.Min(DKc, k - pc);
                        PackBDouble(b, packedB, n, pc, kc, jc, nc);

                        for (int ic = 0; ic < m; ic += Mc)
                        {
                            int mc = Math.Min(Mc, m - ic);
                            PackADouble(a, packedA, k, ic, mc, pc, kc);

                            fixed (double* paPtr = packedA)
                            fixed (double* pbPtr = packedB)
                            {
                                double* cBase = cPtr + (long)ic * n + jc;
                                DgemmMacroKernel(paPtr, pbPtr, cBase, n, mc, nc, kc);
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            pool.Return(packedA);
            pool.Return(packedB);
        }
    }

    /// <summary>
    /// Parallel-M packed-tiled DGEMM: outer ic loop parallelized via per-
    /// row-block tasks. Each task rents its own packedA buffer (per-task
    /// scratch, no contention); the single packedB panel is shared
    /// read-only across all tasks within one (jc, pc) iteration. Mirrors
    /// the FP32 SgemmTiledParallelM (lines ~2482-2562) structure.
    /// </summary>
    private static unsafe void DgemmTiledParallelM(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c,
        int m, int k, int n)
    {
        int Mc = ChooseAdaptiveMcDouble(m, k, n);
        int numRowBlocks = (m + Mc - 1) / Mc;
        var pool = System.Buffers.ArrayPool<double>.Shared;
        var packedB = pool.Rent(DKc * DNc);
        try
        {
            fixed (double* aPtr = a)
            fixed (double* bPtr = b)
            fixed (double* cPtr = c)
            {
                IntPtr aHandle = (IntPtr)aPtr;
                IntPtr bHandle = (IntPtr)bPtr;
                IntPtr cHandle = (IntPtr)cPtr;
                int mCap = m, kCap = k, nCap = n, McCap = Mc;

                for (int jc = 0; jc < n; jc += DNc)
                {
                    int nc = Math.Min(DNc, n - jc);
                    for (int pc = 0; pc < k; pc += DKc)
                    {
                        int kc = Math.Min(DKc, k - pc);
                        PackBDouble(b, packedB, n, pc, kc, jc, nc);

                        int jcCap = jc, pcCap = pc, kcCap = kc, ncCap = nc;
                        IntPtr packedBHandle;
                        fixed (double* pbPtrTmp = packedB) packedBHandle = (IntPtr)pbPtrTmp;

                        AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(
                            0, numRowBlocks, (long)m * nc * kc, (rb) =>
                        {
                            int ic = rb * McCap;
                            int mc = Math.Min(McCap, mCap - ic);
                            if (mc <= 0) return;
                            var localPool = System.Buffers.ArrayPool<double>.Shared;
                            var localPackedA = localPool.Rent(McCap * DKc);
                            try
                            {
                                PackADouble(
                                    new ReadOnlySpan<double>((double*)aHandle, mCap * kCap),
                                    localPackedA, kCap, ic, mc, pcCap, kcCap);
                                fixed (double* paPtr = localPackedA)
                                {
                                    double* pbPtr = (double*)packedBHandle;
                                    double* cBase = (double*)cHandle + (long)ic * nCap + jcCap;
                                    DgemmMacroKernel(paPtr, pbPtr, cBase, nCap, mc, ncCap, kcCap);
                                }
                            }
                            finally { localPool.Return(localPackedA); }
                        });
                    }
                }
            }
        }
        finally { pool.Return(packedB); }
    }

    // ─── Stage 3: pre-packed B cache (FP64) ────────────────────────────────
    //
    // FP32 mirror at SimdGemm.cs:102-360. Training loops re-use the same
    // weight matrix B across forward + backward + next-iter forward, etc.
    // PackB measures 18-20% of GEMM wall time on FP32 BERT FFN shapes; same
    // fraction applies to FP64. The cache uses a ConditionalWeakTable keyed
    // on the B double[] object identity so the GC still reclaims the
    // packed buffers once the source weight array becomes unreachable.

    internal sealed class PrePackedBDouble
    {
        internal int K;
        internal int N;
        internal int Kc;
        // Per (jcIter, pcIter), the packed buffer (DKc × DNc layout).
        internal double[][] PackedPanels = System.Array.Empty<double[]>();
        internal int NumJcIters;
        internal int NumPcIters;
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<double[], PrePackedBDouble> _prePackedBDoubleCache = new();
    private static readonly object _prePackedBDoubleCacheLock = new();

    /// <summary>
    /// Public API: explicit pre-pack of B. Use when the caller knows B will
    /// be reused across many GEMM calls (e.g. inference loop with frozen
    /// weights). The standard <see cref="Dgemm"/> entry auto-uses the cache
    /// when the B array identity matches; this method just primes it.
    /// </summary>
    internal static PrePackedBDouble BuildPrePackedBDouble(double[] b, int k, int n, int m)
    {
        int numJcIters = (n + DNc - 1) / DNc;
        int numPcIters = (k + DKc - 1) / DKc;
        var panels = new double[numJcIters * numPcIters][];
        var bSpan = new ReadOnlySpan<double>(b);
        for (int jcIter = 0; jcIter < numJcIters; jcIter++)
        {
            int jc = jcIter * DNc;
            int nc = Math.Min(DNc, n - jc);
            for (int pcIter = 0; pcIter < numPcIters; pcIter++)
            {
                int pc = pcIter * DKc;
                int kc = Math.Min(DKc, k - pc);
                var buf = new double[DKc * DNc];
                PackBDouble(bSpan, buf, n, pc, kc, jc, nc);
                panels[jcIter * numPcIters + pcIter] = buf;
            }
        }
        return new PrePackedBDouble
        {
            K = k, N = n, Kc = DKc,
            PackedPanels = panels,
            NumJcIters = numJcIters,
            NumPcIters = numPcIters,
        };
    }

    /// <summary>
    /// Try to retrieve a cached pre-packed B for the given array + shape,
    /// or build + cache one. Returns null if the cache lookup itself can't
    /// be performed (e.g. b is an empty array).
    /// </summary>
    private static PrePackedBDouble? GetOrAddPrePackedBDouble(double[] b, int k, int n, int m)
    {
        if (b == null || b.Length == 0) return null;
        if (_prePackedBDoubleCache.TryGetValue(b, out var existing)
            && existing.K == k && existing.N == n && existing.Kc == DKc)
            return existing;

        // Build under lock to avoid two threads packing the same B
        // concurrently. ConditionalWeakTable's AddOrUpdate is thread-safe,
        // but we don't want both threads to do the (expensive) packing work.
        lock (_prePackedBDoubleCacheLock)
        {
            if (_prePackedBDoubleCache.TryGetValue(b, out var existing2)
                && existing2.K == k && existing2.N == n && existing2.Kc == DKc)
                return existing2;
            var fresh = BuildPrePackedBDouble(b, k, n, m);
            try { _prePackedBDoubleCache.Add(b, fresh); }
            catch (ArgumentException)
            {
                // Race — another thread added between our TryGetValue and
                // Add. Re-fetch and use theirs.
                _prePackedBDoubleCache.TryGetValue(b, out fresh!);
            }
            return fresh;
        }
    }

    /// <summary>
    /// Sequential packed-tiled DGEMM that consumes a cached pre-packed B
    /// (no PackBDouble work). Saves 18-20% of wall time on training-loop
    /// hot paths where the weight matrix is reused.
    /// </summary>
    private static unsafe void DgemmTiledWithPrePackedB(
        ReadOnlySpan<double> a, PrePackedBDouble prepacked, Span<double> c,
        int m, int k, int n)
    {
        int Mc = ChooseAdaptiveMcDouble(m, k, n);
        var pool = System.Buffers.ArrayPool<double>.Shared;
        var packedA = pool.Rent(Mc * DKc);
        try
        {
            fixed (double* aPtr = a)
            fixed (double* cPtr = c)
            {
                for (int jcIter = 0; jcIter < prepacked.NumJcIters; jcIter++)
                {
                    int jc = jcIter * DNc;
                    int nc = Math.Min(DNc, n - jc);
                    for (int pcIter = 0; pcIter < prepacked.NumPcIters; pcIter++)
                    {
                        int pc = pcIter * DKc;
                        int kc = Math.Min(DKc, k - pc);
                        var packedB = prepacked.PackedPanels[jcIter * prepacked.NumPcIters + pcIter];

                        for (int ic = 0; ic < m; ic += Mc)
                        {
                            int mc = Math.Min(Mc, m - ic);
                            PackADouble(a, packedA, k, ic, mc, pc, kc);

                            fixed (double* paPtr = packedA)
                            fixed (double* pbPtr = packedB)
                            {
                                double* cBase = cPtr + (long)ic * n + jc;
                                DgemmMacroKernel(paPtr, pbPtr, cBase, n, mc, nc, kc);
                            }
                        }
                    }
                }
            }
        }
        finally { pool.Return(packedA); }
    }

    /// <summary>
    /// Public API to invalidate a cached pre-packed B (e.g. after the
    /// caller mutates the weight array in place).
    /// </summary>
    internal static void InvalidatePrePackedBDouble(double[] b)
    {
        if (b == null) return;
        lock (_prePackedBDoubleCacheLock)
            _prePackedBDoubleCache.Remove(b);
    }

    // ─── Stage 2: DgemmDirect no-packing small-matmul path ────────────────
    //
    // Mirrors FP32 SgemmDirect at SimdGemm.cs:1380-1712. For small shapes
    // (work ≤ 4M FMAs, k ≤ 512, n % Nr == 0) packing overhead exceeds the
    // perf win of cache-friendly access. The direct path walks Mr × Nr
    // tiles reading A and B directly from row-major memory — no PackA /
    // PackB calls.
    //
    // NOTE: per FP32 iter 35 / 41 reverts, we DO NOT JIT-emit the direct
    // kernel. RyuJIT inlines the C# kernel at the call site; a JIT'd
    // function-pointer dispatch added 34% regression at small k on FP32
    // and would do the same on FP64. The fat-kernel JIT (iter 42) is
    // separately deferred per Stage 6 of the plan.

    /// <summary>
    /// Direct (no-packing) Mr×Nr microkernel — reads A row-major directly
    /// from the source tensor, no packed-A layout. Same accumulator
    /// topology as <see cref="DgemmMicroKernel4x8"/> but with hoisted
    /// per-row A base pointers and the FP32-style broadcast-per-K-iter
    /// pattern. Caller cleared C, so the accumulators store on top of
    /// zero — load-add-store at the bottom is correct.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DgemmDirectKernel4x8(
        double* pA, int lda,
        double* pB, int ldb,
        double* pC, int ldc,
        int k)
    {
        var c00 = Vector256<double>.Zero; var c01 = Vector256<double>.Zero;
        var c10 = Vector256<double>.Zero; var c11 = Vector256<double>.Zero;
        var c20 = Vector256<double>.Zero; var c21 = Vector256<double>.Zero;
        var c30 = Vector256<double>.Zero; var c31 = Vector256<double>.Zero;

        // Hoist per-row A base pointers out of the K loop (avoids IMUL per iter).
        double* pA0 = pA;
        double* pA1 = pA + lda;
        double* pA2 = pA + lda * 2;
        double* pA3 = pA + lda * 3;

        for (int p = 0; p < k; p++)
        {
            var b0 = Avx.LoadVector256(pB);       // cols 0-3
            var b1 = Avx.LoadVector256(pB + 4);   // cols 4-7

            var a0 = Vector256.Create(pA0[p]);
            c00 = Fma.MultiplyAdd(a0, b0, c00); c01 = Fma.MultiplyAdd(a0, b1, c01);

            var a1 = Vector256.Create(pA1[p]);
            c10 = Fma.MultiplyAdd(a1, b0, c10); c11 = Fma.MultiplyAdd(a1, b1, c11);

            var a2 = Vector256.Create(pA2[p]);
            c20 = Fma.MultiplyAdd(a2, b0, c20); c21 = Fma.MultiplyAdd(a2, b1, c21);

            var a3 = Vector256.Create(pA3[p]);
            c30 = Fma.MultiplyAdd(a3, b0, c30); c31 = Fma.MultiplyAdd(a3, b1, c31);

            pB += ldb;
        }

        Avx.Store(pC + 0, Avx.Add(c00, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 4, Avx.Add(c01, Avx.LoadVector256(pC + 4)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c10, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 4, Avx.Add(c11, Avx.LoadVector256(pC + 4)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c20, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 4, Avx.Add(c21, Avx.LoadVector256(pC + 4)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c30, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 4, Avx.Add(c31, Avx.LoadVector256(pC + 4)));
    }

    /// <summary>
    /// Stage 2 direct path: tile-walks M and N at Mr × Nr without packing.
    /// Edge rows / cols (m % Mr != 0, n % Nr != 0) fall through to scalar
    /// fallback. Caller ensures C is zeroed and that the gate predicate
    /// <see cref="DgemmShouldUseDirect"/> returned true.
    /// </summary>
    private static unsafe void DgemmDirect(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c,
        int m, int k, int n)
    {
        int mFull = (m / DMr) * DMr;
        int nFull = (n / DNr) * DNr;

        fixed (double* pARoot = a)
        fixed (double* pBRoot = b)
        fixed (double* pCRoot = c)
        {
            // Full Mr × Nr tiles.
            for (int i = 0; i < mFull; i += DMr)
            {
                double* pARow = pARoot + (long)i * k;
                double* pCRow = pCRoot + (long)i * n;
                for (int j = 0; j < nFull; j += DNr)
                {
                    DgemmDirectKernel4x8(pARow, k, pBRoot + j, n, pCRow + j, n, k);
                }
                // Right-edge cols [nFull, n): scalar fallback.
                for (int j = nFull; j < n; j++)
                {
                    for (int ii = 0; ii < DMr; ii++)
                    {
                        double sum = 0.0;
                        for (int p = 0; p < k; p++)
                            sum += pARow[(long)ii * k + p] * pBRoot[(long)p * n + j];
                        pCRow[(long)ii * n + j] += sum;
                    }
                }
            }
            // Bottom-edge rows [mFull, m): scalar over all cols.
            for (int i = mFull; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sum = 0.0;
                    for (int p = 0; p < k; p++)
                        sum += pARoot[(long)i * k + p] * pBRoot[(long)p * n + j];
                    pCRoot[(long)i * n + j] += sum;
                }
            }
        }
    }

    /// <summary>
    /// Gate for the Stage 2 DgemmDirect path. Matches FP32 SgemmDirect gate
    /// scaled for FP64: work ≤ 4M FMAs AND k ≤ 512 AND n%4 == 0 (the
    /// microkernel does 4-wide AVX2 loads on B). M edges are accepted (the
    /// direct path handles them via scalar fallback).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool DgemmShouldUseDirect(int m, int k, int n)
    {
        long work = (long)m * k * n;
        if (work > 4_000_000L) return false;
        if (k > 512) return false;
        if (n % 4 != 0) return false;
        // Need at least one full Mr × Nr tile to fire the SIMD kernel.
        if (m < DMr || n < DNr) return false;
        return true;
    }

    /// <summary>
    /// Gate: parallel-M when there's at least 2 row blocks per worker and
    /// total work justifies parallel dispatch overhead. Mirror of the FP32
    /// SgemmTiled parallel gate (SimdGemm.cs:924-945 dispatch).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool DgemmShouldParallelize(int m, int k, int n)
    {
        if (!UseParallelGemm) return false;
        long work = (long)m * k * n;
        if (work < 4_000_000L) return false;     // small-shape parallel overhead dominates
        int procs = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        if (procs <= 1) return false;
        int Mc = ChooseAdaptiveMcDouble(m, k, n);
        int numRowBlocks = (m + Mc - 1) / Mc;
        // Need at least 2 row blocks per worker to amortize dispatch.
        return numRowBlocks >= 2;
    }
#endif

    /// <summary>
    /// C = A · B where A is [m, k], B is [k, n], C is [m, n], all in
    /// row-major order. C is cleared before computation.
    /// Uses AVX2 Vector256&lt;double&gt; FMA when available, scalar
    /// fallback otherwise.
    /// </summary>
    internal static void Dgemm(
        ReadOnlySpan<double> a,
        ReadOnlySpan<double> b,
        Span<double> c,
        int m, int k, int n)
    {
        if (m <= 0 || n <= 0) return;
        c.Clear();
        if (k <= 0) return;

#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            // Stage 2: direct (no-packing) path for small shapes — saves
            // PackA + PackB setup cost when work is small and access
            // patterns are already cache-friendly.
            if (DgemmShouldUseDirect(m, k, n))
            {
                DgemmDirect(a, b, c, m, k, n);
                return;
            }
            // Stage 1 packed-tiled path: amortizes packing overhead on
            // shapes large enough to benefit (per DgemmShouldUsePackedTiled
            // gate). Small shapes fall through to the existing inline 64-
            // block path which has no packing setup cost.
            if (DgemmShouldUsePackedTiled(m, k, n))
            {
                if (DgemmShouldParallelize(m, k, n))
                    DgemmTiledParallelM(a, b, c, m, k, n);
                else
                    DgemmTiledSequential(a, b, c, m, k, n);
                return;
            }
            DgemmAvx2(a, b, c, m, k, n, allowParallel: true);
            return;
        }
#endif
        DgemmScalar(a, b, c, m, k, n);
    }

    /// <summary>
    /// Stage 3 cached-B variant: when the caller can pass B as a double[]
    /// (not a Span) and the same array is reused across many calls, the
    /// pre-packed B cache skips PackBDouble entirely on hits. C is cleared
    /// before computation. Falls through to <see cref="Dgemm"/> for shapes
    /// outside the packed-tiled gate.
    /// </summary>
    internal static void DgemmWithCachedB(
        ReadOnlySpan<double> a,
        double[] b,
        Span<double> c,
        int m, int k, int n)
    {
        if (m <= 0 || n <= 0) return;
        c.Clear();
        if (k <= 0) return;

#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported
            && DgemmShouldUsePackedTiled(m, k, n))
        {
            var prepacked = GetOrAddPrePackedBDouble(b, k, n, m);
            if (prepacked != null)
            {
                DgemmTiledWithPrePackedB(a, prepacked, c, m, k, n);
                return;
            }
        }
#endif
        // Fall through: caller's expectations are satisfied by the
        // standard Dgemm path (no cache, no perf loss vs the un-Cached
        // entry).
        Dgemm(a, new ReadOnlySpan<double>(b), c, m, k, n);
    }

    /// <summary>
    /// Sequential variant — same kernel as <see cref="Dgemm"/> but never
    /// spawns inner parallelism. Use when called from inside an already-
    /// parallel batch loop (e.g. BatchMatMul over batch slices) so the
    /// thread pool isn't oversubscribed by nested Parallel.For dispatches.
    /// Companion to <see cref="SgemmSequential"/>; same rationale.
    /// </summary>
    internal static void DgemmSequential(
        ReadOnlySpan<double> a,
        ReadOnlySpan<double> b,
        Span<double> c,
        int m, int k, int n)
    {
        if (m <= 0 || n <= 0) return;
        c.Clear();
        if (k <= 0) return;

#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            if (DgemmShouldUseDirect(m, k, n))
            {
                DgemmDirect(a, b, c, m, k, n);
                return;
            }
            if (DgemmShouldUsePackedTiled(m, k, n))
            {
                DgemmTiledSequential(a, b, c, m, k, n);
                return;
            }
            DgemmAvx2(a, b, c, m, k, n, allowParallel: false);
            return;
        }
#endif
        DgemmScalar(a, b, c, m, k, n);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Pre-existing inline 64-block kernel (kept as fallback for small shapes
    // where the packed-tiled path's setup overhead dominates). This is the
    // original SimdGemmDouble path that shipped pre-Stage-1.
    // ─────────────────────────────────────────────────────────────────────
    private const int DgemmBlockSize = 64;

#if NET5_0_OR_GREATER
    private static unsafe void DgemmAvx2(
        ReadOnlySpan<double> a,
        ReadOnlySpan<double> b,
        Span<double> c,
        int m, int k, int n,
        bool allowParallel)
    {
        int numRowBlocks = (m + DgemmBlockSize - 1) / DgemmBlockSize;
        int numColBlocks = (n + DgemmBlockSize - 1) / DgemmBlockSize;
        bool doParallel = allowParallel && (long)m * n >= 16384 && Environment.ProcessorCount > 1;

        fixed (double* aPtr0 = a)
        fixed (double* bPtr0 = b)
        fixed (double* cPtr0 = c)
        {
            IntPtr aHandle = (IntPtr)aPtr0;
            IntPtr bHandle = (IntPtr)bPtr0;
            IntPtr cHandle = (IntPtr)cPtr0;

            void Tile(int iiBlock, int jjBlock)
            {
                double* aP = (double*)aHandle;
                double* bP = (double*)bHandle;
                double* cP = (double*)cHandle;

                int iStart = iiBlock * DgemmBlockSize;
                int iEnd = Math.Min(iStart + DgemmBlockSize, m);
                int jStart = jjBlock * DgemmBlockSize;
                int jEnd = Math.Min(jStart + DgemmBlockSize, n);
                int nLen = jEnd - jStart;

                for (int kk = 0; kk < k; kk += DgemmBlockSize)
                {
                    int kLen = Math.Min(DgemmBlockSize, k - kk);
                    for (int i = iStart; i < iEnd; i++)
                    {
                        double* aRow = aP + (long)i * k + kk;
                        double* cRow = cP + (long)i * n + jStart;
                        for (int kIndex = 0; kIndex < kLen; kIndex++)
                        {
                            double aik = aRow[kIndex];
                            if (aik == 0.0) continue;
                            Vector256<double> vAik = Vector256.Create(aik);
                            double* bRow = bP + (long)(kk + kIndex) * n + jStart;
                            int j = 0;
                            for (; j <= nLen - 4; j += 4)
                            {
                                var vB = Avx.LoadVector256(bRow + j);
                                var vC = Avx.LoadVector256(cRow + j);
                                vC = Fma.MultiplyAdd(vB, vAik, vC);
                                Avx.Store(cRow + j, vC);
                            }
                            for (; j < nLen; j++)
                                cRow[j] += aik * bRow[j];
                        }
                    }
                }
            }

            if (doParallel)
            {
                int total = numRowBlocks * numColBlocks;
                int procs = Environment.ProcessorCount;
                if (numRowBlocks * 2 < procs && numColBlocks > 1)
                {
                    AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, total, (long)m * n * k,
                        bi => Tile(bi / numColBlocks, bi % numColBlocks));
                }
                else
                {
                    AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, numRowBlocks, (long)m * n * k, ii =>
                    {
                        for (int jj = 0; jj < numColBlocks; jj++) Tile(ii, jj);
                    });
                }
            }
            else
            {
                for (int ii = 0; ii < numRowBlocks; ii++)
                for (int jj = 0; jj < numColBlocks; jj++)
                    Tile(ii, jj);
            }
        }
    }
#endif

    /// <summary>Pre-AVX2 scalar fallback — row-wise FMA accumulation.</summary>
    private static void DgemmScalar(
        ReadOnlySpan<double> a,
        ReadOnlySpan<double> b,
        Span<double> c,
        int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            int aRow = i * k;
            int cRow = i * n;
            for (int kk = 0; kk < k; kk++)
            {
                double aik = a[aRow + kk];
                if (aik == 0.0) continue;
                int bRow = kk * n;
                for (int j = 0; j < n; j++)
                    c[cRow + j] += aik * b[bRow + j];
            }
        }
    }
}
