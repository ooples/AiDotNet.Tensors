using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Pack-A-Only strategy — packs A into vpanel layout but reads B directly
/// from caller-supplied memory. Used by <see cref="BlasManaged.Gemm{T}"/>
/// when packing B is not worthwhile (typically: B fits in L1 already, or
/// the K depth is too small for pack amortization to pay off).
///
/// <para>
/// This Phase B implementation supports only transB=false (B stored
/// row-major [K, N]). transB=true is handled by <see cref="PackBothStrategy"/>
/// (which packs B and absorbs the transpose) or <see cref="StreamingStrategy"/>.
/// </para>
///
/// <para>
/// <b>Sub-issue B (#370) task B.3:</b> when <see cref="AxisSelector"/> picks
/// <see cref="ParallelismAxis.N"/>, the outer <c>jc</c> loop is partitioned
/// across threads. Pack-A is computed once per (pc, ic) outer block and shared
/// read-only inside the parallel region; each thread writes a disjoint column
/// slice of C — bit-exact across thread counts.
/// </para>
/// </summary>
internal static class PackAOnlyStrategy
{
    /// <summary>
    /// Compute C += op(A) · B with no B-side pack. C is read-modify-write.
    /// </summary>
    public static void Run<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb,
        Span<T> c, int ldc,
        int m, int n, int k,
        int mc, int kc,
        int mr, int nr,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (mr <= 0 || nr <= 0)
            throw new ArgumentException($"mr and nr must be positive (mr={mr}, nr={nr}).");

        // M/N tail handling (#409 S.4 follow-through). The strided microkernels operate
        // on full mr×nr tiles, so the aligned interior [0,mAligned)×[0,nAligned) runs the
        // fast packed path and the remaining tail rows/columns are computed by a correct
        // scalar edge pass. This makes the strategy robust to ANY (m, n) regardless of how
        // the caller chose its tile width — e.g. BlasManaged.Gemm can compute its split
        // with mr=6 (Fast 6×8) while this strategy runs with mr=4 (a concurrent flip of
        // the process-global BlasProvider.IsDeterministicMode, or AVX-512-vs-AVX2 hardware
        // differences, change PickMicrokernelTile's result between the split and dispatch),
        // which previously threw NotSupportedException for the misaligned remainder.
        int mAligned = m - (m % mr);
        int nAligned = n - (n % nr);

        if (mAligned > 0 && nAligned > 0)
        {
            int procs = options.NumThreads > 0 ? options.NumThreads : Environment.ProcessorCount;
            if (options.NumThreads < 0) procs = 1;
            bool isDeterministic = BlasProvider.IsDeterministicMode;

            var axis = AxisSelector.Select(mAligned, nAligned, k, mr, nr, procs, isDeterministic);
            if (axis == ParallelismAxis.N && nAligned >= procs * nr * 2)
                RunNParallel(a, lda, transA, b, ldb, c, ldc, mAligned, nAligned, k, mc, kc, mr, nr, in options, procs);
            else
                RunSerial(a, lda, transA, b, ldb, c, ldc, mAligned, nAligned, k, mc, kc, mr, nr, in options);
        }

        // M-tail: rows [mAligned, m) over ALL columns [0, n).
        if (m > mAligned)
            EdgeGemm(a, lda, transA, b, ldb, c, ldc, mAligned, m, 0, n, k);
        // N-tail: the aligned rows [0, mAligned) over the remaining columns [nAligned, n).
        if (n > nAligned && mAligned > 0)
            EdgeGemm(a, lda, transA, b, ldb, c, ldc, 0, mAligned, nAligned, n, k);
    }

    /// <summary>
    /// Correct scalar edge pass for tail tiles the strided microkernels can't cover
    /// (partial mr rows or partial nr columns). Computes <c>C[i,j] += Σₚ A(i,p)·B(p,j)</c>
    /// over the half-open row/column rectangle, accumulating in the element type to match
    /// the microkernel's precision. Only float/double are supported (same set as
    /// <see cref="DispatchStridedMicrokernel{T}"/>).
    /// </summary>
    private static void EdgeGemm<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb,
        Span<T> c, int ldc,
        int rowStart, int rowEnd, int colStart, int colEnd, int k) where T : unmanaged
    {
        if (rowEnd <= rowStart || colEnd <= colStart) return;

        if (typeof(T) == typeof(double))
        {
            var ad = MemoryMarshal.Cast<T, double>(a);
            var bd = MemoryMarshal.Cast<T, double>(b);
            var cd = MemoryMarshal.Cast<T, double>(c);
            for (int i = rowStart; i < rowEnd; i++)
                for (int j = colStart; j < colEnd; j++)
                {
                    double acc = 0;
                    for (int p = 0; p < k; p++)
                        acc += (transA ? ad[p * lda + i] : ad[i * lda + p]) * bd[p * ldb + j];
                    cd[i * ldc + j] += acc;
                }
            return;
        }
        if (typeof(T) == typeof(float))
        {
            var af = MemoryMarshal.Cast<T, float>(a);
            var bf = MemoryMarshal.Cast<T, float>(b);
            var cf = MemoryMarshal.Cast<T, float>(c);
            for (int i = rowStart; i < rowEnd; i++)
                for (int j = colStart; j < colEnd; j++)
                {
                    float acc = 0f;
                    for (int p = 0; p < k; p++)
                        acc += (transA ? af[p * lda + i] : af[i * lda + p]) * bf[p * ldb + j];
                    cf[i * ldc + j] += acc;
                }
            return;
        }
        throw new NotSupportedException($"PackAOnlyStrategy edge does not support T={typeof(T).Name}.");
    }

    /// <summary>
    /// Partition N tiles across <paramref name="procs"/> threads. Each thread
    /// runs the serial body on its sub-range of columns. Pack-A is recomputed
    /// per (pc, ic) block inside each thread — this matches the serial behavior
    /// exactly and avoids cross-thread pack-A sharing complexity (pack-A is cheap
    /// relative to the inner microkernel work).
    /// </summary>
    private static void RunNParallel<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb,
        Span<T> c, int ldc,
        int m, int n, int k,
        int mc, int kc,
        int mr, int nr,
        in BlasOptions<T> options,
        int procs) where T : unmanaged
    {
        // Pre-snapshot options because ref struct fields can't be captured into a lambda.
        // We discard PackedA in the parallel path — sharing the pre-pack across threads is
        // safe (it's read-only) but the per-thread Run will re-route to its own packing.
        var packingMode = options.PackingMode;
        var packedA = options.PackedA;
        var packedB = options.PackedB;
        // Workspace can't be shared across threads — drop it; threads use [ThreadStatic] pools.
        // NumThreads is overridden to -1 inside the worker so the sub-call doesn't recurse.

        int nTilesTotal = n / nr;
        int procsClamped = Math.Min(procs, nTilesTotal);

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
                int ldaLocal = lda, ldbLocal = ldb, ldcLocal = ldc;
                int mLocal = m, kLocal = k, mcLocal = mc, kcLocal = kc, mrLocal = mr, nrLocal = nr;
                bool taLocal = transA;

                CpuParallelSettings.ParallelForRegion(procsClamped, p =>
                {
                    int tileStart = (int)(((long)p * nTilesTotal) / procsClamped);
                    int tileEnd = (int)(((long)(p + 1) * nTilesTotal) / procsClamped);
                    int nStart = tileStart * nrLocal;
                    int nLocal = (tileEnd - tileStart) * nrLocal;
                    if (nLocal <= 0) return;

                    // B[K, N] row-major: column nStart starts at b[nStart]; ldb unchanged.
                    var aSpan = new ReadOnlySpan<T>(aLocal, aLen);
                    var bSpan = new ReadOnlySpan<T>(bLocal + nStart, bLen - nStart);
                    var cSpan = new Span<T>(cLocal + nStart, cLen - nStart);

                    // Sub-call: force single-thread so it routes to RunSerial without
                    // re-recursing into RunNParallel.
                    var subOpts = new BlasOptions<T>
                    {
                        PackingMode = packingMode,
                        PackedA = packedA,
                        PackedB = packedB,
                        NumThreads = -1,
                        // Workspace, Epilogue, AutotuneKey, MaxJitCacheBytes, Mode intentionally
                        // not propagated: epilogue is applied once post-strategy by BlasManaged.Gemm,
                        // and workspace can't be split across threads.
                    };

                    RunSerial(aSpan, ldaLocal, taLocal,
                              bSpan, ldbLocal,
                              cSpan, ldcLocal,
                              mLocal, nLocal, kLocal,
                              mcLocal, kcLocal, mrLocal, nrLocal,
                              in subOpts);
                });
            }
        }
    }

    /// <summary>Serial body — original Phase B implementation.</summary>
    private static void RunSerial<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb,
        Span<T> c, int ldc,
        int m, int n, int k,
        int mc, int kc,
        int mr, int nr,
        in BlasOptions<T> options) where T : unmanaged
    {
        int elemSize = Unsafe.SizeOf<T>();
        int packABytes = mc * kc * elemSize;

        var carver = new WorkspaceCarver(options.Workspace);

        Span<byte> packABytesSpan = carver.HasWorkspace ? carver.TryCarve(packABytes) : Span<byte>.Empty;
        if (packABytesSpan.IsEmpty) packABytesSpan = ArenaIntegration.TryRentBytes(packABytes);
        if (packABytesSpan.IsEmpty) packABytesSpan = PerThreadPool.Current.RentPackA(packABytes);

        Span<T> packA = MemoryMarshal.Cast<byte, T>(packABytesSpan).Slice(0, mc * kc);

        // Sub-E (#373): detect multi-panel PackedA layout so the (ic, pc) loop
        // below picks the right tile slice rather than always reading offset 0.
        // PackMr gate: the panel stripes are mr-interleaved at pack time; a
        // consumer running a different active mr would read them at the wrong
        // stride (see WeightPackHandle.PackMr). Mismatch -> live pack below.
        bool multiPanelA = options.PackedA != null
            && WeightPackCache.IsCacheCurrent(options.PackedA)
            && options.PackedA.PackMr == mr
            && options.PackedA.TilingMatches(mc, kc)
            && options.PackedA.FullM >= m
            && options.PackedA.FullK >= k;

        for (int pc = 0; pc < k; pc += kc)
        {
            int effectiveKc = Math.Min(kc, k - pc);
            int pcIdx = pc / kc;

            for (int ic = 0; ic < m; ic += mc)
            {
                int effectiveMc = Math.Min(mc, m - ic);
                int icIdx = ic / mc;

                int effectivePackABytes = effectiveMc * effectiveKc * elemSize;
                bool packAFromPrePack = false;
                Span<T> activePackA = packA;

                if (multiPanelA)
                {
                    var tileBytes = options.PackedA!.GetTileSlice(icIdx, pcIdx);
                    if (!tileBytes.IsEmpty && tileBytes.Length >= effectivePackABytes)
                    {
                        activePackA = MemoryMarshal.Cast<byte, T>(tileBytes.Slice(0, effectivePackABytes));
                        packAFromPrePack = true;
                    }
                }
                else if (options.PackedA != null && WeightPackCache.IsCacheCurrent(options.PackedA)
                    && options.PackedA.PackMr == mr
                    && options.PackedA.PackedBuffer.Length >= effectivePackABytes)
                {
                    // Legacy single-panel pre-pack: only valid when the entire weight fits one tile.
                    // PackMr gate as above — the buffer is striped at the PACK-time mr;
                    // consuming with a different active mr reads the wrong stride.
                    activePackA = MemoryMarshal.Cast<byte, T>(options.PackedA.PackedBuffer.AsSpan(0, effectivePackABytes));
                    packAFromPrePack = true;
                }

                if (packAFromPrePack)
                {
                    BlasManagedStatsTracker.IncrementPackCacheHit();
                }
                else
                {
                    if (options.PackedA != null) BlasManagedStatsTracker.IncrementPackCacheMiss();
                    int aSliceOffset = transA ? pc * lda + ic : ic * lda + pc;
                    Avx2Pack.PackA<T>(
                        a: a.Slice(aSliceOffset), lda, transA,
                        packed: activePackA.Slice(0, effectiveMc * effectiveKc),
                        mc: effectiveMc, kc: effectiveKc, mr);
                }

                for (int jc = 0; jc < n; jc += nr)
                {
                    if (jc + nr > n) break;

                    for (int ir = 0; ir < effectiveMc; ir += mr)
                    {
                        int packedAStripeOff = (ir / mr) * effectiveKc * mr;
                        int cTileOff = (ic + ir) * ldc + jc;
                        int bSliceOffset = pc * ldb + jc;

                        DispatchStridedMicrokernel<T>(
                            activePackA.Slice(packedAStripeOff, effectiveKc * mr),
                            b.Slice(bSliceOffset), ldb,
                            c.Slice(cTileOff), ldc, effectiveKc,
                            mr, nr);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Routes to the appropriate strided-B microkernel matching T and the
    /// Mr×Nr tile width chosen by the caller. AVX2 8×8 FP32 fires when
    /// <paramref name="mr"/>==8 AND <paramref name="nr"/>==8 AND AVX2+FMA
    /// are runtime-available; else falls through to scalar 4×4.
    /// </summary>
    private static void DispatchStridedMicrokernel<T>(
        ReadOnlySpan<T> packedA, ReadOnlySpan<T> b, int ldb,
        Span<T> c, int ldc, int kc, int mr, int nr) where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            // #409 S.3: higher-intensity 6×16 strided-B FP32 tile (Fast-mode default).
            if (mr == 6 && nr == 16 && Avx2Fp32_6x16.IsSupported)
            {
                Avx2Fp32_6x16.RunStridedB(
                    MemoryMarshal.Cast<T, float>(packedA),
                    MemoryMarshal.Cast<T, float>(b), ldb,
                    MemoryMarshal.Cast<T, float>(c), ldc, kc);
                return;
            }
            // Sub-D (#372) D.3: AVX2 8×8 strided-B when shape and hardware allow.
            if (mr == 8 && nr == 8 && Avx2Fp32_8x8.IsSupported)
            {
                Avx2Fp32_8x8.RunStridedB(
                    MemoryMarshal.Cast<T, float>(packedA),
                    MemoryMarshal.Cast<T, float>(b), ldb,
                    MemoryMarshal.Cast<T, float>(c), ldc, kc);
                return;
            }
            ScalarFp32_4x4.RunStridedB(
                MemoryMarshal.Cast<T, float>(packedA),
                MemoryMarshal.Cast<T, float>(b), ldb,
                MemoryMarshal.Cast<T, float>(c), ldc, kc);
            return;
        }
        if (typeof(T) == typeof(double))
        {
            // #409 S.4: higher-intensity 6×8 strided-B FP64 tile (Fast-mode default).
            if (mr == 6 && nr == 8 && Avx2Fp64_6x8.IsSupported)
            {
                Avx2Fp64_6x8.RunStridedB(
                    MemoryMarshal.Cast<T, double>(packedA),
                    MemoryMarshal.Cast<T, double>(b), ldb,
                    MemoryMarshal.Cast<T, double>(c), ldc, kc);
                return;
            }
            // Sub-D2 follow-up: AVX2 4×8 strided-B for FP64. Mr=4, Nr=8.
            if (mr == 4 && nr == 8 && Avx2Fp64_4x8.IsSupported)
            {
                Avx2Fp64_4x8.RunStridedB(
                    MemoryMarshal.Cast<T, double>(packedA),
                    MemoryMarshal.Cast<T, double>(b), ldb,
                    MemoryMarshal.Cast<T, double>(c), ldc, kc);
                return;
            }
            ScalarFp64_4x4.RunStridedB(
                MemoryMarshal.Cast<T, double>(packedA),
                MemoryMarshal.Cast<T, double>(b), ldb,
                MemoryMarshal.Cast<T, double>(c), ldc, kc);
            return;
        }
        throw new NotSupportedException($"PackAOnlyStrategy does not support T={typeof(T).Name}.");
    }
}
