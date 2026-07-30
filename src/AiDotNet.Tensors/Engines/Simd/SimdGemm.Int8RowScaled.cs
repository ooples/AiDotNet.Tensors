// Copyright (c) AiDotNet. All rights reserved.
// Tensors#401 / AiDotNet#1349: per-row-scaled INT8 weight-only GEMM.
//
// The existing SgemmWithInt8CachedB computes a per-tensor symmetric scale
// during pack (one float scale for all of B). AiDotNet's Int8InferenceModel
// produces per-row scales (one per output channel) via QuantizePerRow, which
// preserves much more SNR on weight matrices whose row magnitudes vary widely
// (the common case in trained transformers). Routing AiDotNet's quantized
// layers through SgemmWithInt8CachedB would require re-quantizing already-
// quantized weights — net loss vs the current dequant+Sgemm path.
//
// This file adds a second cached path that:
//   * Takes weights already as sbyte[] in [n, k] layout (matches AiDotNet's
//     Int8WeightOnlyQuantization.QuantizePerRow output exactly — no copy or
//     re-quantize at the entry point).
//   * Takes per-row scales (length n, one per output column).
//   * Packs the int8 weights into the same tile layout SgemmWithInt8CachedB
//     uses, so MacroKernel + MicroKernel6x16 are identical to the float and
//     per-tensor-int8 paths.
//   * Per-tile dequant during compute folds the per-row scales into the
//     output: the Nr scales for the current Nr-column panel are broadcast
//     into a SIMD vector and applied to every kc-row across that panel.
//
// Expected perf (per Tensors#401 downstream section): the 16×64 transformer
// canary's INT8/FP32 ratio drops from ~15-20x to ~3-5x, and BERT-class shapes
// hit the full 4x DRAM-bandwidth saving on weight loads (the prior FP32
// dequant scratch defeated that win).

// The per-row-scaled INT8 path follows the same TFM gating as the
// per-tensor INT8 path in SimdGemm.cs — PackA/PackB/MacroKernel and
// their AVX2 fast paths are net5.0+ only, so the kernel + cache only
// compile when those primitives are available. net471 consumers fall
// back to the legacy AiDotNet.Inference dequant+Sgemm path (which
// itself doesn't compile on net471's lower SIMD baseline either).
#if NET5_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    // ─────────────────────────────────────────────────────────────────────
    // Cached pre-packed B for per-row-scaled INT8 weight-only GEMM.
    //
    // Mirrors Int8PrePackedB but:
    //   * No per-tensor Scale field — scales are per-row, sized [n].
    //   * Pack input is already sbyte[] (skip the FP32→INT8 quantize step).
    //
    // Cache keying is on the source sbyte[] reference (the consumer's
    // weight array), so the cache survives across Predict calls and is
    // reclaimed when the weight tensor goes out of scope.
    // ─────────────────────────────────────────────────────────────────────
    internal sealed class Int8RowScaledPrePackedB
    {
        internal int K;
        internal int N;
        internal int Kc;
        internal int Mc;
        internal int NumColSubBlocks;
        internal int ColSubSize;
        internal sbyte[][] PackedSubs = Array.Empty<sbyte[]>();
        internal int NumPcIters;
        // Per-row scales, length N. Defensive copy of the consumer's array
        // so a subsequent caller mutation doesn't silently change the
        // dequant scale of cached tiles.
        internal float[] RowScales = Array.Empty<float>();
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<sbyte[], Int8RowScaledPrePackedB>
        _int8RowScaledPrePackedBCache = new();
    private static readonly object _int8RowScaledPrePackedBCacheLock = new();

    /// <summary>
    /// Upper bound on the col-sub-block parallelism grid baked into the cached prepacked-B
    /// (see <see cref="BuildInt8RowScaledPrePackedB"/>). The grid is sized to the machine width
    /// so the int8 GEMM scales regardless of the MaxDegreeOfParallelism at first build; this cap
    /// keeps it bounded on very-many-core hosts (matches the worker-pool ceiling in
    /// PersistentParallelExecutor).
    /// </summary>
    private const int Int8RowScaledMaxGridThreads = 64;

    /// <summary>
    /// Compute <c>C = A · dequant(B_int8, rowScales)</c> using a cached
    /// pre-packed form of B. Weights are already int8 with per-row scales,
    /// avoiding the FP32 round-trip the consumer's prior dequant+Sgemm path
    /// required. B is laid out as <c>[n, k]</c> row-major; <paramref name="rowScales"/>
    /// has length <paramref name="n"/> and <c>rowScales[r]</c> scales row
    /// <c>r</c> of B (= output column <c>r</c>).
    /// </summary>
    /// <param name="a">Activations <c>A</c> in <c>[m, k]</c> row-major.</param>
    /// <param name="bInt8">Pre-quantized weights in <c>[n, k]</c> row-major.</param>
    /// <param name="rowScales">Per-row scales, length <paramref name="n"/>.</param>
    /// <param name="c">Output <c>C</c> in <c>[m, n]</c> row-major; cleared on entry.</param>
    /// <param name="m">Output rows (batch).</param>
    /// <param name="k">Inner dim.</param>
    /// <param name="n">Output cols (features).</param>
    public static void SgemmWithInt8RowScaledCachedB(
        ReadOnlySpan<float> a,
        sbyte[] bInt8,
        ReadOnlySpan<float> rowScales,
        Span<float> c,
        int m, int k, int n)
    {
        if (bInt8 is null) throw new ArgumentNullException(nameof(bInt8));
        if (rowScales.Length < n)
            throw new ArgumentException(
                $"rowScales.Length ({rowScales.Length}) must be >= n ({n}).",
                nameof(rowScales));
        if ((long)bInt8.Length < (long)n * k)
            throw new ArgumentException(
                $"bInt8.Length ({bInt8.Length}) must be >= n*k ({(long)n * k}) for the [n, k] layout.",
                nameof(bInt8));
        if (m <= 0 || n <= 0 || k <= 0) { c.Clear(); return; }

        c.Clear();

        // Path A: large-n / AVX-512 — fall back to a one-time on-the-fly
        // dequant + standard SGEMM. The packed tiled path is sized for the
        // Nc=4096 sub-block budget; once n exceeds that, the per-tile dequant
        // overhead in the cached path equals or exceeds a single-pass FP32
        // dequant + the existing SgemmAddInternal large-n fast path.
        if (n > Nc || Avx512Sgemm.CanUse)
        {
            // Allocate a single FP32 buffer for the full dequantized B and
            // run the standard Sgemm. This path is rare (n > 4096 is unusual
            // for transformer FFN; or AVX-512 hosts where Sgemm dispatches
            // straight to the 512-bit kernel and per-tile dequant isn't a
            // win). We don't cache this path — it's the cold/edge case.
            var dequantFull = System.Buffers.ArrayPool<float>.Shared.Rent(n * k);
            try
            {
                DequantizeInt8WithRowScalesToFloat32_Reference(
                    bInt8, dequantFull.AsSpan(0, n * k), rowScales, n, k);
                // bInt8 is [n, k] row-major; Sgemm wants B in [k, n].
                // Use transB:true to read the dequantized [n, k] buffer
                // through Sgemm's transposed-B path.
                SgemmAddInternal(
                    a, k, false,
                    dequantFull.AsSpan(0, n * k), k, true,
                    c, m, k, n,
                    allowParallel: true, clearedOutput: true);
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(dequantFull);
            }
            return;
        }

        int expectedMc = ChooseAdaptiveMc(m, k, n);
        var cached = GetOrBuildInt8RowScaledPrePackedB(bInt8, rowScales, k, n, m, expectedMc);

        SgemmTiledWithInt8RowScaledCached(a, cached, c, m, k, n);
    }

    private static Int8RowScaledPrePackedB GetOrBuildInt8RowScaledPrePackedB(
        sbyte[] bInt8, ReadOnlySpan<float> rowScales, int k, int n, int m, int expectedMc)
    {
        if (_int8RowScaledPrePackedBCache.TryGetValue(bInt8, out var existing)
            && existing.K == k && existing.N == n
            && existing.Mc == expectedMc
            && RowScalesEqual(existing.RowScales, rowScales))
        {
            return existing;
        }

        lock (_int8RowScaledPrePackedBCacheLock)
        {
            if (_int8RowScaledPrePackedBCache.TryGetValue(bInt8, out existing))
            {
                if (existing.K == k && existing.N == n
                    && existing.Mc == expectedMc
                    && RowScalesEqual(existing.RowScales, rowScales))
                {
                    return existing;
                }
                _int8RowScaledPrePackedBCache.Remove(bInt8);
            }
            var built = BuildInt8RowScaledPrePackedB(bInt8, rowScales, k, n, m);
            _int8RowScaledPrePackedBCache.Add(bInt8, built);
            return built;
        }
    }

    private static bool RowScalesEqual(float[] cached, ReadOnlySpan<float> current)
    {
        if (cached.Length != current.Length) return false;
        // Bitwise compare — cached holds the EXACT bytes the caller passed
        // at first build; if any rowScales[r] changed the dequant must be
        // rebuilt. This is one read per output channel; negligible vs the
        // pack cost it gates.
        for (int i = 0; i < cached.Length; i++)
            if (cached[i] != current[i]) return false;
        return true;
    }

    /// <summary>
    /// Pack INT8 weights laid out as <c>[n, k]</c> row-major into the
    /// tile-friendly layout MacroKernel + MicroKernel6x16 expect. Mirrors
    /// <see cref="BuildInt8PrePackedB"/>'s tile-grid math exactly so all
    /// downstream kernel code is shared — the only differences are
    /// (1) input is sbyte not float (skip quantize), (2) input is [n, k]
    /// not [k, n] (transposed read pattern in PackBInt8FromNK).
    /// </summary>
    private static Int8RowScaledPrePackedB BuildInt8RowScaledPrePackedB(
        sbyte[] bInt8, ReadOnlySpan<float> rowScales, int k, int n, int m)
    {
        int Mc = ChooseAdaptiveMc(m, k, n);
        int numRowBlocks = (m + Mc - 1) / Mc;
        // Size the col-sub-block grid (→ how many tiles the parallel path can fan out over)
        // to the MACHINE width, NOT the current MaxDegreeOfParallelism. This prepacked-B is
        // cached per weight array and BUILT ONCE on the first GEMM; folding in the current
        // MaxDoP would bake the tile count at first-call time — a DOP=1 first call leaves
        // NumColSubBlocks=1, so useParallelPath is false and the int8 GEMM stays SEQUENTIAL
        // for that weight forever (measured: flat 75 GF/s at every DOP vs 7.5× scaling when
        // the cache is built at full width). The per-call path only fans out to the current
        // MaxDoP anyway (LightweightParallel honors it), so a machine-width grid scales when
        // threads are available and costs only a few extra sequential col-block iterations
        // when they aren't. The ceiling keeps the grid bounded on very-many-core hosts.
        int maxThreads = Math.Max(1, Math.Min(Environment.ProcessorCount, Int8RowScaledMaxGridThreads));
        int numPcIters = (k + Kc - 1) / Kc;

        int nc0 = Math.Min(Nc, n);
        int desiredColSubs = Math.Max(1, maxThreads / numRowBlocks);
        int maxColSubs = Math.Max(1, nc0 / (Nr * 4));
        int numColSubBlocks = Math.Min(desiredColSubs, maxColSubs);
        int colSubSize = (nc0 / numColSubBlocks / Nr) * Nr;
        if (colSubSize < Nr)
        {
            colSubSize = nc0;
            numColSubBlocks = 1;
        }

        int colSubRounded = ((colSubSize + Nr - 1) / Nr) * Nr;
        int lastColSubWidth = nc0 - (numColSubBlocks - 1) * colSubSize;
        int lastColSubRounded = ((lastColSubWidth + Nr - 1) / Nr) * Nr;
        int packedBSizePerSub = Kc * Math.Max(colSubRounded, lastColSubRounded);

        var packedSubs = new sbyte[numPcIters * numColSubBlocks][];
        for (int pcIter = 0; pcIter < numPcIters; pcIter++)
        {
            int pc = pcIter * Kc;
            int kc = Math.Min(Kc, k - pc);
            for (int cs = 0; cs < numColSubBlocks; cs++)
            {
                int jStart = cs * colSubSize;
                int subNc = (cs == numColSubBlocks - 1) ? (nc0 - jStart) : colSubSize;
                var int8Buf = new sbyte[packedBSizePerSub];
                PackBInt8FromNK(bInt8, int8Buf, k, n, pc, kc, jStart, subNc);
                packedSubs[pcIter * numColSubBlocks + cs] = int8Buf;
            }
        }

        // Defensive copy so the cache key (the sbyte[] reference) is
        // honoured at lookup time — if the caller mutates rowScales
        // in-place we must NOT silently keep using the new values with
        // the previously-packed tiles. The lookup compares the cached
        // copy against the caller's argument and rebuilds on mismatch.
        var scalesCopy = new float[n];
        rowScales.Slice(0, n).CopyTo(scalesCopy);

        return new Int8RowScaledPrePackedB
        {
            K = k, N = n, Kc = Kc, Mc = Mc,
            NumColSubBlocks = numColSubBlocks,
            ColSubSize = colSubSize,
            PackedSubs = packedSubs,
            NumPcIters = numPcIters,
            RowScales = scalesCopy,
        };
    }

    /// <summary>
    /// Pack an <c>[n, k]</c>-row-major sbyte panel into the same tile
    /// layout PackB produces for FP32 <c>[k, n]</c>. The translation:
    /// each PackB output position <c>(p, jj)</c> reads conceptual
    /// <c>B[pc + p, jc + jj]</c> with B as <c>[k, n]</c>. We have
    /// <c>bInt8</c> as <c>[n, k]</c>, so the same logical element is
    /// at <c>bInt8[(jc + jj) * k + (pc + p)]</c>.
    /// </summary>
    private static void PackBInt8FromNK(
        sbyte[] bInt8, sbyte[] packed,
        int k, int n,
        int pc, int kc, int jc, int subNc)
    {
        int pos = 0;
        int j = 0;
        // Full Nr-column panels first.
        for (; j + Nr <= subNc; j += Nr)
        {
            for (int p = 0; p < kc; p++)
            {
                int rowK = pc + p;
                for (int jj = 0; jj < Nr; jj++)
                {
                    int colN = jc + j + jj;
                    packed[pos++] = bInt8[colN * k + rowK];
                }
            }
        }
        // Tail panel (subNc % Nr != 0): pad to Nr with zeros so the
        // micro-kernel reads valid sbyte values for the zero-padded
        // columns (zero × any-scale == 0, contributes nothing).
        int remaining = subNc - j;
        if (remaining > 0)
        {
            for (int p = 0; p < kc; p++)
            {
                int rowK = pc + p;
                for (int jj = 0; jj < remaining; jj++)
                {
                    int colN = jc + j + jj;
                    packed[pos++] = bInt8[colN * k + rowK];
                }
                for (int jj = remaining; jj < Nr; jj++)
                    packed[pos++] = 0;
            }
        }
    }

    /// <summary>
    /// Tiled SGEMM driver mirroring <see cref="SgemmTiledWithInt8Cached"/>.
    /// The only differences: dequant uses per-row scales (one Nr-wide
    /// vector of scales per Nr-column panel, broadcast then applied across
    /// the panel's kc rows), and the cache key is the consumer's
    /// sbyte[] weight reference.
    /// </summary>
    private static unsafe void SgemmTiledWithInt8RowScaledCached(
        ReadOnlySpan<float> a,
        Int8RowScaledPrePackedB cached,
        Span<float> c,
        int m, int k, int n)
    {
        int Mc = cached.Mc;
        int numRowBlocks = (m + Mc - 1) / Mc;
        int maxThreads = Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        bool canParallelize = UseParallelGemm
            && maxThreads > 1
            && numRowBlocks >= 1
            && (long)m * k * n >= ParallelWorkThreshold;

        int mcRounded = ((Mc + Mr - 1) / Mr) * Mr;
        int packedASizePerRow = mcRounded * Kc;

        // The parallel path requires at least 2 column sub-blocks (nothing to
        // parallelize across otherwise). When canParallelize is true but
        // NumColSubBlocks < 2, fall through to the sequential branch.
        bool useParallelPath = canParallelize && cached.NumColSubBlocks >= 2;

        // Only pack-A buffers are needed now — the fused macro-kernel reads the
        // cached packed int8 weights directly (widened in-register), so there's
        // no FP32 dequant buffer to rent per call.
        var packedABufs = useParallelPath ? new float[numRowBlocks][] : null;
        if (useParallelPath)
            for (int r = 0; r < numRowBlocks; r++)
                packedABufs![r] = System.Buffers.ArrayPool<float>.Shared.Rent(packedASizePerRow);
        var packedABuf = useParallelPath ? null : System.Buffers.ArrayPool<float>.Shared.Rent(packedASizePerRow);

        try
        {
            int jc = 0;
            int nc = Math.Min(Nc, n);
            float[] rowScalesArr = cached.RowScales;

            for (int pcIter = 0; pcIter < cached.NumPcIters; pcIter++)
            {
                int pc = pcIter * Kc;
                int kc = Math.Min(Kc, k - pc);
                int subsBase = pcIter * cached.NumColSubBlocks;

                if (useParallelPath)
                {
                    int localNumRowBlocks = numRowBlocks;
                    int localMc = Mc;
                    int localM = m;
                    int localK = k;
                    int localN = n;
                    int localPc = pc;
                    int localKc = kc;
                    int localColSubSize = cached.ColSubSize;
                    int localNumColSubs = cached.NumColSubBlocks;
                    int localNc = nc;
                    var localPackedABufs = packedABufs!;
                    var localCachedSubs = cached.PackedSubs;
                    int localSubsBase = subsBase;
                    float[] localRowScales = rowScalesArr;

                    fixed (float* aPtr0 = a)
                    fixed (float* cPtr0 = c)
                    {
                        float* localAPtr = aPtr0;
                        int localALen = a.Length;
                        float* localCPtr = cPtr0;
                        int localCLen = c.Length;

                        // Pack-A only (one task per row block).
                        Helpers.CpuParallelSettings.LightweightParallel(localNumRowBlocks, taskId =>
                        {
                            int r = taskId;
                            int ic = r * localMc;
                            int mcLocal = Math.Min(localMc, localM - ic);
                            if (mcLocal > 0)
                            {
                                var aSpan = new ReadOnlySpan<float>(localAPtr, localALen);
                                PackA(aSpan, localPackedABufs[r], localK, ic, mcLocal, localPc, localKc);
                            }
                        });

                        int totalTiles = localNumRowBlocks * localNumColSubs;
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
                                MacroKernelInt8RowScaledFused(
                                    localPackedABufs[r], localCachedSubs[localSubsBase + cs],
                                    localRowScales, jStart,
                                    cSpan, mcLocal, subNc, localKc, localN,
                                    ic, jc + jStart);
                            }
                        });
                    }
                }
                else
                {
                    // Sequential fallback. Pack-A inline; the fused macro-kernel
                    // reads the cached packed int8 weights directly.
                    PackA(a, packedABuf!, k, false, ic: 0, mc: Math.Min(Mc, m), pc, kc);
                    for (int ic = 0; ic < m; ic += Mc)
                    {
                        int mc = Math.Min(Mc, m - ic);
                        if (ic > 0)
                            PackA(a, packedABuf!, k, false, ic, mc, pc, kc);

                        for (int cs = 0; cs < cached.NumColSubBlocks; cs++)
                        {
                            int jStart = cs * cached.ColSubSize;
                            int subNc = (cs == cached.NumColSubBlocks - 1) ? (nc - jStart) : cached.ColSubSize;
                            if (subNc > 0)
                            {
                                MacroKernelInt8RowScaledFused(
                                    packedABuf!, cached.PackedSubs[subsBase + cs],
                                    rowScalesArr, jStart,
                                    c, mc, subNc, kc, n,
                                    ic, jc + jStart);
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            if (packedABuf is not null)
                System.Buffers.ArrayPool<float>.Shared.Return(packedABuf);
            if (packedABufs is not null)
                for (int r = 0; r < numRowBlocks; r++)
                    System.Buffers.ArrayPool<float>.Shared.Return(packedABufs[r]);
        }
    }

    /// <summary>
    /// Reference dequantizer for the large-n / AVX-512 fallback path. Walks
    /// the source <c>[n, k]</c> sbyte layout directly (no pack) and writes
    /// the same <c>[n, k]</c> layout in FP32 — the caller passes this to
    /// SgemmAddInternal with <c>transB:true</c>.
    /// </summary>
    private static void DequantizeInt8WithRowScalesToFloat32_Reference(
        sbyte[] bInt8, Span<float> output, ReadOnlySpan<float> rowScales, int n, int k)
    {
        for (int row = 0; row < n; row++)
        {
            float scale = rowScales[row];
            int baseIdx = row * k;
            for (int col = 0; col < k; col++)
                output[baseIdx + col] = bInt8[baseIdx + col] * scale;
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Fused int8×fp32 macro/micro-kernels.
    //
    // The weights stay packed int8 all the way into the micro-kernel: each
    // K-row's 16 int8 weights are widened to fp32 IN-REGISTER (VPMOVSXBD +
    // CVTDQ2PS) during the FMA loop — no separate dequant pass, no FP32 B
    // buffer, no per-call ArrayPool rent for it. The per-column row-scale is
    // constant across the K loop, so it's folded into the store via one FMA
    // per output row (C += accum·scale) instead of a multiply per K step. This
    // is the realization of the "INT8 all the way through the kernel" goal:
    // C[r,c] += scale[c] · Σ_k A[r,k]·B_int8[c,k].
    // ───────────────────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void WidenInt8RowToFloat(
        sbyte* p, out Vector256<float> b0, out Vector256<float> b1)
    {
        // 16 packed int8 weights (one K-row of an Nr=16 panel) → two fp32 vectors.
        Vector128<sbyte> raw = Sse2.LoadVector128(p);
        b0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(raw));
        b1 = Avx.ConvertToVector256Single(
            Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(raw, 8)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreScaledAccumRow(
        ref float cRef, int row, int col, int ldc,
        Vector256<float> v0, Vector256<float> v1,
        Vector256<float> s0, Vector256<float> s1)
    {
        ref float target = ref Unsafe.Add(ref cRef, row * ldc + col);
        var e0 = Unsafe.ReadUnaligned<Vector256<float>>(ref Unsafe.As<float, byte>(ref target));
        var e1 = Unsafe.ReadUnaligned<Vector256<float>>(
            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref target, 8)));
        Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref target), Fma.MultiplyAdd(v0, s0, e0));
        Unsafe.WriteUnaligned(
            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref target, 8)), Fma.MultiplyAdd(v1, s1, e1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreMaskedScaledAccumRow(
        float* row, Vector256<int> mask0, Vector256<int> mask1,
        Vector256<float> v0, Vector256<float> v1,
        Vector256<float> s0, Vector256<float> s1)
    {
        var m0f = mask0.AsSingle();
        var m1f = mask1.AsSingle();
        var e0 = Avx.MaskLoad(row, m0f);
        var e1 = Avx.MaskLoad(row + 8, m1f);
        Avx.MaskStore(row, m0f, Fma.MultiplyAdd(v0, s0, e0));
        Avx.MaskStore(row + 8, m1f, Fma.MultiplyAdd(v1, s1, e1));
    }

    private static unsafe void MicroKernel6x16Int8RowScaledFused(
        float[] packedA, int aOffset,
        sbyte[] packedBInt8, int bOffset,
        Vector256<float> s0, Vector256<float> s1,
        Span<float> c, int ldc, int cRow, int cCol, int kc)
    {
        var c00 = Vector256<float>.Zero; var c01 = Vector256<float>.Zero;
        var c10 = Vector256<float>.Zero; var c11 = Vector256<float>.Zero;
        var c20 = Vector256<float>.Zero; var c21 = Vector256<float>.Zero;
        var c30 = Vector256<float>.Zero; var c31 = Vector256<float>.Zero;
        var c40 = Vector256<float>.Zero; var c41 = Vector256<float>.Zero;
        var c50 = Vector256<float>.Zero; var c51 = Vector256<float>.Zero;

        ref float aRef = ref MemoryMarshal.GetArrayDataReference(packedA);
        fixed (sbyte* pB0 = packedBInt8)
        {
            sbyte* pB = pB0 + bOffset;
            for (int p = 0; p < kc; p++)
            {
                WidenInt8RowToFloat(pB + p * Nr, out var b0, out var b1);
                int aBase = aOffset + p * Mr;
                var a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 0));
                c00 = Fma.MultiplyAdd(a, b0, c00); c01 = Fma.MultiplyAdd(a, b1, c01);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 1));
                c10 = Fma.MultiplyAdd(a, b0, c10); c11 = Fma.MultiplyAdd(a, b1, c11);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 2));
                c20 = Fma.MultiplyAdd(a, b0, c20); c21 = Fma.MultiplyAdd(a, b1, c21);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 3));
                c30 = Fma.MultiplyAdd(a, b0, c30); c31 = Fma.MultiplyAdd(a, b1, c31);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 4));
                c40 = Fma.MultiplyAdd(a, b0, c40); c41 = Fma.MultiplyAdd(a, b1, c41);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 5));
                c50 = Fma.MultiplyAdd(a, b0, c50); c51 = Fma.MultiplyAdd(a, b1, c51);
            }
        }

        ref float cRef = ref MemoryMarshal.GetReference(c);
        StoreScaledAccumRow(ref cRef, cRow + 0, cCol, ldc, c00, c01, s0, s1);
        StoreScaledAccumRow(ref cRef, cRow + 1, cCol, ldc, c10, c11, s0, s1);
        StoreScaledAccumRow(ref cRef, cRow + 2, cCol, ldc, c20, c21, s0, s1);
        StoreScaledAccumRow(ref cRef, cRow + 3, cCol, ldc, c30, c31, s0, s1);
        StoreScaledAccumRow(ref cRef, cRow + 4, cCol, ldc, c40, c41, s0, s1);
        StoreScaledAccumRow(ref cRef, cRow + 5, cCol, ldc, c50, c51, s0, s1);
    }

    private static unsafe void MicroKernelMxNMaskedInt8RowScaledFused(
        float[] packedA, int aOffset,
        sbyte[] packedBInt8, int bOffset,
        Vector256<float> s0, Vector256<float> s1,
        Span<float> c, int ldc, int cRow, int cCol,
        int kc, int mc_actual, int nc_actual)
    {
        var c00 = Vector256<float>.Zero; var c01 = Vector256<float>.Zero;
        var c10 = Vector256<float>.Zero; var c11 = Vector256<float>.Zero;
        var c20 = Vector256<float>.Zero; var c21 = Vector256<float>.Zero;
        var c30 = Vector256<float>.Zero; var c31 = Vector256<float>.Zero;
        var c40 = Vector256<float>.Zero; var c41 = Vector256<float>.Zero;
        var c50 = Vector256<float>.Zero; var c51 = Vector256<float>.Zero;

        ref float aRef = ref MemoryMarshal.GetArrayDataReference(packedA);
        fixed (sbyte* pB0 = packedBInt8)
        {
            sbyte* pB = pB0 + bOffset;
            for (int p = 0; p < kc; p++)
            {
                WidenInt8RowToFloat(pB + p * Nr, out var b0, out var b1);
                int aBase = aOffset + p * Mr;
                var a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 0));
                c00 = Fma.MultiplyAdd(a, b0, c00); c01 = Fma.MultiplyAdd(a, b1, c01);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 1));
                c10 = Fma.MultiplyAdd(a, b0, c10); c11 = Fma.MultiplyAdd(a, b1, c11);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 2));
                c20 = Fma.MultiplyAdd(a, b0, c20); c21 = Fma.MultiplyAdd(a, b1, c21);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 3));
                c30 = Fma.MultiplyAdd(a, b0, c30); c31 = Fma.MultiplyAdd(a, b1, c31);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 4));
                c40 = Fma.MultiplyAdd(a, b0, c40); c41 = Fma.MultiplyAdd(a, b1, c41);
                a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 5));
                c50 = Fma.MultiplyAdd(a, b0, c50); c51 = Fma.MultiplyAdd(a, b1, c51);
            }
        }

        int lane0N = nc_actual >= 8 ? 8 : nc_actual;
        int lane1N = nc_actual >= 8 ? nc_actual - 8 : 0;
        Vector256<int> mask0 = _partialNrMasks[lane0N];
        Vector256<int> mask1 = _partialNrMasks[lane1N];

        fixed (float* pC = c)
        {
            float* row = pC + cRow * ldc + cCol;
            if (mc_actual > 0) StoreMaskedScaledAccumRow(row,           mask0, mask1, c00, c01, s0, s1);
            if (mc_actual > 1) StoreMaskedScaledAccumRow(row + ldc,     mask0, mask1, c10, c11, s0, s1);
            if (mc_actual > 2) StoreMaskedScaledAccumRow(row + ldc * 2, mask0, mask1, c20, c21, s0, s1);
            if (mc_actual > 3) StoreMaskedScaledAccumRow(row + ldc * 3, mask0, mask1, c30, c31, s0, s1);
            if (mc_actual > 4) StoreMaskedScaledAccumRow(row + ldc * 4, mask0, mask1, c40, c41, s0, s1);
            if (mc_actual > 5) StoreMaskedScaledAccumRow(row + ldc * 5, mask0, mask1, c50, c51, s0, s1);
        }
    }

    /// <summary>
    /// Fused macro-kernel: drives the int8×fp32 micro-kernels over an
    /// (mc × nc) C tile reading packed int8 weights directly (no dequant
    /// buffer). <paramref name="rowScaleBase"/> is the column offset of this
    /// sub-block within the per-row-scale array.
    /// </summary>
    private static unsafe void MacroKernelInt8RowScaledFused(
        float[] packedA, sbyte[] packedBInt8,
        float[] rowScales, int rowScaleBase,
        Span<float> c, int mc, int nc, int kc, int ldc, int icOffset, int jcOffset)
    {
        int nrBlocks = (nc + Nr - 1) / Nr;
        int mrBlocks = (mc + Mr - 1) / Mr;
        float* sc = stackalloc float[Nr];

        for (int jr = 0; jr < nrBlocks; jr++)
        {
            int jLocal = jr * Nr;
            int nc_actual = Math.Min(Nr, nc - jLocal);
            int bPanelOffset = jr * kc * Nr;

            // Gather the 16 per-column scales for this Nr panel; pad columns
            // past nc_actual with 0 so the zero-padded packed weights × 0
            // contribute nothing (mirrors PackBInt8FromNK's tail padding).
            for (int jj = 0; jj < Nr; jj++)
                sc[jj] = jj < nc_actual ? rowScales[rowScaleBase + jLocal + jj] : 0f;
            var s0 = Avx.LoadVector256(sc);
            var s1 = Avx.LoadVector256(sc + 8);

            for (int ir = 0; ir < mrBlocks; ir++)
            {
                int iLocal = ir * Mr;
                int mc_actual = Math.Min(Mr, mc - iLocal);
                int aPanelOffset = ir * kc * Mr;

                if (mc_actual == Mr && nc_actual == Nr)
                {
                    MicroKernel6x16Int8RowScaledFused(
                        packedA, aPanelOffset, packedBInt8, bPanelOffset,
                        s0, s1, c, ldc, icOffset + iLocal, jcOffset + jLocal, kc);
                }
                else if (mc_actual > 0 && nc_actual > 0)
                {
                    MicroKernelMxNMaskedInt8RowScaledFused(
                        packedA, aPanelOffset, packedBInt8, bPanelOffset,
                        s0, s1, c, ldc, icOffset + iLocal, jcOffset + jLocal,
                        kc, mc_actual, nc_actual);
                }
            }
        }
    }
}
#endif // NET5_0_OR_GREATER
