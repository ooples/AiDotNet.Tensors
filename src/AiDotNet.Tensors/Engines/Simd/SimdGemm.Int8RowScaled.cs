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
        int maxThreads = Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
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
        int packedBSizePerSub = cached.PackedSubs.Length > 0 ? cached.PackedSubs[0].Length : 0;

        // The parallel path requires at least 2 column sub-blocks — pack-A and
        // dequant are dispatched as taskId>=numRowBlocks tasks alongside the
        // pack-A tasks, and a single column sub-block leaves nothing to
        // parallelize on the dequant side. When canParallelize is true but
        // NumColSubBlocks < 2, the code falls through to the sequential branch
        // below; that branch reads packedABuf!/dequantBuf!. Pre-fix those were
        // null in this case → NRE under low n (n < Nr*4 ≈ 64). Compute the
        // effective branch once here and key all buffer allocations off it.
        bool useParallelPath = canParallelize && cached.NumColSubBlocks >= 2;

        var packedABufs = useParallelPath ? new float[numRowBlocks][] : null;
        if (useParallelPath)
            for (int r = 0; r < numRowBlocks; r++)
                packedABufs![r] = System.Buffers.ArrayPool<float>.Shared.Rent(packedASizePerRow);
        var packedABuf = useParallelPath ? null : System.Buffers.ArrayPool<float>.Shared.Rent(packedASizePerRow);

        var dequantBufs = useParallelPath ? new float[cached.NumColSubBlocks][] : null;
        if (useParallelPath)
            for (int cs = 0; cs < cached.NumColSubBlocks; cs++)
                dequantBufs![cs] = System.Buffers.ArrayPool<float>.Shared.Rent(packedBSizePerSub);
        var dequantBuf = useParallelPath ? null : System.Buffers.ArrayPool<float>.Shared.Rent(packedBSizePerSub);

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
                    var localDequantBufs = dequantBufs!;
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

                        int numPackTasks = localNumRowBlocks + localNumColSubs;
                        Helpers.CpuParallelSettings.LightweightParallel(numPackTasks, taskId =>
                        {
                            if (taskId < localNumRowBlocks)
                            {
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
                                int cs = taskId - localNumRowBlocks;
                                int jStart = cs * localColSubSize;
                                int subNc = (cs == localNumColSubs - 1) ? (localNc - jStart) : localColSubSize;
                                var int8Sub = localCachedSubs[localSubsBase + cs];
                                DequantizeInt8WithRowScalesToFloat32(
                                    int8Sub, localDequantBufs[cs],
                                    localRowScales, jStart, localKc, subNc);
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
                                MacroKernel(
                                    localPackedABufs[r], localDequantBufs[cs],
                                    cSpan, mcLocal, subNc, localKc, localN,
                                    ic, jc + jStart);
                            }
                        });
                    }
                }
                else
                {
                    // Sequential fallback. Pack-A and dequant happen inline.
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
                                DequantizeInt8WithRowScalesToFloat32(
                                    cached.PackedSubs[subsBase + cs], dequantBuf!,
                                    rowScalesArr, jStart, kc, subNc);
                                MacroKernel(
                                    packedABuf!, dequantBuf!,
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
            if (dequantBuf is not null)
                System.Buffers.ArrayPool<float>.Shared.Return(dequantBuf);
            if (packedABufs is not null)
                for (int r = 0; r < numRowBlocks; r++)
                    System.Buffers.ArrayPool<float>.Shared.Return(packedABufs[r]);
            if (dequantBufs is not null)
                for (int cs = 0; cs < cached.NumColSubBlocks; cs++)
                    System.Buffers.ArrayPool<float>.Shared.Return(dequantBufs[cs]);
        }
    }

    /// <summary>
    /// Dequantize one packed sub-block of int8 weights with per-row scales,
    /// producing the FP32 buffer the macro-kernel reads. Within ONE Nr-wide
    /// panel of the packed buffer, the same Nr scales apply to every kc-row,
    /// so we broadcast them once and multiply per row — the per-row scale
    /// lookup happens Nr times per panel (not Nr × kc times).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DequantizeInt8WithRowScalesToFloat32(
        ReadOnlySpan<sbyte> int8Sub,
        Span<float> output,
        ReadOnlySpan<float> rowScales,
        int jStart,
        int kc,
        int subNc)
    {
        int numPanels = (subNc + Nr - 1) / Nr;

        Span<float> panelScales = stackalloc float[Nr];
        for (int panel = 0; panel < numPanels; panel++)
        {
            int panelStart = panel * Nr;
            int panelCols = Math.Min(Nr, subNc - panelStart);

            // Gather the Nr scales for this panel; pad with zero so a
            // partial tail panel contributes zero through the
            // (zero-padded packed weight) × (zero scale) multiply.
            for (int jj = 0; jj < Nr; jj++)
                panelScales[jj] = jj < panelCols ? rowScales[jStart + panelStart + jj] : 0f;

            int packedRowStride = Nr;
            int panelBase = panel * kc * Nr;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported)
            {
                fixed (sbyte* pIn = int8Sub)
                fixed (float* pOut = output)
                fixed (float* pScales = panelScales)
                {
                    // Two AVX2 256-bit vectors cover the 16-wide Nr.
                    var vScale0 = Avx.LoadVector256(pScales + 0);
                    var vScale1 = Avx.LoadVector256(pScales + 8);
                    for (int p = 0; p < kc; p++)
                    {
                        int row = panelBase + p * packedRowStride;
                        sbyte* pInRow = pIn + row;
                        float* pOutRow = pOut + row;
                        var v0 = Vector256.Create(
                            (float)pInRow[0], (float)pInRow[1], (float)pInRow[2], (float)pInRow[3],
                            (float)pInRow[4], (float)pInRow[5], (float)pInRow[6], (float)pInRow[7]);
                        var v1 = Vector256.Create(
                            (float)pInRow[8], (float)pInRow[9], (float)pInRow[10], (float)pInRow[11],
                            (float)pInRow[12], (float)pInRow[13], (float)pInRow[14], (float)pInRow[15]);
                        Avx.Store(pOutRow + 0, Avx.Multiply(v0, vScale0));
                        Avx.Store(pOutRow + 8, Avx.Multiply(v1, vScale1));
                    }
                }
                continue;
            }
#endif
            // Scalar fallback — used on net471 / non-AVX2 hosts.
            for (int p = 0; p < kc; p++)
            {
                int row = panelBase + p * packedRowStride;
                for (int jj = 0; jj < Nr; jj++)
                    output[row + jj] = (float)int8Sub[row + jj] * panelScales[jj];
            }
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
}
#endif // NET5_0_OR_GREATER
