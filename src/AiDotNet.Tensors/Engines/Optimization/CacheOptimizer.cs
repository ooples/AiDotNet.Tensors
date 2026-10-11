// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Optimization
{
    /// <summary>
    /// CPU cache optimization helpers — the read/write side of the
    /// optimization API split (control-flow lives in <see cref="LoopOptimizer"/>).
    /// All public APIs are generic over <c>T : unmanaged</c> and operate
    /// on <see cref="ReadOnlySpan{T}"/> / <see cref="Span{T}"/> so callers
    /// pay no element-size penalty for working in float / double / Half /
    /// int / etc. — the type drives the byte math through
    /// <see cref="Unsafe.SizeOf{T}"/>.
    ///
    /// <para><b>Single source of truth for L1-aware tile sizing:</b>
    /// <see cref="ComputeOptimalTiling{T}(int, int, int)"/> is the
    /// canonical matmul-shape tile picker. <see cref="LoopOptimizer.DetermineOptimalTileSize{T}"/>
    /// and <see cref="AiDotNet.Tensors.Helpers.MatrixMultiplyHelper.GetBlockSize{T}"/>
    /// (when used) delegate to it instead of duplicating the L1 math.</para>
    /// </summary>
    public static class CacheOptimizer
    {
        /// <summary>
        /// Optimal block size for the L1 cache, expressed as float-sized
        /// elements (64 elements × 4 bytes = 256 bytes, one typical
        /// cache line). Use <see cref="ComputeOptimalTiling{T}"/> for
        /// element-type-aware tile sizing — this constant is kept for
        /// callers that want a coarse default.
        /// </summary>
        public static int L1BlockSize => 64;

        /// <summary>Optimal block size for the L2 cache (float-element units).</summary>
        public static int L2BlockSize => 512;

        /// <summary>Optimal block size for the L3 cache (float-element units).</summary>
        public static int L3BlockSize => 2048;

        /// <summary>
        /// Computes optimal tiling parameters for a 2D matmul-shaped
        /// operation using the runtime's detected L1 cache size.
        ///
        /// <para>For matmul C[m,n] += A[m,k] · B[k,n] the working set
        /// per tile is <c>tileM·tileK + tileK·tileN + tileM·tileN</c>
        /// elements; we pick a per-dim tile size that fits all three in
        /// L1 with the formula <c>tile = sqrt(L1 / (3·sizeof(T)))</c>,
        /// then round down to a power of two for alignment.</para>
        ///
        /// <para><b>This method is the single source of truth for L1-aware
        /// tiling.</b> Loop-flow callers should consume this directly
        /// rather than re-deriving the cache math.</para>
        /// </summary>
        public static (int tileM, int tileN, int tileK) ComputeOptimalTiling<T>(int m, int n, int k)
            where T : unmanaged
        {
            if (m < 0) throw new ArgumentOutOfRangeException(nameof(m));
            if (n < 0) throw new ArgumentOutOfRangeException(nameof(n));
            if (k < 0) throw new ArgumentOutOfRangeException(nameof(k));

            int elementSize = Unsafe.SizeOf<T>();
            var caps = PlatformDetector.Capabilities;
            int l1Size = caps.L1CacheSize;

            // Aim for the working set (3 tile blocks for matmul) to fit
            // in L1. sqrt(L1 / (3·elementSize)) gives the per-dim tile
            // length that satisfies that constraint.
            int maxTileSize = (int)Math.Sqrt(l1Size / (3.0 * Math.Max(1, elementSize)));

            // Round down to power-of-two for memory alignment friendliness.
            int tileSize = 1;
            while (tileSize * 2 <= maxTileSize) tileSize *= 2;

            // Floor at 16 — below this the tile overhead dominates the
            // useful compute and dispatch loops thrash.
            tileSize = Math.Max(tileSize, 16);

            return (Math.Min(tileSize, m), Math.Min(tileSize, n), Math.Min(tileSize, k));
        }

        /// <summary>
        /// Cache-line-blocked transpose. Walks the source in row-major
        /// order in 32-element-square blocks so both the read and write
        /// streams see contiguous memory at the block boundary, which is
        /// dramatically cache-friendlier than a naive O(rows × cols)
        /// strided write.
        ///
        /// <para>Generic over <c>T : unmanaged</c> — works for any
        /// blittable element type. <paramref name="src"/> is treated as
        /// a logical <c>[rows, cols]</c> matrix in row-major layout,
        /// and <paramref name="dst"/> receives a <c>[cols, rows]</c>
        /// matrix in row-major layout (i.e., the transpose).</para>
        ///
        /// <para><b>Overlap behaviour</b>: this routine is NOT in-place
        /// safe. Block-major writes touch <c>dst[jj*rows + ii]</c> while
        /// reads still need <c>src[(jj')*cols + ii']</c> from across the
        /// diagonal — overlapping spans would clobber unread source data
        /// for any non-trivial shape (and even for square shapes the
        /// access pattern is not a swap). When overlap is detected we
        /// throw <see cref="ArgumentException"/> rather than silently
        /// producing garbage.</para>
        /// </summary>
        public static void TransposeBlocked<T>(ReadOnlySpan<T> src, Span<T> dst, int rows, int cols)
            where T : unmanaged
        {
            if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols));
            long total = (long)rows * cols;
            if (src.Length < total)
                throw new ArgumentException($"src length {src.Length} too short for [{rows}, {cols}].", nameof(src));
            if (dst.Length < total)
                throw new ArgumentException($"dst length {dst.Length} too short for [{cols}, {rows}].", nameof(dst));
#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
            if (total > 0 && SpansOverlap(src, dst))
                throw new ArgumentException(
                    "TransposeBlocked does not support overlapping src and dst spans — block-major writes "
                    + "would clobber source data across the diagonal. Allocate a separate destination buffer.",
                    nameof(dst));
#endif

            // 32-element block fits in two cache lines for T=float (256B
            // line, 32×4=128B), one line for T=double, etc. Larger
            // blocks would spill the inner loop's registers.
            const int blockSize = 32;

            for (int i = 0; i < rows; i += blockSize)
            {
                int iMax = Math.Min(i + blockSize, rows);
                for (int j = 0; j < cols; j += blockSize)
                {
                    int jMax = Math.Min(j + blockSize, cols);
                    for (int ii = i; ii < iMax; ii++)
                    {
                        for (int jj = j; jj < jMax; jj++)
                            dst[jj * rows + ii] = src[ii * cols + jj];
                    }
                }
            }
        }

        /// <summary>
        /// Streaming copy with hardware prefetch hints for the read
        /// stream. On x86 hosts with SSE we issue a <c>PREFETCHT0</c>
        /// for cache-line-ahead source data — hides L2-miss latency on
        /// streaming patterns ≥1KB. On non-x86 hosts (or below the
        /// threshold), devolves to <see cref="Span{T}.CopyTo"/>, which
        /// is already SIMD-vectorized via the BCL.
        ///
        /// <para><b>Overlap behaviour</b>: the prefetch path uses
        /// forward-direction line-by-line copies which are NOT
        /// overlap-safe — overlapping <paramref name="src"/> and
        /// <paramref name="dst"/> spans where <paramref name="dst"/>
        /// trails <paramref name="src"/> would clobber data still
        /// pending read. The method detects overlap up front and
        /// routes to <see cref="Span{T}.CopyTo"/>, which dispatches
        /// to <c>Buffer.Memmove</c> (handles overlap correctly via
        /// directional choice). Same overlap guarantee as
        /// <see cref="Array.Copy(Array, Array, int)"/>.</para>
        ///
        /// <para>Generic over <c>T : unmanaged</c>. The prefetch stride
        /// is computed in cache-line units (64 bytes assumed) divided
        /// by <see cref="Unsafe.SizeOf{T}"/> so each iteration prefetches
        /// the next line worth of source data while the CPU is reading
        /// the current line.</para>
        /// </summary>
        public static void CopyWithPrefetch<T>(ReadOnlySpan<T> src, Span<T> dst)
            where T : unmanaged
        {
            if (dst.Length < src.Length)
                throw new ArgumentException($"dst length {dst.Length} < src length {src.Length}.", nameof(dst));

            int length = src.Length;
            if (length == 0) return;

#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
            // Only emit prefetch on platforms whose JIT can lower it AND
            // where length is large enough that the prefetch latency
            // (~100 cycles to hide L2 miss) is shorter than the inner
            // loop. Below 1KB the plain CopyTo's microcoded rep movsb
            // / SIMD store stream wins.
            //
            // Overlap check: if the spans overlap, our forward-direction
            // line-by-line copy can clobber data the next iteration
            // still needs to read. Detect overlap by comparing the
            // underlying refs; when overlap exists, fall through to
            // Span.CopyTo which routes to Buffer.Memmove (picks the
            // correct copy direction).
            int byteCount = length * Unsafe.SizeOf<T>();
            // Only x86 SSE has a stable cross-runtime intrinsic for
            // PREFETCHT0 in .NET. ARM gets the plain SIMD-vectorized
            // CopyTo (BCL uses NEON internally where available); skipping
            // the manual prefetch is fine because ARM cores are
            // aggressive about hardware prefetch on streaming patterns.
            if (byteCount >= 1024 && Sse.IsSupported && !SpansOverlap(src, dst))
            {
                CopyWithPrefetchUnsafe(src, dst);
                return;
            }
#endif
            // Fallback: built-in copy is already SIMD-vectorized for
            // unmanaged T on .NET 6+ and overlap-safe via Buffer.Memmove
            // (it picks forward or backward direction as needed).
            src.CopyTo(dst);
        }

#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
        /// <summary>
        /// True when the two spans share any byte of memory, i.e. a
        /// forward-only line-by-line copy would clobber data still
        /// needed for read. False for the common case of disjoint
        /// arrays. Computes the byte-range intersection via
        /// <see cref="MemoryMarshal.GetReference"/> + length without
        /// allocating.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool SpansOverlap<T>(ReadOnlySpan<T> a, Span<T> b)
            where T : unmanaged
        {
            ref T ra = ref MemoryMarshal.GetReference(a);
            ref T rb = ref MemoryMarshal.GetReference(b);
            // Convert refs to byte offsets in the GC heap. ByteOffset is
            // safe across managed objects (works as long as the GC
            // doesn't move the references mid-call — the spans pin the
            // memory region for the call duration via the ref).
            long delta = Unsafe.ByteOffset(ref ra, ref rb).ToInt64();
            long aBytes = (long)a.Length * Unsafe.SizeOf<T>();
            long bBytes = (long)b.Length * Unsafe.SizeOf<T>();
            // a starts at 0; b starts at delta. Overlap iff
            // [0, aBytes) ∩ [delta, delta + bBytes) ≠ ∅.
            // i.e. delta < aBytes && delta + bBytes > 0.
            return delta < aBytes && delta + bBytes > 0;
        }
#endif

#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
        /// <summary>
        /// Pointer-based copy that issues a one-cache-line-ahead prefetch
        /// for each iteration of the bulk loop. The .NET 6+ <c>Span.CopyTo</c>
        /// path is already fast for short ranges; this kicks in only
        /// once we have enough work that hiding L2 latency matters.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void CopyWithPrefetchUnsafe<T>(ReadOnlySpan<T> src, Span<T> dst)
            where T : unmanaged
        {
            int length = src.Length;
            int elemSize = Unsafe.SizeOf<T>();
            // 64-byte cache lines are universal on x86 and current ARM
            // server cores. Two-line lookahead covers the typical
            // ~100-cycle L2 miss latency at modern clock speeds.
            const int CacheLineBytes = 64;
            int lineStride = Math.Max(1, CacheLineBytes / Math.Max(1, elemSize));
            int prefetchAhead = lineStride * 2;

            ref T srcRef = ref MemoryMarshal.GetReference(src);
            ref T dstRef = ref MemoryMarshal.GetReference(dst);

            // Bulk loop: prefetch the source line two ahead, copy the
            // current line. ref-arithmetic avoids the bounds check that
            // span indexers introduce, since we've already validated
            // dst.Length >= src.Length above.
            int i = 0;
            int bulkEnd = length - prefetchAhead;
            for (; i < bulkEnd; i += lineStride)
            {
                fixed (T* p = &Unsafe.Add(ref srcRef, i + prefetchAhead))
                {
                    if (Sse.IsSupported)
                        Sse.Prefetch0(p);
                    // ARM has data-prefetch instructions but the .NET
                    // intrinsic surface for them landed differently
                    // across runtimes; rather than thread a feature-by-
                    // feature gate, we let ARM use the plain CopyTo
                    // path (still SIMD via NEON via the BCL) and only
                    // emit the explicit hint on x86 SSE which is
                    // unambiguously available.
                }
                // Slice + CopyTo is the SIMD-vectorized path for the
                // current line(s); we just hint the CPU to start
                // bringing in the next ones in parallel.
                int copyLen = Math.Min(lineStride, length - i);
                src.Slice(i, copyLen).CopyTo(dst.Slice(i, copyLen));
            }
            // Tail: just copy whatever's left without prefetch (the
            // remaining work is shorter than the prefetch window).
            if (i < length)
                src.Slice(i, length - i).CopyTo(dst.Slice(i, length - i));
        }
#endif

        /// <summary>
        /// Z-order (Morton) encoding for 2D-locality-preserving 1D
        /// indexing. Useful when iterating a 2D region in cache-friendly
        /// order without paying the loop-tile overhead of explicit
        /// blocking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int MortonEncode(int x, int y) => (Part1By1(y) << 1) | Part1By1(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Part1By1(int n)
        {
            n &= 0x0000ffff;
            n = (n ^ (n << 8)) & 0x00ff00ff;
            n = (n ^ (n << 4)) & 0x0f0f0f0f;
            n = (n ^ (n << 2)) & 0x33333333;
            n = (n ^ (n << 1)) & 0x55555555;
            return n;
        }

        /// <summary>Decodes a Morton index back to its 2D coordinates.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static (int x, int y) MortonDecode(int code) => (Compact1By1(code), Compact1By1(code >> 1));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Compact1By1(int n)
        {
            n &= 0x55555555;
            n = (n ^ (n >> 1)) & 0x33333333;
            n = (n ^ (n >> 2)) & 0x0f0f0f0f;
            n = (n ^ (n >> 4)) & 0x00ff00ff;
            n = (n ^ (n >> 8)) & 0x0000ffff;
            return n;
        }

        /// <summary>
        /// Pre-flight diagnostic: returns an estimated cache-miss count
        /// for an access pattern over <paramref name="dataSize"/> elements
        /// stepping by <paramref name="accessStride"/>. Element-type
        /// aware via <typeparamref name="T"/>'s size.
        ///
        /// <para>This is a coarse heuristic, not a hardware-counter
        /// reading. Useful for choosing between candidate loop orders
        /// at design time; for production tuning use BenchmarkDotNet
        /// with hardware-counter providers.</para>
        /// </summary>
        public static double EstimateCacheMisses<T>(int dataSize, int accessStride, int cacheSize, int cacheLineSize)
            where T : unmanaged
        {
            if (cacheLineSize <= 0) throw new ArgumentOutOfRangeException(nameof(cacheLineSize));
            int elementSize = Math.Max(1, Unsafe.SizeOf<T>());
            int elementsPerLine = Math.Max(1, cacheLineSize / elementSize);
            int totalLines = (dataSize + elementsPerLine - 1) / elementsPerLine;
            int cacheLinesAvailable = cacheSize / cacheLineSize;

            if (accessStride <= elementsPerLine) return totalLines * 0.1;       // sequential
            if (totalLines <= cacheLinesAvailable) return totalLines * 0.05;    // fits in cache
            return totalLines * 0.8;                                            // strided + thrashing
        }

        // ── [Obsolete] float[]-based compatibility shims ───────────────
        //
        // The previous public surface took plain float[] and an explicit
        // length / elementSize parameter. Issue #294 generalized to
        // ReadOnlySpan<T> / Span<T> / Unsafe.SizeOf<T> so the same APIs
        // serve double / Half / int callers. The old signatures are
        // kept here as [Obsolete]-marked forwarding shims to avoid a
        // hard source-break for package consumers.

        /// <summary>[Obsolete] Use the generic
        /// <see cref="TransposeBlocked{T}(ReadOnlySpan{T}, Span{T}, int, int)"/>
        /// overload. Forwards to the generic path with T = float.</summary>
        [Obsolete("Use TransposeBlocked<T>(ReadOnlySpan<T>, Span<T>, int, int) where T : unmanaged. Replaced in issue #294 to support double/Half/int callers.")]
        public static void TransposeBlocked(float[] src, float[] dst, int rows, int cols)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            TransposeBlocked<float>(src.AsSpan(), dst.AsSpan(), rows, cols);
        }

        /// <summary>[Obsolete] Use the generic
        /// <see cref="CopyWithPrefetch{T}(ReadOnlySpan{T}, Span{T})"/>
        /// overload. Forwards to the generic path with T = float;
        /// the old <paramref name="length"/> parameter is honoured by
        /// slicing.</summary>
        [Obsolete("Use CopyWithPrefetch<T>(ReadOnlySpan<T>, Span<T>) where T : unmanaged. Replaced in issue #294; the explicit length parameter is now carried by span lengths.")]
        public static void CopyWithPrefetch(float[] src, float[] dst, int length)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
            if (src.Length < length) throw new ArgumentException("src too short for requested length.", nameof(src));
            if (dst.Length < length) throw new ArgumentException("dst too short for requested length.", nameof(dst));
            CopyWithPrefetch<float>(src.AsSpan(0, length), dst.AsSpan(0, length));
        }

        /// <summary>[Obsolete] Use the generic
        /// <see cref="ComputeOptimalTiling{T}(int, int, int)"/>
        /// overload. The old <paramref name="elementSize"/> parameter
        /// is now derived via <see cref="Unsafe.SizeOf{T}"/>; this
        /// shim assumes T = float (4 bytes) when the caller passes
        /// the historical default.</summary>
        [Obsolete("Use ComputeOptimalTiling<T>(int, int, int) where T : unmanaged. The element-size argument is now carried by T's size; this shim defaults to float.")]
        public static (int tileM, int tileN, int tileK) ComputeOptimalTiling(int m, int n, int k, int elementSize = 4)
        {
            // Old behaviour: pick a tile such that 3 blocks fit in L1.
            // We replicate the historical formula here rather than
            // routing through ComputeOptimalTiling<float> to preserve
            // exact byte-for-byte compatibility with callers that
            // passed elementSize ≠ 4.
            var caps = PlatformDetector.Capabilities;
            int l1 = caps?.L1CacheSize ?? 32 * 1024;
            int maxTileSize = (int)Math.Sqrt(l1 / (3.0 * Math.Max(1, elementSize)));
            int tile = 1;
            while (tile * 2 <= maxTileSize) tile *= 2;
            tile = Math.Max(tile, 16);
            return (Math.Min(tile, m), Math.Min(tile, n), Math.Min(tile, k));
        }

        /// <summary>[Obsolete] Use the generic
        /// <see cref="EstimateCacheMisses{T}(int, int, int, int)"/>
        /// overload. The element size is now derived from T.</summary>
        [Obsolete("Use EstimateCacheMisses<T>(int, int, int, int) where T : unmanaged. Replaced in issue #294 so the elements-per-line math uses the actual element size, not a hardcoded sizeof(float).")]
        public static double EstimateCacheMisses(int dataSize, int accessStride, int cacheSize, int cacheLineSize)
            => EstimateCacheMisses<float>(dataSize, accessStride, cacheSize, cacheLineSize);
    }
}
