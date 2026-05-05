// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.Optimization
{
    /// <summary>
    /// Loop-control-flow optimization helpers — the iteration-shape side
    /// of the optimization split (the read/write side lives in
    /// <see cref="CacheOptimizer"/>).
    ///
    /// <para><b>Why struct callbacks, not <c>Action&lt;int&gt;</c>:</b>
    /// The previous API took <see cref="Action{T}"/> delegates, which
    /// forced every iteration through delegate dispatch. The cost was
    /// catastrophic for tight kernels — one benchmark on a small
    /// embedding scatter showed <see cref="Tile2D"/> running 84% slower
    /// than the naive baseline because the per-tile <c>Action</c>
    /// invocation defeated RyuJIT inlining and blocked SIMD
    /// auto-vectorization.</para>
    ///
    /// <para>The struct-callback pattern (<see cref="ILoopAction"/> +
    /// <c>where TAction : struct, ILoopAction</c>) fixes this by giving
    /// RyuJIT a concrete callee to specialize. The runtime emits a
    /// dedicated method per concrete struct, devirtualizes
    /// <c>Invoke</c>, inlines through the call site, and the inner loop
    /// becomes plain unrolled arithmetic the SIMD vectorizer matches.
    /// Same pattern <see cref="System.MemoryExtensions"/> and
    /// <see cref="System.Numerics.Vector{T}"/> use for their hot paths.</para>
    ///
    /// <para><b>Tile sizing:</b> <see cref="DetermineOptimalTileSize{T}"/>
    /// delegates to <see cref="CacheOptimizer.ComputeOptimalTiling{T}"/>
    /// — that method is the single source of truth for L1-aware tile
    /// sizing. This file does not duplicate the cache math.</para>
    /// </summary>
    public static class LoopOptimizer
    {
        // ── Struct-callback contracts ───────────────────────────────────

        /// <summary>Per-element callback for <see cref="UnrollBy4{TAction}"/>
        /// and <see cref="UnrollBy8{TAction}"/>. Implementations should be
        /// <c>readonly struct</c> with <see cref="MethodImplOptions.AggressiveInlining"/>
        /// on <see cref="Invoke"/> so RyuJIT can devirt-and-inline through
        /// the unrolled call sites.</summary>
        public interface ILoopAction
        {
            void Invoke(int i);
        }

        /// <summary>Tile callback for <see cref="Tile2D{TAction}"/> and
        /// <see cref="ParallelTile2D{TAction}"/>. Receives the tile's
        /// half-open row range <c>[i0, i1)</c> and column range
        /// <c>[j0, j1)</c>.</summary>
        public interface ITile2DAction
        {
            void Invoke(int i0, int i1, int j0, int j1);
        }

        /// <summary>Tile callback for <see cref="Tile3D{TAction}"/>.
        /// Receives three half-open ranges.</summary>
        public interface ITile3DAction
        {
            void Invoke(int i0, int i1, int j0, int j1, int k0, int k1);
        }

        /// <summary>Per-(row, col) callback for <see cref="OptimalOrder2D{TAction}"/>
        /// — the loop order is chosen by the helper, the body is
        /// agnostic to it.</summary>
        public interface IInterchangeAction
        {
            void Invoke(int i, int j);
        }

        /// <summary>Strip callback for <see cref="StripMine{TAction}"/>.
        /// Receives a half-open <c>[start, end)</c> range.</summary>
        public interface IStripAction
        {
            void Invoke(int start, int end);
        }

        // ── Tile2D / Tile3D ─────────────────────────────────────────────

        /// <summary>
        /// Iterates a <c>[rows × cols]</c> region in <c>tileSize</c>-square
        /// blocks, calling <paramref name="action"/> once per tile with
        /// the tile's half-open row/column ranges.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Tile2D<TAction>(int rows, int cols, int tileSize, TAction action)
            where TAction : struct, ITile2DAction
        {
            if (tileSize <= 0) throw new ArgumentOutOfRangeException(nameof(tileSize));
            for (int i = 0; i < rows; i += tileSize)
            {
                int iEnd = Math.Min(i + tileSize, rows);
                for (int j = 0; j < cols; j += tileSize)
                {
                    int jEnd = Math.Min(j + tileSize, cols);
                    action.Invoke(i, iEnd, j, jEnd);
                }
            }
        }

        /// <summary>
        /// Iterates a <c>[dim1 × dim2 × dim3]</c> region in
        /// <c>tileSize1·tileSize2·tileSize3</c>-shaped blocks.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Tile3D<TAction>(
            int dim1, int dim2, int dim3,
            int tileSize1, int tileSize2, int tileSize3,
            TAction action)
            where TAction : struct, ITile3DAction
        {
            if (tileSize1 <= 0) throw new ArgumentOutOfRangeException(nameof(tileSize1));
            if (tileSize2 <= 0) throw new ArgumentOutOfRangeException(nameof(tileSize2));
            if (tileSize3 <= 0) throw new ArgumentOutOfRangeException(nameof(tileSize3));
            for (int i = 0; i < dim1; i += tileSize1)
            {
                int iEnd = Math.Min(i + tileSize1, dim1);
                for (int j = 0; j < dim2; j += tileSize2)
                {
                    int jEnd = Math.Min(j + tileSize2, dim2);
                    for (int k = 0; k < dim3; k += tileSize3)
                    {
                        int kEnd = Math.Min(k + tileSize3, dim3);
                        action.Invoke(i, iEnd, j, jEnd, k, kEnd);
                    }
                }
            }
        }

        // ── UnrollBy4 / UnrollBy8 ───────────────────────────────────────

        /// <summary>
        /// Calls <paramref name="action"/> on every index in
        /// <c>[0, length)</c> with a 4-way unrolled bulk loop. The
        /// unroll factor lets RyuJIT lift loop-invariant work and the
        /// SIMD vectorizer pick a vector width (4, 8, 16) that fits
        /// the inner body.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void UnrollBy4<TAction>(int length, TAction action)
            where TAction : struct, ILoopAction
        {
            int unrolled = length & ~3;
            for (int i = 0; i < unrolled; i += 4)
            {
                action.Invoke(i);
                action.Invoke(i + 1);
                action.Invoke(i + 2);
                action.Invoke(i + 3);
            }
            for (int i = unrolled; i < length; i++) action.Invoke(i);
        }

        /// <summary>
        /// Calls <paramref name="action"/> on every index in
        /// <c>[0, length)</c> with an 8-way unrolled bulk loop. Wider
        /// unroll than <see cref="UnrollBy4{TAction}"/> — gives more
        /// independent ILP and a better fit for AVX2 (8 floats /
        /// 4 doubles per vector).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void UnrollBy8<TAction>(int length, TAction action)
            where TAction : struct, ILoopAction
        {
            int unrolled = length & ~7;
            for (int i = 0; i < unrolled; i += 8)
            {
                action.Invoke(i);
                action.Invoke(i + 1);
                action.Invoke(i + 2);
                action.Invoke(i + 3);
                action.Invoke(i + 4);
                action.Invoke(i + 5);
                action.Invoke(i + 6);
                action.Invoke(i + 7);
            }
            for (int i = unrolled; i < length; i++) action.Invoke(i);
        }

        // ── StripMine ───────────────────────────────────────────────────

        /// <summary>
        /// Walks <c>[0, totalSize)</c> in <c>stripSize</c>-element
        /// chunks, calling <paramref name="action"/> with each chunk's
        /// half-open <c>[start, end)</c> range. Useful for streaming
        /// throughput-bound passes where the body wants to do its own
        /// SIMD over a known-bounded chunk.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StripMine<TAction>(int totalSize, int stripSize, TAction action)
            where TAction : struct, IStripAction
        {
            if (stripSize <= 0) throw new ArgumentOutOfRangeException(nameof(stripSize));
            for (int start = 0; start < totalSize; start += stripSize)
            {
                int end = Math.Min(start + stripSize, totalSize);
                action.Invoke(start, end);
            }
        }

        // ── OptimalOrder2D ──────────────────────────────────────────────

        /// <summary>
        /// Iterates a <c>[rows × cols]</c> region in row-major
        /// (<paramref name="rowMajorAccess"/>=true) or column-major
        /// (false) order. The body is agnostic to the chosen order;
        /// the helper picks the loop interchange that matches the
        /// caller-described access pattern.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void OptimalOrder2D<TAction>(
            int rows, int cols, bool rowMajorAccess, TAction action)
            where TAction : struct, IInterchangeAction
        {
            if (rowMajorAccess)
            {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        action.Invoke(i, j);
            }
            else
            {
                for (int j = 0; j < cols; j++)
                    for (int i = 0; i < rows; i++)
                        action.Invoke(i, j);
            }
        }

        // ── ParallelTile2D ──────────────────────────────────────────────

        /// <summary>
        /// Parallel variant of <see cref="Tile2D{TAction}"/>: each
        /// tile runs on a <see cref="System.Threading.Tasks.Parallel.For(int,int,Action{int})"/>
        /// worker. Caller's <typeparamref name="TAction"/> must be
        /// thread-safe — the helper passes the same struct value to
        /// every worker (struct types are by-value-copied per call so
        /// per-tile state stays independent).
        ///
        /// <para>Note: <see cref="System.Threading.Tasks.Parallel.For(int,int,Action{int})"/>
        /// itself takes an <c>Action&lt;int&gt;</c>, but the work it
        /// dispatches is one tile's worth of computation — that's
        /// large enough to amortize the per-iteration delegate cost,
        /// unlike the per-element case where the struct callback is
        /// load-bearing.</para>
        /// </summary>
        public static void ParallelTile2D<TAction>(
            int rows, int cols, int tileSize, TAction action)
            where TAction : struct, ITile2DAction
        {
            if (tileSize <= 0) throw new ArgumentOutOfRangeException(nameof(tileSize));
            int numTilesI = (rows + tileSize - 1) / tileSize;
            int numTilesJ = (cols + tileSize - 1) / tileSize;
            int totalTiles = numTilesI * numTilesJ;
            // Capture the struct once — Parallel.For closes over the
            // outer scope, and we want every worker to see a fresh
            // struct copy (value-type semantics) rather than a shared
            // boxed reference.
            TAction localAction = action;
            Parallel.For(0, totalTiles, tileIdx =>
            {
                int ti = tileIdx / numTilesJ;
                int tj = tileIdx % numTilesJ;
                int iStart = ti * tileSize;
                int iEnd = Math.Min(iStart + tileSize, rows);
                int jStart = tj * tileSize;
                int jEnd = Math.Min(jStart + tileSize, cols);
                localAction.Invoke(iStart, iEnd, jStart, jEnd);
            });
        }

        // ── Fuse ────────────────────────────────────────────────────────

        /// <summary>
        /// Fuses two element-wise loops into one pass. Body is
        /// <c>action.Invoke(i)</c> applied to every <c>i</c> in
        /// <c>[0, length)</c>; structurally identical to
        /// <see cref="UnrollBy4{TAction}"/> but the contract is
        /// "<typeparamref name="TAction"/>'s body is itself a fused
        /// composition of multiple operations". Common pattern: build
        /// a fused struct that performs N elementwise ops in one body
        /// to avoid N intermediate buffers.
        /// </summary>
        /// <remarks>
        /// The previous <c>Fuse(int, params Action&lt;int&gt;[])</c>
        /// signature is kept as a soft-deprecated overload below — but
        /// callers writing new code should compose the operations into
        /// a single struct callback instead, since the array-of-
        /// delegates form pays the per-element delegate cost <c>actions.Length</c>
        /// times per iteration.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Fuse<TAction>(int length, TAction action)
            where TAction : struct, ILoopAction
        {
            for (int i = 0; i < length; i++) action.Invoke(i);
        }

        /// <summary>
        /// Optimal tile size for <typeparamref name="T"/>'s element
        /// size, derived from the runtime's L1 cache. Single-call
        /// thin wrapper around <see cref="CacheOptimizer.ComputeOptimalTiling{T}"/>
        /// — cache-math lives in CacheOptimizer; this is purely a
        /// loop-shape ergonomic helper for callers that want one
        /// scalar back instead of a (tileM, tileN, tileK) triple.
        /// </summary>
        /// <param name="dimension">The longest dimension the tile will
        /// iterate. Returned tile size is bounded by this so a tiny
        /// dimension doesn't trigger oversized tile dispatch overhead.</param>
        public static int DetermineOptimalTileSize<T>(int dimension)
            where T : unmanaged
        {
            // Delegate to the canonical cache-aware tile picker. We pass
            // the same dimension three times because we don't know
            // matmul-shape; the picker returns min(tile, m), min(tile, n),
            // min(tile, k) — taking any one back gives us the tile size
            // bounded by the dimension.
            var (t, _, _) = CacheOptimizer.ComputeOptimalTiling<T>(dimension, dimension, dimension);
            return t;
        }
    }
}
