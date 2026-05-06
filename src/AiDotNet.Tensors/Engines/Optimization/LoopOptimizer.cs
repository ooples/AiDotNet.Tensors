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
            if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols));
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
            if (dim1 < 0) throw new ArgumentOutOfRangeException(nameof(dim1));
            if (dim2 < 0) throw new ArgumentOutOfRangeException(nameof(dim2));
            if (dim3 < 0) throw new ArgumentOutOfRangeException(nameof(dim3));
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
            if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
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
            if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
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
            if (totalSize < 0) throw new ArgumentOutOfRangeException(nameof(totalSize));
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
            if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols));
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
        /// tile runs on a worker dispatched via the repo's
        /// <see cref="AiDotNet.Tensors.Helpers.CpuParallelSettings.LightweightParallel"/>
        /// path so the persistent worker pool and configured
        /// <see cref="AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism"/>
        /// cap are honoured (matches the convention used by
        /// <see cref="AiDotNet.Tensors.Helpers.SimdConvHelper"/> and
        /// <see cref="AiDotNet.Tensors.Helpers.Im2ColHelper"/>).
        ///
        /// <para><b>Per-worker callback isolation</b>: each worker
        /// makes its OWN copy of the struct callback before invoking,
        /// so a callback with mutable instance state observes
        /// per-worker semantics (no race on a shared closure field).
        /// Same value-type semantics as the sequential
        /// <see cref="Tile2D{TAction}"/>, just dispatched in parallel.
        /// Callbacks that intentionally share state (e.g. accumulators
        /// into a shared array passed by reference) need their own
        /// thread-safety story; this helper guarantees the struct
        /// itself isn't shared.</para>
        /// </summary>
        public static void ParallelTile2D<TAction>(
            int rows, int cols, int tileSize, TAction action)
            where TAction : struct, ITile2DAction
        {
            if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols));
            if (tileSize <= 0) throw new ArgumentOutOfRangeException(nameof(tileSize));
            int numTilesI = (rows + tileSize - 1) / tileSize;
            int numTilesJ = (cols + tileSize - 1) / tileSize;
            int totalTiles = numTilesI * numTilesJ;
            // Capture the original action ONCE so the closure has a
            // single read-only reference. Each worker then COPIES the
            // struct into its own local — value-type semantics mean
            // the workers can't race on the same struct fields.
            TAction sharedAction = action;
            int sharedNumTilesJ = numTilesJ;
            int sharedTileSize = tileSize;
            int sharedRows = rows;
            int sharedCols = cols;
            AiDotNet.Tensors.Helpers.CpuParallelSettings.LightweightParallel(totalTiles, tileIdx =>
            {
                // Per-worker struct copy — assignment to a local copies
                // the value-type, isolating mutations to this worker.
                TAction localAction = sharedAction;
                int ti = tileIdx / sharedNumTilesJ;
                int tj = tileIdx % sharedNumTilesJ;
                int iStart = ti * sharedTileSize;
                int iEnd = Math.Min(iStart + sharedTileSize, sharedRows);
                int jStart = tj * sharedTileSize;
                int jEnd = Math.Min(jStart + sharedTileSize, sharedCols);
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

        // ── [Obsolete] delegate-based compatibility shims ──────────────
        //
        // The previous public surface took Action<int> / Action<int,int,...>
        // delegates. Issue #294 replaced those with struct-callback
        // generics for JIT inlining. To avoid a hard source-break for
        // package consumers, the old signatures are kept here as
        // [Obsolete]-marked forwarding overloads that wrap the caller's
        // delegate in a private DelegateAdapter struct. Compiler emits
        // warning CS0618 with the migration hint pointing to the struct-
        // callback path.

        private readonly struct DelegateLoopAction : ILoopAction
        {
            private readonly Action<int> _fn;
            public DelegateLoopAction(Action<int> fn) { _fn = fn; }
            public void Invoke(int i) => _fn(i);
        }

        private readonly struct DelegateTile2DAction : ITile2DAction
        {
            private readonly Action<int, int, int, int> _fn;
            public DelegateTile2DAction(Action<int, int, int, int> fn) { _fn = fn; }
            public void Invoke(int i0, int i1, int j0, int j1) => _fn(i0, i1, j0, j1);
        }

        private readonly struct DelegateInterchangeAction : IInterchangeAction
        {
            private readonly Action<int, int> _fn;
            public DelegateInterchangeAction(Action<int, int> fn) { _fn = fn; }
            public void Invoke(int i, int j) => _fn(i, j);
        }

        private readonly struct DelegateStripAction : IStripAction
        {
            private readonly Action<int, int> _fn;
            public DelegateStripAction(Action<int, int> fn) { _fn = fn; }
            public void Invoke(int start, int end) => _fn(start, end);
        }

        /// <summary>[Obsolete] Use the struct-callback overload that
        /// takes <see cref="ITile2DAction"/> for JIT-inlined dispatch.
        /// This delegate-based shim is retained for one transition
        /// release; new code should not call it.</summary>
        [Obsolete("Use Tile2D<TAction>(int, int, int, TAction) where TAction : struct, ITile2DAction. The delegate overload prevents JIT inlining of the inner body and was replaced in issue #294.")]
        public static void Tile2D(int rows, int cols, int tileSize, Action<int, int, int, int> tileAction)
        {
            if (tileAction is null) throw new ArgumentNullException(nameof(tileAction));
            Tile2D(rows, cols, tileSize, new DelegateTile2DAction(tileAction));
        }

        /// <summary>[Obsolete] Use the struct-callback overload taking
        /// <see cref="ILoopAction"/>.</summary>
        [Obsolete("Use UnrollBy4<TAction>(int, TAction) where TAction : struct, ILoopAction. Replaced in issue #294 for JIT inlining.")]
        public static void UnrollBy4(int length, Action<int> action)
        {
            if (action is null) throw new ArgumentNullException(nameof(action));
            UnrollBy4(length, new DelegateLoopAction(action));
        }

        /// <summary>[Obsolete] Use the struct-callback overload taking
        /// <see cref="ILoopAction"/>.</summary>
        [Obsolete("Use UnrollBy8<TAction>(int, TAction) where TAction : struct, ILoopAction. Replaced in issue #294 for JIT inlining.")]
        public static void UnrollBy8(int length, Action<int> action)
        {
            if (action is null) throw new ArgumentNullException(nameof(action));
            UnrollBy8(length, new DelegateLoopAction(action));
        }

        /// <summary>[Obsolete] Use the struct-callback overload taking
        /// <see cref="IStripAction"/>.</summary>
        [Obsolete("Use StripMine<TAction>(int, int, TAction) where TAction : struct, IStripAction. Replaced in issue #294 for JIT inlining.")]
        public static void StripMine(int totalSize, int stripSize, Action<int, int> stripAction)
        {
            if (stripAction is null) throw new ArgumentNullException(nameof(stripAction));
            StripMine(totalSize, stripSize, new DelegateStripAction(stripAction));
        }

        /// <summary>[Obsolete] The variadic delegate-array form is
        /// retained for one transition release. New code should compose
        /// the operations into a single <see cref="ILoopAction"/>
        /// struct callback to avoid the per-element delegate dispatch
        /// over <c>actions.Length</c> entries.</summary>
        [Obsolete("Use Fuse<TAction>(int, TAction) where TAction : struct, ILoopAction with a single composed callback. The variadic-delegate form pays delegate dispatch per (element, action) pair and was replaced in issue #294.")]
        public static void Fuse(int length, params Action<int>[] actions)
        {
            if (actions is null) throw new ArgumentNullException(nameof(actions));
            for (int i = 0; i < length; i++)
                foreach (var a in actions) a(i);
        }

        /// <summary>[Obsolete] Use the struct-callback overload taking
        /// <see cref="IInterchangeAction"/>.</summary>
        [Obsolete("Use OptimalOrder2D<TAction>(int, int, bool, TAction) where TAction : struct, IInterchangeAction. Replaced in issue #294 for JIT inlining.")]
        public static void OptimalOrder2D(int rows, int cols, bool rowMajorAccess, Action<int, int> action)
        {
            if (action is null) throw new ArgumentNullException(nameof(action));
            OptimalOrder2D(rows, cols, rowMajorAccess, new DelegateInterchangeAction(action));
        }

        /// <summary>[Obsolete] Use the struct-callback overload taking
        /// <see cref="ITile2DAction"/>.</summary>
        [Obsolete("Use ParallelTile2D<TAction>(int, int, int, TAction) where TAction : struct, ITile2DAction. Replaced in issue #294 for JIT inlining + per-worker struct isolation.")]
        public static void ParallelTile2D(int rows, int cols, int tileSize, Action<int, int, int, int> tileAction)
        {
            if (tileAction is null) throw new ArgumentNullException(nameof(tileAction));
            ParallelTile2D(rows, cols, tileSize, new DelegateTile2DAction(tileAction));
        }

        /// <summary>[Obsolete] Non-generic form deferred to a single
        /// element size. Use the generic <see cref="DetermineOptimalTileSize{T}"/>
        /// overload so the cache picker sees the right
        /// <c>Unsafe.SizeOf&lt;T&gt;</c>.</summary>
        [Obsolete("Use DetermineOptimalTileSize<T>(int) where T : unmanaged so the L1 picker uses the right element size for your tensor. The non-generic form assumes 4-byte elements.")]
        public static int DetermineOptimalTileSize(int dimension, int elementSize = 4)
        {
            // Old behaviour: replicate the local sqrt(L1 / (2 * size))
            // formula for source compat. New code should call the
            // generic overload.
            var caps = PlatformDetector.Capabilities;
            int l1 = caps?.L1CacheSize ?? 32 * 1024;
            int maxElems = l1 / Math.Max(1, 2 * elementSize);
            int tile = 16;
            while (tile * tile * 2 < maxElems && tile < dimension) tile *= 2;
            return Math.Min(tile, dimension);
        }
    }
}
