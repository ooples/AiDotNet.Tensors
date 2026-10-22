// Copyright (c) AiDotNet. All rights reserved.
// The deterministic scatter-add as a stable sort and an ordered segment sum (CudaSortedScatterKernels): bit-identical
// to the scanning deterministic kernels, which cost O(destinations x sources).

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    // AIDOTNET_SORTED_SCATTER=0: the deterministic scatters take the scanning kernels again.
    private static readonly bool s_sortedScatter = Environment.GetEnvironmentVariable("AIDOTNET_SORTED_SCATTER") != "0";

    /// <summary>Overrides AIDOTNET_SORTED_SCATTER for this backend (tooling and tests); null follows it.</summary>
    internal bool? SortedScatterSwitch { get; set; }

    /// <summary>Number of deterministic scatters served by the sorted kernels, for tests and diagnostics.</summary>
    internal long SortedScatterLaunches => System.Threading.Interlocked.Read(ref _sortedScatterLaunches);
    private long _sortedScatterLaunches;

    private IntPtr _sortedScatterModule;
    private readonly object _sortedScatterLock = new();

    // Scratch, grown to the largest call and reused: two key/value pairs (the passes ping-pong), the per-tile digit
    // counts and the run starts.
    private IGpuBuffer? _sortKeysA, _sortValuesA, _sortKeysB, _sortValuesB, _sortTileCounts, _sortStarts;

    /// <summary>
    /// <c>output[d, f] = (accumulate ? output[d, f] : 0) + sum of source[i, f] over every i with indices[i] == d</c>,
    /// summed in increasing i, for d in [0, destinations): an index outside that range contributes nowhere. The same
    /// result, bit for bit, as embedding_backward_deterministic (accumulate false) and
    /// scatter_add_accumulate_deterministic (accumulate true, one feature), and -- item i reading source row rows[i],
    /// scaled by weights[i], counts[d] taking run d's length -- scatter_add_edges_deterministic and
    /// scatter_mean_activation_deterministic. False -- nothing launched -- when switched
    /// off, the kernels are missing, or the scratch would have to grow inside a stream capture.
    /// </summary>
    internal unsafe bool TrySortedScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int count,
        int destinations, int features, bool accumulate, IGpuBuffer? rows = null, IGpuBuffer? weights = null,
        IGpuBuffer? counts = null)
    {
        if (!(SortedScatterSwitch ?? s_sortedScatter) || count <= 0 || destinations <= 0 || features <= 0 ||
            destinations == int.MaxValue)
            return false;
        // The sort costs ~50 us of launches whatever the size; the scanning kernels cost destinations x features x count
        // additions at ~24 a microsecond per thread, ~1.5e5 a microsecond in all once the device is full (GTX 1660 Ti).
        // Below ~150 us of scanning they win (300 tokens into 7 rows: 29 us against 51).
        long cells = (long)destinations * features;
        double scanMicroseconds = (double)cells * count / Math.Min(1.5e5, 24.0 * cells);
        if (scanMicroseconds < 150 || cells > int.MaxValue) return false;
        if (!_kernelCache.TryGetValue("sorted_scatter_init", out var init) ||
            !_kernelCache.TryGetValue("sorted_scatter_histogram", out var histogram) ||
            !_kernelCache.TryGetValue("sorted_scatter_scan", out var scan) ||
            !_kernelCache.TryGetValue("sorted_scatter_place", out var place) ||
            !_kernelCache.TryGetValue("sorted_scatter_starts", out var startsKernel) ||
            !_kernelCache.TryGetValue("sorted_scatter_sum", out var sumKernel))
            return false;
        int tiles = (count + CudaSortedScatterKernels.Tile - 1) / CudaSortedScatterKernels.Tile;
        int countSlots = checked(tiles * CudaSortedScatterKernels.Buckets);
        // Keys run to `destinations` (the out-of-range sentinel), so the sort covers its bit length.
        int bits = 32 - LeadingZeros((uint)destinations);
        int passes = (bits + 3) / 4;

        lock (_sortedScatterLock)
        {
            if (!EnsureSortScratch(count, countSlots, destinations + 1) ||
                _sortKeysA is not { } keysA || _sortValuesA is not { } valuesA || _sortKeysB is not { } keysB ||
                _sortValuesB is not { } valuesB || _sortTileCounts is not { } tileCounts || _sortStarts is not { } starts)
                return false;
            using var _ = PushContext();
            uint itemBlocks = (uint)tiles;
            IntPtr idx = indices.Handle, ka = keysA.Handle, va = valuesA.Handle, kb = keysB.Handle, vb = valuesB.Handle;
            IntPtr tc = tileCounts.Handle, st = starts.Handle, src = source.Handle, outp = output.Handle;
            int n = count, dest = destinations, feat = features, acc = accumulate ? 1 : 0, t = tiles, slots = countSlots;

            void** initArgs = stackalloc void*[5];
            initArgs[0] = &idx; initArgs[1] = &ka; initArgs[2] = &va; initArgs[3] = &n; initArgs[4] = &dest;
            LaunchKernel(init, itemBlocks, CudaSortedScatterKernels.Tile, initArgs);

            void** histArgs = stackalloc void*[5];
            void** scanArgs = stackalloc void*[2];
            void** placeArgs = stackalloc void*[8];
            for (int pass = 0; pass < passes; pass++)
            {
                int shift = pass * 4;
                IntPtr keysIn = pass % 2 == 0 ? ka : kb, valuesIn = pass % 2 == 0 ? va : vb;
                IntPtr keysOut = pass % 2 == 0 ? kb : ka, valuesOut = pass % 2 == 0 ? vb : va;
                histArgs[0] = &keysIn; histArgs[1] = &tc; histArgs[2] = &n; histArgs[3] = &shift; histArgs[4] = &t;
                LaunchKernel(histogram, itemBlocks, CudaSortedScatterKernels.Tile, histArgs);
                scanArgs[0] = &tc; scanArgs[1] = &slots;
                LaunchKernel(scan, 1, 1024, scanArgs);
                placeArgs[0] = &keysIn; placeArgs[1] = &valuesIn; placeArgs[2] = &keysOut; placeArgs[3] = &valuesOut;
                placeArgs[4] = &tc; placeArgs[5] = &n; placeArgs[6] = &shift; placeArgs[7] = &t;
                LaunchKernel(place, itemBlocks, CudaSortedScatterKernels.Tile, placeArgs);
            }
            IntPtr sortedKeys = passes % 2 == 0 ? ka : kb, sortedValues = passes % 2 == 0 ? va : vb;

            void** startArgs = stackalloc void*[4];
            startArgs[0] = &sortedKeys; startArgs[1] = &st; startArgs[2] = &n; startArgs[3] = &dest;
            LaunchKernel(startsKernel, (uint)((count + 1 + 255) / 256), 256, startArgs);

            IntPtr rowPtr = rows?.Handle ?? IntPtr.Zero, weightPtr = weights?.Handle ?? IntPtr.Zero;
            IntPtr countPtr = counts?.Handle ?? IntPtr.Zero;
            void** sumArgs = stackalloc void*[10];
            sumArgs[0] = &src; sumArgs[1] = &sortedValues; sumArgs[2] = &st; sumArgs[3] = &outp; sumArgs[4] = &dest;
            sumArgs[5] = &feat; sumArgs[6] = &acc; sumArgs[7] = &rowPtr; sumArgs[8] = &weightPtr; sumArgs[9] = &countPtr;
            LaunchKernel(sumKernel, checked((uint)((cells + 255) / 256)), 256, sumArgs);
        }
        System.Threading.Interlocked.Increment(ref _sortedScatterLaunches);
        return true;
    }

    private static readonly string[] s_embeddingNames = { "ids", "table", "out" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildEmbedding =
        key => CodegenBackendPrograms.Embedding(key.D0, key.D1, key.D2);

    /// <summary>The key of an embedding lookup: tokens, vocabulary (the table's rows), width.</summary>
    internal static DeclaredKey EmbeddingKey(int tokens, int vocabulary, int width) =>
        new(DeclaredFamily.Embedding, tokens, vocabulary, width, 0, 0, 0);

    /// <summary>
    /// Embedding / Gather through the declared lookup (CodegenBackendPrograms.Embedding): one thread per output
    /// element, so a warp reads a row contiguously where the hand-written kernel gave each thread a whole row. The
    /// vocabulary is the table's row count. False -- nothing launched -- when the route is off or the shape declines.
    /// </summary>
    private bool TryDeclaredEmbedding(IGpuBuffer indices, IGpuBuffer table, IGpuBuffer output, int tokens, int width)
    {
        // A lookup of a few thousand elements is one launch either way, and the declared call's bookkeeping measured
        // ~0.4 us more (300 x 5: 9.1 against 8.8 us), so it keeps the hand-written kernel.
        if (tokens <= 0 || width <= 0 || (long)tokens * width < 16384 || table.Size < width || indices.Size < tokens)
            return false;
        int vocabulary = table.Size / width;
        return TryRunDeclaredProgram(EmbeddingKey(tokens, vocabulary, width), s_buildEmbedding, s_embeddingNames,
            new DeclaredBindings(indices, table, output));
    }

    private static readonly string[] s_scatterAddNames = { "indices", "source", "out" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildScatterAdd =
        key => CodegenBackendPrograms.ScatterAddDeterministic(key.D0, key.D1, key.D2);

    /// <summary>The key of a deterministic scatter-add: count, destinations, features.</summary>
    internal static DeclaredKey ScatterAddDeterministicKey(int count, int destinations, int features) =>
        new(DeclaredFamily.ScatterAddDeterministic, count, destinations, features, 0, 0, 0);

    /// <summary>
    /// The deterministic scatter-add through its declared program (#890 phase 4e): the same result, bit for bit, as
    /// the scanning deterministic kernels, and with no atomic emitted at all. The program reduces over every source
    /// for each destination, so it costs destinations x count: above the crossover the sorted kernels
    /// (<see cref="TrySortedScatterAdd"/>) remain the faster route and this one declines, as they do to it when the
    /// destination count is small. False -- nothing launched -- when the route is off or the shape declines.
    /// </summary>
    internal bool TryDeclaredScatterAddDeterministic(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int count, int destinations, int features)
    {
        if (count <= 0 || destinations <= 0 || features <= 0 || indices.Size < count ||
            (long)source.Size < (long)count * features || (long)output.Size < (long)destinations * features)
            return false;
        // The scan is quadratic in the destination count, so it is admitted only where that product stays near the
        // work the sorted path does anyway (its sort is count log count over the same rows).
        if ((long)destinations * count > 1L << 22) return false;
        return TryRunDeclaredProgram(ScatterAddDeterministicKey(count, destinations, features), s_buildScatterAdd,
            s_scatterAddNames, new DeclaredBindings(indices, source, output));
    }

    /// <summary>The deterministic scatter-add as a resident-program stage.</summary>
    internal DeclaredResidentStage ScatterAddDeterministicStage(IGpuBuffer source, IGpuBuffer indices,
        IGpuBuffer output, int count, int destinations, int features) =>
        new(ScatterAddDeterministicKey(count, destinations, features), s_buildScatterAdd, s_scatterAddNames,
            new DeclaredBindings(indices, source, output));

    private static readonly string[] s_indexSelectNames = { "indices", "source", "out" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildIndexSelect =
        key => CodegenBackendPrograms.IndexSelect(key.D0, key.D1, key.D2, key.D3);

    internal static DeclaredKey IndexSelectKey(int outer, int sourceAxis, int indexAxis, int inner) =>
        new(DeclaredFamily.IndexSelect, outer, sourceAxis, indexAxis, inner, 0, 0);

    /// <summary>
    /// IndexSelect through the declared gather (CodegenBackendPrograms.IndexSelect); below the same size as Embedding's
    /// it keeps the hand-written kernel. False -- nothing launched -- when the route is off or the shape declines.
    /// </summary>
    internal bool TryDeclaredIndexSelect(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int outer,
        int sourceAxis, int indexAxis, int inner)
    {
        if (outer <= 0 || sourceAxis <= 0 || indexAxis <= 0 || inner <= 0 ||
            (long)outer * indexAxis * inner < 16384 || indices.Size < indexAxis ||
            (long)source.Size < (long)outer * sourceAxis * inner)
            return false;
        return TryRunDeclaredProgram(IndexSelectKey(outer, sourceAxis, indexAxis, inner), s_buildIndexSelect,
            s_indexSelectNames, new DeclaredBindings(indices, source, output));
    }

    /// <summary>IndexSelect as a resident-program stage, in IndexSelect's contract.</summary>
    internal DeclaredResidentStage IndexSelectStage(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int outer,
        int sourceAxis, int indexAxis, int inner) =>
        new(IndexSelectKey(outer, sourceAxis, indexAxis, inner), s_buildIndexSelect, s_indexSelectNames,
            new DeclaredBindings(indices, source, output));

    private static readonly string[] s_takeAlongDimNames = { "indices", "input", "out" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildTakeAlongDim =
        key => CodegenBackendPrograms.TakeAlongDim(key.D0, key.D1, key.D2, key.D3);

    internal static DeclaredKey TakeAlongDimKey(int outer, int axisOut, int inner, int axisIn) =>
        new(DeclaredFamily.TakeAlongDim, outer, axisOut, inner, axisIn, 0, 0);

    /// <summary>
    /// TakeAlongDim through the declared gather (CodegenBackendPrograms.TakeAlongDim, float indices truncated as the
    /// hand-written kernel's (int)); below the same size as Embedding's it keeps the hand-written kernel. False --
    /// nothing launched -- when the route is off or the shape declines.
    /// </summary>
    internal bool TryDeclaredTakeAlongDim(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int outer,
        int axisOut, int inner, int axisIn)
    {
        long count = (long)outer * axisOut * inner;
        if (outer <= 0 || axisOut <= 0 || inner <= 0 || axisIn <= 0 || count < 16384 || indices.Size < count ||
            (long)input.Size < (long)outer * axisIn * inner)
            return false;
        return TryRunDeclaredProgram(TakeAlongDimKey(outer, axisOut, inner, axisIn), s_buildTakeAlongDim,
            s_takeAlongDimNames, new DeclaredBindings(indices, input, output));
    }

    /// <summary>TakeAlongDim as a resident-program stage, in TakeAlongDim's contract.</summary>
    internal DeclaredResidentStage TakeAlongDimStage(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int outer,
        int axisOut, int inner, int axisIn) =>
        new(TakeAlongDimKey(outer, axisOut, inner, axisIn), s_buildTakeAlongDim, s_takeAlongDimNames,
            new DeclaredBindings(indices, input, output));

    /// <summary>The embedding lookup as a resident-program stage, in Embedding's contract.</summary>
    internal DeclaredResidentStage EmbeddingStage(IGpuBuffer indices, IGpuBuffer table, IGpuBuffer output, int tokens, int width) =>
        new(EmbeddingKey(tokens, table.Size / width, width), s_buildEmbedding, s_embeddingNames,
            new DeclaredBindings(indices, table, output));

    // grid_sample's items, rows, weights and per-cell sums, grown and reused like the sort scratch.
    private IGpuBuffer? _gridItemKeys, _gridItemRows, _gridItemWeights, _gridCellSums;

    /// <summary>
    /// grid_sample's deterministic input gradient through the sorted scatter: each output position's four corners are
    /// items keyed by the input cell they add to, summed in the scanning kernel's order -- output position, then corner
    /// -- with its arithmetic, so the result is bit-identical to grid_sample_backward_grad_input_deterministic, which
    /// scanned every output position for every input cell. False -- nothing launched -- as TrySortedScatterAdd.
    /// </summary>
    internal unsafe bool TrySortedGridSampleGradInput(IGpuBuffer gradOutput, IGpuBuffer grid, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth, int paddingMode, bool alignCorners)
    {
        if (!(SortedScatterSwitch ?? s_sortedScatter) || batch <= 0 || channels <= 0 || inHeight <= 0 || inWidth <= 0 ||
            outHeight <= 0 || outWidth <= 0 ||
            !_kernelCache.TryGetValue("grid_sample_scatter_items", out var itemsKernel) ||
            !_kernelCache.TryGetValue("grid_sample_scatter_finish", out var finishKernel))
            return false;
        long positions = (long)batch * outHeight * outWidth, items = 4 * positions;
        long cells = (long)batch * inHeight * inWidth;
        if (items > int.MaxValue || items * channels > int.MaxValue || cells * channels > int.MaxValue) return false;
        lock (_sortedScatterLock)
        {
            if (!EnsureGridScratch((int)items, (int)(items * channels), (int)(cells * channels)) ||
                _gridItemKeys is not { } keys || _gridItemRows is not { } rows || _gridItemWeights is not { } weights ||
                _gridCellSums is not { } sums)
                return false;
            using var _ = PushContext();
            IntPtr go = gradOutput.Handle, gr = grid.Handle, k = keys.Handle, r = rows.Handle, wt = weights.Handle;
            int b = batch, c = channels, ih = inHeight, iw = inWidth, oh = outHeight, ow = outWidth, pad = paddingMode;
            int align = alignCorners ? 1 : 0;
            void** itemArgs = stackalloc void*[13];
            itemArgs[0] = &go; itemArgs[1] = &gr; itemArgs[2] = &k; itemArgs[3] = &r; itemArgs[4] = &wt; itemArgs[5] = &b;
            itemArgs[6] = &c; itemArgs[7] = &ih; itemArgs[8] = &iw; itemArgs[9] = &oh; itemArgs[10] = &ow;
            itemArgs[11] = &pad; itemArgs[12] = &align;
            LaunchKernel(itemsKernel, (uint)((positions + 255) / 256), 256, itemArgs);
            // The sort takes the same (re-entrant) lock; its threshold may decline a small case, which then takes the
            // scanning kernel -- the items computed here are simply unused.
            if (!TrySortedScatterAdd(rows, keys, sums, (int)items, (int)cells, channels, accumulate: false,
                    weights: weights))
                return false;
            IntPtr s = sums.Handle, gi = gradInput.Handle;
            void** finishArgs = stackalloc void*[6];
            finishArgs[0] = &s; finishArgs[1] = &gi; finishArgs[2] = &b; finishArgs[3] = &c; finishArgs[4] = &ih;
            finishArgs[5] = &iw;
            LaunchKernel(finishKernel, (uint)((cells * channels + 255) / 256), 256, finishArgs);
        }
        return true;
    }

    private bool EnsureGridScratch(int items, int rowFloats, int cellFloats)
    {
        bool Fits(IGpuBuffer? buffer, int size) => buffer is not null && buffer.Size >= size;
        if (Fits(_gridItemKeys, items) && Fits(_gridItemRows, rowFloats) && Fits(_gridItemWeights, items) &&
            Fits(_gridCellSums, cellFloats))
            return true;
        if (IsStreamCapturing()) return false;
        if (!Fits(_gridItemKeys, items)) { _gridItemKeys?.Dispose(); _gridItemKeys = AllocateIntBuffer(items); }
        if (!Fits(_gridItemRows, rowFloats)) { _gridItemRows?.Dispose(); _gridItemRows = AllocateBuffer(rowFloats); }
        if (!Fits(_gridItemWeights, items)) { _gridItemWeights?.Dispose(); _gridItemWeights = AllocateBuffer(items); }
        if (!Fits(_gridCellSums, cellFloats)) { _gridCellSums?.Dispose(); _gridCellSums = AllocateBuffer(cellFloats); }
        return true;
    }

    private static int LeadingZeros(uint value)
    {
        int zeros = 0;
        for (uint bit = 0x80000000u; bit != 0 && (value & bit) == 0; bit >>= 1) zeros++;
        return zeros;
    }

    // Grows the scratch to hold this call (doubling, so a growing batch reallocates rarely); false under a capture,
    // which cannot allocate.
    private bool EnsureSortScratch(int count, int countSlots, int startSlots)
    {
        bool Fits(IGpuBuffer? buffer, int size) => buffer is not null && buffer.Size >= size;
        if (Fits(_sortKeysA, count) && Fits(_sortTileCounts, countSlots) && Fits(_sortStarts, startSlots)) return true;
        if (IsStreamCapturing()) return false;
        IGpuBuffer Grow(IGpuBuffer? old, int size)
        {
            if (Fits(old, size) && old is not null) return old;
            old?.Dispose();
            return AllocateIntBuffer(Math.Max(size, old is null ? size : (int)Math.Min(int.MaxValue, 2L * old.Size)));
        }
        _sortKeysA = Grow(_sortKeysA, count);
        _sortValuesA = Grow(_sortValuesA, count);
        _sortKeysB = Grow(_sortKeysB, count);
        _sortValuesB = Grow(_sortValuesB, count);
        _sortTileCounts = Grow(_sortTileCounts, countSlots);
        _sortStarts = Grow(_sortStarts, startSlots);
        return true;
    }

    private void DisposeSortedScatter()
    {
        foreach (var buffer in new[] { _sortKeysA, _sortValuesA, _sortKeysB, _sortValuesB, _sortTileCounts, _sortStarts,
                     _gridItemKeys, _gridItemRows, _gridItemWeights, _gridCellSums })
            buffer?.Dispose();
        _gridItemKeys = _gridItemRows = _gridItemWeights = _gridCellSums = null;
        _sortKeysA = _sortValuesA = _sortKeysB = _sortValuesB = _sortTileCounts = _sortStarts = null;
        if (_sortedScatterModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_sortedScatterModule);
            _sortedScatterModule = IntPtr.Zero;
        }
    }
}
