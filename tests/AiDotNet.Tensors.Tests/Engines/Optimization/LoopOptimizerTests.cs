// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Threading;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// Tests for the issue #294 struct-callback refactor of
/// <see cref="LoopOptimizer"/>. Each test defines a small
/// <c>readonly struct</c> implementing the matching contract
/// (<see cref="LoopOptimizer.ILoopAction"/> /
/// <see cref="LoopOptimizer.ITile2DAction"/> / etc.) — the same
/// shape production callers will use to get JIT devirtualization +
/// inlining + SIMD vectorization through the loop helper.
/// </summary>
public class LoopOptimizerTests
{
    // ── ILoopAction ─────────────────────────────────────────────────

    /// <summary>Counter that increments on every Invoke. Used as the
    /// canonical exercise for unroll / strip / fuse coverage tests.</summary>
    private readonly struct CountAction : LoopOptimizer.ILoopAction
    {
        private readonly StrongBox _box;
        public CountAction(StrongBox box) { _box = box; }
        public void Invoke(int i) => Interlocked.Increment(ref _box.Value);
    }

    /// <summary>Mutable counter referenced by the struct callbacks via
    /// a class wrapper — needed because struct fields can't directly
    /// hold a writable reference. Simpler than <c>ref</c> fields and
    /// works on every supported framework target.</summary>
    private sealed class StrongBox { public int Value; }

    [Fact]
    public void UnrollBy4_InvokesActionForAllIndices()
    {
        var box = new StrongBox();
        LoopOptimizer.UnrollBy4(17, new CountAction(box));
        Assert.Equal(17, box.Value);
    }

    [Fact]
    public void UnrollBy4_PowerOfFour_AllIndicesHit_NoTail()
    {
        // Length divisible by 4 → all iterations go through the
        // unrolled bulk; the tail loop should run zero times.
        var box = new StrongBox();
        LoopOptimizer.UnrollBy4(16, new CountAction(box));
        Assert.Equal(16, box.Value);
    }

    [Fact]
    public void UnrollBy8_InvokesActionForAllIndices()
    {
        var box = new StrongBox();
        LoopOptimizer.UnrollBy8(17, new CountAction(box));
        Assert.Equal(17, box.Value);
    }

    [Fact]
    public void UnrollBy8_PowerOfEight_AllIndicesHit_NoTail()
    {
        var box = new StrongBox();
        LoopOptimizer.UnrollBy8(32, new CountAction(box));
        Assert.Equal(32, box.Value);
    }

    [Fact]
    public void Fuse_InvokesActionForAllIndices()
    {
        // Fuse with the generic struct callback — the callback's body
        // is a "fused composition of multiple operations" by contract.
        // Functionally identical to a plain UnrollBy4 from the test's
        // point of view; tests the count semantics.
        var box = new StrongBox();
        LoopOptimizer.Fuse(5, new CountAction(box));
        Assert.Equal(5, box.Value);
    }

    /// <summary>Struct that records the bounds of every tile passed to
    /// it. Verifies tile dispatch correctness for Tile2D /
    /// ParallelTile2D.</summary>
    private readonly struct RecordTilesAction : LoopOptimizer.ITile2DAction
    {
        private readonly ConcurrentBag<(int, int, int, int)> _tiles;
        public RecordTilesAction(ConcurrentBag<(int, int, int, int)> tiles) { _tiles = tiles; }
        public void Invoke(int i0, int i1, int j0, int j1) => _tiles.Add((i0, i1, j0, j1));
    }

    [Fact]
    public void Tile2D_VisitsAllTiles()
    {
        const int rows = 10, cols = 9, tileSize = 4;
        var tiles = new ConcurrentBag<(int, int, int, int)>();

        LoopOptimizer.Tile2D(rows, cols, tileSize, new RecordTilesAction(tiles));

        int expectedTilesI = (rows + tileSize - 1) / tileSize;
        int expectedTilesJ = (cols + tileSize - 1) / tileSize;
        Assert.Equal(expectedTilesI * expectedTilesJ, tiles.Count);
        // Every tile range must be in-bounds.
        foreach (var (i0, i1, j0, j1) in tiles)
        {
            Assert.InRange(i0, 0, rows - 1);
            Assert.InRange(i1, 1, rows);
            Assert.InRange(j0, 0, cols - 1);
            Assert.InRange(j1, 1, cols);
        }
    }

    [Fact]
    public void Tile2D_TileSizeZero_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LoopOptimizer.Tile2D(10, 10, 0, new RecordTilesAction(new ConcurrentBag<(int, int, int, int)>())));
    }

    private readonly struct Record3DTilesAction : LoopOptimizer.ITile3DAction
    {
        private readonly ConcurrentBag<(int, int, int, int, int, int)> _tiles;
        public Record3DTilesAction(ConcurrentBag<(int, int, int, int, int, int)> tiles) { _tiles = tiles; }
        public void Invoke(int i0, int i1, int j0, int j1, int k0, int k1) =>
            _tiles.Add((i0, i1, j0, j1, k0, k1));
    }

    [Fact]
    public void Tile3D_VisitsEveryProductTile()
    {
        // 6×4×3 region with 2-cube tiles → 3×2×2 = 12 tiles. Tile3D
        // is the matmul-blocked iteration shape that
        // MatrixMultiplyHelper.MultiplyBlocked will consume after the
        // wiring follow-up.
        const int dim1 = 6, dim2 = 4, dim3 = 3;
        const int t = 2;
        var tiles = new ConcurrentBag<(int, int, int, int, int, int)>();

        LoopOptimizer.Tile3D(dim1, dim2, dim3, t, t, t, new Record3DTilesAction(tiles));

        int expected = ((dim1 + t - 1) / t) * ((dim2 + t - 1) / t) * ((dim3 + t - 1) / t);
        Assert.Equal(expected, tiles.Count);
    }

    private readonly struct StripAction : LoopOptimizer.IStripAction
    {
        private readonly StrongBox _coveredBox;
        public StripAction(StrongBox box) { _coveredBox = box; }
        public void Invoke(int start, int end) =>
            Interlocked.Add(ref _coveredBox.Value, end - start);
    }

    [Fact]
    public void StripMine_CoversFullRange()
    {
        var box = new StrongBox();
        LoopOptimizer.StripMine(10, 4, new StripAction(box));
        Assert.Equal(10, box.Value);
    }

    [Fact]
    public void StripMine_StripSizeZero_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LoopOptimizer.StripMine(10, 0, new StripAction(new StrongBox())));
    }

    private readonly struct CountInterchange : LoopOptimizer.IInterchangeAction
    {
        private readonly StrongBox _box;
        public CountInterchange(StrongBox box) { _box = box; }
        public void Invoke(int i, int j) => Interlocked.Increment(ref _box.Value);
    }

    [Fact]
    public void OptimalOrder2D_RowMajorAndColumnMajorVisitAll()
    {
        const int rows = 3, cols = 4;

        var rmBox = new StrongBox();
        LoopOptimizer.OptimalOrder2D(rows, cols, rowMajorAccess: true, new CountInterchange(rmBox));
        Assert.Equal(rows * cols, rmBox.Value);

        var cmBox = new StrongBox();
        LoopOptimizer.OptimalOrder2D(rows, cols, rowMajorAccess: false, new CountInterchange(cmBox));
        Assert.Equal(rows * cols, cmBox.Value);
    }

    [Fact]
    public void OptimalOrder2D_RowMajor_IteratesRowsContiguously()
    {
        // The interchange contract: rowMajorAccess=true visits (i, j)
        // in row-major order. Confirm by recording the (i, j) pairs
        // and verifying they're sorted by (i, j) lexicographically.
        const int rows = 3, cols = 4;
        var seen = new System.Collections.Generic.List<(int, int)>();
        LoopOptimizer.OptimalOrder2D(rows, cols, rowMajorAccess: true,
            new RecordInterchange(seen));

        Assert.Equal(rows * cols, seen.Count);
        for (int k = 1; k < seen.Count; k++)
        {
            // Lexicographic order: (i, j) ≤ (i', j') iff i < i' || (i == i' && j < j')
            var (i0, j0) = seen[k - 1];
            var (i1, j1) = seen[k];
            Assert.True(i0 < i1 || (i0 == i1 && j0 < j1),
                $"Row-major order violated at index {k}: ({i0},{j0}) → ({i1},{j1})");
        }
    }

    private readonly struct RecordInterchange : LoopOptimizer.IInterchangeAction
    {
        private readonly System.Collections.Generic.List<(int, int)> _seen;
        public RecordInterchange(System.Collections.Generic.List<(int, int)> seen) { _seen = seen; }
        public void Invoke(int i, int j) => _seen.Add((i, j));
    }

    [Fact]
    public void ParallelTile2D_VisitsAllTiles()
    {
        const int rows = 9, cols = 9, tileSize = 4;
        var tiles = new ConcurrentBag<(int, int, int, int)>();

        LoopOptimizer.ParallelTile2D(rows, cols, tileSize, new RecordTilesAction(tiles));

        int expectedTilesI = (rows + tileSize - 1) / tileSize;
        int expectedTilesJ = (cols + tileSize - 1) / tileSize;
        Assert.Equal(expectedTilesI * expectedTilesJ, tiles.Count);
    }

    [Fact]
    public void DetermineOptimalTileSize_ReturnsAtLeastOne_AndAtMostDimension()
    {
        int tileSize = LoopOptimizer.DetermineOptimalTileSize<float>(dimension: 128);
        Assert.InRange(tileSize, 1, 128);
    }

    [Fact]
    public void DetermineOptimalTileSize_DelegatesToCacheOptimizer()
    {
        // The single-source-of-truth contract: DetermineOptimalTileSize<T>
        // must return the same first component as
        // CacheOptimizer.ComputeOptimalTiling<T>(d, d, d). If they
        // diverge, we've reintroduced the duplicated cache math the
        // refactor was supposed to eliminate.
        const int d = 256;
        int loopSide = LoopOptimizer.DetermineOptimalTileSize<float>(d);
        var (cacheM, _, _) = CacheOptimizer.ComputeOptimalTiling<float>(d, d, d);
        Assert.Equal(cacheM, loopSide);
    }

    [Fact]
    public void DetermineOptimalTileSize_GenericT_DiffersByElementSize()
    {
        // double's larger element size should yield a smaller tile
        // — same as the parallel test in CacheOptimizerTests, but
        // routed through the LoopOptimizer wrapper. Confirms the
        // delegation didn't drop the generic-T information.
        int floatTile = LoopOptimizer.DetermineOptimalTileSize<float>(1024);
        int doubleTile = LoopOptimizer.DetermineOptimalTileSize<double>(1024);
        Assert.True(doubleTile <= floatTile);
    }
}
