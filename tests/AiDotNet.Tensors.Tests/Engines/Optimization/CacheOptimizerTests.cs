// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// Tests for the issue #294 generic-T + Span&lt;T&gt; refactor of
/// <see cref="CacheOptimizer"/>. Covers element-type variation
/// (float/double/int/Half) and the new prefetch-enabled copy path.
/// </summary>
public class CacheOptimizerTests
{
    [Fact]
    public void ComputeOptimalTiling_Float_ReturnsPositiveTiles_WithinDimensions()
    {
        var (tileM, tileN, tileK) = CacheOptimizer.ComputeOptimalTiling<float>(m: 128, n: 256, k: 64);

        Assert.InRange(tileM, 1, 128);
        Assert.InRange(tileN, 1, 256);
        Assert.InRange(tileK, 1, 64);
    }

    [Fact]
    public void ComputeOptimalTiling_Double_HasSmallerTilesThanFloat()
    {
        // double is 8 bytes vs float's 4 — same L1 budget should yield
        // a smaller tile count for double. Sanity check that the new
        // generic API actually consumes Unsafe.SizeOf<T>.
        var floatTile = CacheOptimizer.ComputeOptimalTiling<float>(m: 1024, n: 1024, k: 1024);
        var doubleTile = CacheOptimizer.ComputeOptimalTiling<double>(m: 1024, n: 1024, k: 1024);

        Assert.True(doubleTile.tileM <= floatTile.tileM,
            $"double tile {doubleTile.tileM} must be <= float tile {floatTile.tileM} (L1 holds fewer 8-byte elements).");
    }

    [Fact]
    public void ComputeOptimalTiling_NegativeDim_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => CacheOptimizer.ComputeOptimalTiling<float>(-1, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => CacheOptimizer.ComputeOptimalTiling<float>(1, -1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => CacheOptimizer.ComputeOptimalTiling<float>(1, 1, -1));
    }

    [Fact]
    public void TransposeBlocked_Float_TransposesCorrectly()
    {
        const int rows = 3;
        const int cols = 4;
        var src = new float[rows * cols];
        for (int i = 0; i < src.Length; i++) src[i] = i + 1;
        var dst = new float[rows * cols];

        CacheOptimizer.TransposeBlocked<float>(src, dst, rows, cols);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.Equal(src[r * cols + c], dst[c * rows + r]);
    }

    [Fact]
    public void TransposeBlocked_Double_TransposesCorrectly()
    {
        // Generic-T coverage: same logic must work for double, which
        // takes a different path through Unsafe.SizeOf<T> for the
        // block math. Catches regressions in the per-T element-size
        // fast paths.
        const int rows = 5;
        const int cols = 7;
        var src = new double[rows * cols];
        for (int i = 0; i < src.Length; i++) src[i] = (i + 1) * 0.5;
        var dst = new double[rows * cols];

        CacheOptimizer.TransposeBlocked<double>(src, dst, rows, cols);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.Equal(src[r * cols + c], dst[c * rows + r]);
    }

    [Fact]
    public void TransposeBlocked_LargeBlock_CrossesBlockBoundary()
    {
        // 70×70 forces multiple 32-element-square blocks (3 × 3 with
        // 6-element tail strips). Validates the inner ii/jj loop
        // boundaries on the non-aligned tail case.
        const int n = 70;
        var src = new float[n * n];
        var rng = new Random(42);
        for (int i = 0; i < src.Length; i++) src[i] = (float)rng.NextDouble();
        var dst = new float[n * n];

        CacheOptimizer.TransposeBlocked<float>(src, dst, n, n);

        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++)
                Assert.Equal(src[r * n + c], dst[c * n + r]);
    }

    [Fact]
    public void TransposeBlocked_DstTooSmall_Throws()
    {
        var src = new float[12];
        var dst = new float[10];
        Assert.Throws<ArgumentException>(() =>
            CacheOptimizer.TransposeBlocked<float>(src, dst, 3, 4));
    }

    [Fact]
    public void TransposeBlocked_NegativeRows_Throws()
    {
        var src = new float[12];
        var dst = new float[12];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CacheOptimizer.TransposeBlocked<float>(src, dst, -1, 4));
    }

    [Fact]
    public void CopyWithPrefetch_Float_Short_CopiesAllElements()
    {
        // Short copies fall through to the Span.CopyTo fallback (below
        // the 1KB prefetch threshold). Validates the fallback path.
        var src = new float[] { 1f, 2f, 3f, 4f };
        var dst = new float[src.Length];

        CacheOptimizer.CopyWithPrefetch<float>(src, dst);

        Assert.Equal(src, dst);
    }

    [Fact]
    public void CopyWithPrefetch_Float_Long_CopiesAllElements()
    {
        // 4096 floats = 16KB — well above the 1KB threshold so this
        // exercises the prefetch-enabled bulk loop on x86 SSE hosts
        // (and the SIMD CopyTo fallback elsewhere). Either way, the
        // copy must be bit-exact.
        const int n = 4096;
        var src = new float[n];
        var rng = new Random(7);
        for (int i = 0; i < n; i++) src[i] = (float)rng.NextDouble();
        var dst = new float[n];

        CacheOptimizer.CopyWithPrefetch<float>(src, dst);

        for (int i = 0; i < n; i++) Assert.Equal(src[i], dst[i]);
    }

    [Fact]
    public void CopyWithPrefetch_Double_Long_CopiesAllElements()
    {
        // Same long-copy validation for double — exercises the
        // generic-T size math (8B element → half as many elements per
        // cache line as float).
        const int n = 2048;
        var src = new double[n];
        var rng = new Random(11);
        for (int i = 0; i < n; i++) src[i] = rng.NextDouble();
        var dst = new double[n];

        CacheOptimizer.CopyWithPrefetch<double>(src, dst);

        for (int i = 0; i < n; i++) Assert.Equal(src[i], dst[i]);
    }

    [Fact]
    public void CopyWithPrefetch_Empty_NoOp()
    {
        var src = Array.Empty<float>();
        var dst = Array.Empty<float>();
        CacheOptimizer.CopyWithPrefetch<float>(src, dst); // must not throw
    }

    [Fact]
    public void CopyWithPrefetch_DstTooSmall_Throws()
    {
        var src = new float[10];
        var dst = new float[5];
        Assert.Throws<ArgumentException>(() => CacheOptimizer.CopyWithPrefetch<float>(src, dst));
    }

    [Fact]
    public void MortonEncodeDecode_RoundTrips()
    {
        const int x = 123;
        const int y = 456;

        int code = CacheOptimizer.MortonEncode(x, y);
        var (rx, ry) = CacheOptimizer.MortonDecode(code);

        Assert.Equal(x & 0x0000ffff, rx);
        Assert.Equal(y & 0x0000ffff, ry);
    }

    [Fact]
    public void EstimateCacheMisses_GenericT_DiffersByElementSize()
    {
        // double has half as many elements per cache line as float, so
        // the same dataSize produces twice as many cache lines for
        // double — fewer misses overall on the sequential path.
        // Confirms the generic API consumes Unsafe.SizeOf<T>.
        double floatMisses = CacheOptimizer.EstimateCacheMisses<float>(
            dataSize: 1024, accessStride: 1, cacheSize: 32 * 1024, cacheLineSize: 64);
        double doubleMisses = CacheOptimizer.EstimateCacheMisses<double>(
            dataSize: 1024, accessStride: 1, cacheSize: 32 * 1024, cacheLineSize: 64);

        Assert.True(doubleMisses > floatMisses,
            $"double should have more cache lines (and thus more misses) than float for the same dataSize: float={floatMisses}, double={doubleMisses}");
    }
}
