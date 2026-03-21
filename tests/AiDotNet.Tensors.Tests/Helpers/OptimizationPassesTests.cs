using System;
using AiDotNet.Tensors.Helpers;
using Xunit;
using static AiDotNet.Tensors.Helpers.ComputationGraph;
using static AiDotNet.Tensors.Helpers.OptimizationPasses;

namespace AiDotNet.Tensors.Tests.Helpers;

public class OptimizationPassesTests
{
    [Fact]
    public void CacheInfo_ReturnsPositiveValues()
    {
        Assert.True(CacheInfo.L1DataCacheSize > 0);
        Assert.True(CacheInfo.L2CacheSize > 0);
        Assert.True(CacheInfo.L3CacheSize > 0);
        Assert.True(CacheInfo.L1TileSizeFloat > 0);
        Assert.True(CacheInfo.L2TileSizeFloat > 0);
    }

    [Fact]
    public void ComputeMatMulTiles_FitsInL2()
    {
        var (tileM, tileN, tileK) = ComputeMatMulTiles(1024, 1024, 1024, elementSize: 4);

        // Tiles should be reasonable
        Assert.True(tileM >= 16 && tileM <= 1024);
        Assert.True(tileN >= 8 && tileN <= 1024);
        Assert.True(tileK >= 16 && tileK <= 1024);

        // Total tile memory should fit in L2
        long tileBytes = ((long)tileM * tileK + (long)tileK * tileN + (long)tileM * tileN) * 4;
        Assert.True(tileBytes <= CacheInfo.L2CacheSize,
            $"Tile bytes {tileBytes} exceeds L2 {CacheInfo.L2CacheSize}");
    }

    [Fact]
    public void ComputeMatMulTiles_SmallMatrix_NoTiling()
    {
        var (tileM, tileN, tileK) = ComputeMatMulTiles(16, 16, 16, elementSize: 4);

        // Small matrix should use full dimensions (no need to tile)
        Assert.Equal(16, tileM);
        Assert.True(tileN >= 8); // at least SIMD width
        Assert.Equal(16, tileK);
    }

    [Fact]
    public void ComputeMatMulTiles_Double_SmallerTiles()
    {
        var (tileMf, tileNf, tileKf) = ComputeMatMulTiles(1024, 1024, 1024, elementSize: 4);
        var (tileMd, tileNd, tileKd) = ComputeMatMulTiles(1024, 1024, 1024, elementSize: 8);

        // Double needs larger memory per element, so tiles should be smaller
        Assert.True(tileMd <= tileMf || tileNd <= tileNf,
            "Double tiles should be no larger than float tiles");
    }

    [Fact]
    public void ComputeConv2DTileSize_ReturnsPositive()
    {
        int tile = ComputeConv2DTileSize(
            outputHeight: 64, outputWidth: 64,
            kernelSize: 9, inChannels: 128,
            outChannels: 256, elementSize: 4);

        Assert.True(tile >= 1);
        Assert.True(tile <= 64 * 64);
    }

    [Fact]
    public void ReorderForLocality_LinearChain_PreservesOrder()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int inp = graph.RecordInput([1, 64]);
        int a = graph.RecordOp(OpType.ReLU, [inp], [1, 64]);
        int b = graph.RecordOp(OpType.ReLU, [a], [1, 64]);
        int c = graph.RecordOp(OpType.ReLU, [b], [1, 64]);
        graph.RecordOutput(c);
        graph.EndCapture();

        var order = ReorderForLocality(graph);

        // Linear chain has only one valid order
        Assert.Equal(4, order.Length);
        Assert.Equal(0, order[0]); // input first
    }

    [Fact]
    public void ReorderForLocality_DiamondPattern_GroupsRelatedOps()
    {
        // Diamond: input -> a, input -> b, a+b -> c
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int inp = graph.RecordInput([1, 64]);
        int a = graph.RecordOp(OpType.Conv2D, [inp], [1, 64, 32, 32]);
        int b = graph.RecordOp(OpType.Conv2D, [inp], [1, 64, 32, 32]);
        int c = graph.RecordOp(OpType.Add, [a, b], [1, 64, 32, 32]);
        graph.RecordOutput(c);
        graph.EndCapture();

        var order = ReorderForLocality(graph);

        Assert.Equal(4, order.Length);
        // Input must come first
        Assert.Equal(0, order[0]);
        // c must come after both a and b
        int cIdx = Array.IndexOf(order, 3);
        int aIdx = Array.IndexOf(order, 1);
        int bIdx = Array.IndexOf(order, 2);
        Assert.True(cIdx > aIdx && cIdx > bIdx);
    }

    [Fact]
    public void ReorderForLocality_IndependentBranches_LocalityPreferred()
    {
        // Two independent branches from the same input:
        // input -> a1 -> a2 (branch A)
        // input -> b1 -> b2 (branch B)
        // We want a1,a2 grouped and b1,b2 grouped (not interleaved)
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int inp = graph.RecordInput([1, 64]);
        int a1 = graph.RecordOp(OpType.Conv2D, [inp], [1, 64, 32, 32]);
        int a2 = graph.RecordOp(OpType.ReLU, [a1], [1, 64, 32, 32]);
        int b1 = graph.RecordOp(OpType.Conv2D, [inp], [1, 128, 16, 16]);
        int b2 = graph.RecordOp(OpType.ReLU, [b1], [1, 128, 16, 16]);
        graph.RecordOutput(a2);
        graph.RecordOutput(b2);
        graph.EndCapture();

        var order = ReorderForLocality(graph);

        Assert.Equal(5, order.Length);
        Assert.Equal(0, order[0]); // input first

        // a1 and a2 should be adjacent, b1 and b2 should be adjacent
        int a1Idx = Array.IndexOf(order, 1);
        int a2Idx = Array.IndexOf(order, 2);
        int b1Idx = Array.IndexOf(order, 3);
        int b2Idx = Array.IndexOf(order, 4);

        bool aGrouped = Math.Abs(a1Idx - a2Idx) == 1;
        bool bGrouped = Math.Abs(b1Idx - b2Idx) == 1;

        Assert.True(aGrouped || bGrouped,
            "At least one branch should have its ops adjacent for locality");
    }
}
