using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1157: OpenCL SumAxis gave each row to one work-item, so a full reduction ran on a single work-item. Long rows now
/// take a work-group each, and few long rows are split across work-groups into partials that are reduced again (the
/// CUDA #1150 split). Values are small integers, so every float partial sum is exact and the sums must equal the
/// reference exactly; repeated runs are bit-identical.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClSumAxisTests
{
    private static float[] Values(int seed, int n)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)rng.Next(-7, 9)).ToArray();
    }

    private static float[] SumAxis(OpenClBackend backend, float[] data, int outer, int reduce)
    {
        using var input = backend.AllocateBuffer(data);
        using var output = backend.AllocateBuffer(outer);
        backend.SumAxis(input, output, outer, reduce);
        return backend.DownloadBuffer(output);
    }

    private static void AssertRowSums(float[] data, float[] got, int outer, int reduce)
    {
        for (int r = 0; r < outer; r++)
        {
            double expected = 0;
            for (int c = 0; c < reduce; c++) expected += data[(long)r * reduce + c];
            Assert.True(expected == got[r], $"row {r} of [{outer}, {reduce}]: expected {expected}, got {got[r]}");
        }
    }

    [SkippableTheory]
    [InlineData(1, 1_000_003)]   // full reduction: split, chunk length not dividing the row
    [InlineData(1, 65_536)]      // exactly the split threshold
    [InlineData(3, 200_001)]     // few long rows: split
    public void FewLongRows_AreSplit_AndExact(int outer, int reduce)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var data = Values(outer * 31 + reduce, outer * reduce);
        long before = backend.SumAxisSplitLaunches;
        var got = SumAxis(backend, data, outer, reduce);
        AssertRowSums(data, got, outer, reduce);
        Assert.True(backend.SumAxisSplitLaunches > before, "the few-long-rows sum did not take the split path");
    }

    [SkippableTheory]
    [InlineData(257, 300)]       // a work-group per row
    [InlineData(4096, 32)]       // the row threshold
    [InlineData(5000, 3)]        // short rows: one work-item per row
    [InlineData(1, 1)]
    public void OtherShapes_AreExact_AndNotSplit(int outer, int reduce)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var data = Values(outer + reduce, outer * reduce);
        long before = backend.SumAxisSplitLaunches;
        AssertRowSums(data, SumAxis(backend, data, outer, reduce), outer, reduce);
        Assert.Equal(before, backend.SumAxisSplitLaunches);
    }

    [SkippableFact]
    public void SplitSum_IsDeterministic()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var rng = new Random(9);
        var data = Enumerable.Range(0, 2_500_000).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float first = SumAxis(backend, data, 1, data.Length)[0];
        for (int i = 0; i < 3; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(first), BitConverter.SingleToInt32Bits(SumAxis(backend, data, 1, data.Length)[0]));
    }
}
