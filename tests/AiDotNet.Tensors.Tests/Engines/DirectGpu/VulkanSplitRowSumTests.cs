using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1157 on Vulkan: a full reduction, or a few long rows, is split into chunks that a whole workgroup each sums, and
/// the partials are reduced again. Values are small integers, so every float partial sum is exact (below 2^24) and
/// the sums must equal the reference exactly -- a dropped or double-counted element shows. Repeated runs are
/// bit-identical. Mirrors <see cref="SplitRowSumTests"/> (CUDA).
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanSplitRowSumTests
{
    // The backend is a process-wide singleton other test classes share, so nothing here disposes it.
    private static VulkanBackend BackendOrSkip()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.IsGlslCompilerAvailable) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device with the runtime GLSL compiler.");
        return backend!;
    }

    private static float[] Values(int seed, int count)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = rng.Next(-7, 9);
        return a;
    }

    private static void AssertExact(double expected, float got, string what)
        => Assert.True(expected == got, $"{what}: expected {expected}, got {got}");

    [SkippableTheory]
    [InlineData(1, 65_536)]       // exactly the split threshold
    [InlineData(1, 65_535)]       // one short of it: the unsplit kernel
    [InlineData(1, 1_000_003)]    // not a multiple of any chunk length
    [InlineData(3, 200_001)]      // a few long rows
    [InlineData(5, 65_537)]       // the last chunk of each row holds one element
    [InlineData(300, 70_000)]     // more rows than chunks per row
    public void RowSums_MatchTheDoubleReference(int rows, int cols)
    {
        var backend = BackendOrSkip();
        float[] x = Values(rows * 31 + cols, rows * cols);
        using var input = backend.AllocateBuffer(x);
        using var output = backend.AllocateBuffer(rows);
        backend.Fill(output, float.NaN, rows);
        long splits = backend.SplitRowSumLaunches;
        backend.SumAxis(input, output, rows, cols);
        bool expectSplit = cols >= VulkanBackend.SplitRowMinReduce;
        Assert.Equal(expectSplit, backend.SplitRowSumLaunches > splits);
        float[] got = backend.DownloadBuffer(output);
        for (int r = 0; r < rows; r++)
        {
            double expected = 0;
            for (int c = 0; c < cols; c++) expected += x[r * cols + c];
            AssertExact(expected, got[r], $"row {r} of {rows}x{cols}");
        }
    }

    [SkippableFact]
    public void EveryElementIsCountedOnce()
    {
        // One nonzero element at a time, at the positions where a chunk or a lane begins and ends.
        var backend = BackendOrSkip();
        const int rows = 2, cols = 70_001;
        using var output = backend.AllocateBuffer(rows);
        int[] positions = { 0, 1, 255, 256, 257, 8_749, 8_750, 8_751, 8_752, 35_000, 69_999, 70_000 };
        foreach (int row in new[] { 0, 1 })
        {
            foreach (int position in positions)
            {
                var x = new float[rows * cols];
                x[row * cols + position] = 3f;
                using var input = backend.AllocateBuffer(x);
                backend.SumAxis(input, output, rows, cols);
                float[] got = backend.DownloadBuffer(output);
                Assert.True(got[row] == 3f && got[1 - row] == 0f, $"element ({row},{position}): sums {got[0]}, {got[1]}");
            }
        }
    }

    [SkippableFact]
    public void TheEnginesFullReduction_TakesTheSplitPath_AndIsDeterministic()
    {
        var backend = BackendOrSkip();
        IEngine engine = new DirectGpuTensorEngine(new DirectGpuEngine(backend));
        const int n = 2_500_000;
        float[] data = Values(9, n);
        var x = new Tensor<float>(data, new[] { n });
        double expected = data.Sum(v => (double)v);
        long splits = backend.SplitRowSumLaunches;
        float first = engine.ReduceSum(x, null).ToArray()[0];
        Assert.True(backend.SplitRowSumLaunches > splits, "the full sum did not take the split-row path");
        AssertExact(expected, first, $"sum of {n}");
        for (int i = 0; i < 3; i++)
            Assert.Equal(BitConverter.ToInt32(BitConverter.GetBytes(first), 0),
                BitConverter.ToInt32(BitConverter.GetBytes(engine.ReduceSum(x, null).ToArray()[0]), 0));
    }

    [SkippableFact]
    public void TheScalarSum_AndLongRowMeans_TakeTheSplitPath()
    {
        var backend = BackendOrSkip();
        const int rows = 3, cols = 131_072;   // a power of two, so each mean is an exact division
        float[] x = Values(21, rows * cols);
        using var input = backend.AllocateBuffer(x);

        long splits = backend.SplitRowSumLaunches;
        float total = backend.Sum(input, x.Length);
        Assert.True(backend.SplitRowSumLaunches > splits, "the scalar sum did not take the split-row path");
        AssertExact(x.Sum(v => (double)v), total, "scalar sum");

        using var means = backend.AllocateBuffer(rows);
        backend.Fill(means, float.NaN, rows);
        splits = backend.SplitRowSumLaunches;
        backend.MeanAxis(input, means, rows, cols);
        Assert.True(backend.SplitRowSumLaunches > splits, "the row means did not take the split-row path");
        float[] got = backend.DownloadBuffer(means);
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += x[r * cols + c];
            AssertExact(sum / cols, got[r], $"mean of row {r}");
        }

        // A short buffer keeps the single-dispatch kernels.
        splits = backend.SplitRowSumLaunches;
        AssertExact(x.Take(1000).Sum(v => (double)v), backend.Sum(input, 1000), "short scalar sum");
        Assert.Equal(splits, backend.SplitRowSumLaunches);
    }

    [SkippableFact]
    public void InsideABatch_TheUnsplitKernelRuns()
    {
        var backend = BackendOrSkip();
        const int cols = 100_000;
        float[] x = Values(4, cols);
        using var input = backend.AllocateBuffer(x);
        using var output = backend.AllocateBuffer(1);
        long splits = backend.SplitRowSumLaunches;
        backend.BeginBatch();
        try { backend.SumAxis(input, output, 1, cols); }
        finally { backend.EndBatch(); }
        Assert.Equal(splits, backend.SplitRowSumLaunches);
        AssertExact(x.Sum(v => (double)v), backend.DownloadBuffer(output)[0], "batched sum");
    }
}
