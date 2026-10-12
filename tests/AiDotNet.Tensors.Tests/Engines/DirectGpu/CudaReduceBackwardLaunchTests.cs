using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// CUDA ReduceSumBackward / ReduceMeanBackward write every element of the [outer, reduce] gradient. They were
/// launched with one thread per row, so everything past the first outerSize elements (rounded up to a block) kept
/// whatever the buffer held.
/// </summary>
public class CudaReduceBackwardLaunchTests
{
    [SkippableTheory]
    [InlineData(300, 5)]
    [InlineData(2, 600)]
    [InlineData(1, 1)]
    public void ReduceSumAndMeanBackward_WriteEveryElement(int outer, int reduce)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        int n = outer * reduce;
        float[] grad = Enumerable.Range(0, outer).Select(i => 0.25f * (i % 7) - 0.5f).ToArray();
        float[] fill = Enumerable.Repeat(123.25f, n).ToArray();   // anything left unwritten shows up as 123.25
        using var g = backend.AllocateBuffer(grad);

        using (var sum = backend.AllocateBuffer(fill))
        {
            backend.ReduceSumBackward(g, sum, outer, reduce);
            var got = backend.DownloadBuffer(sum);
            for (int i = 0; i < n; i++) Assert.Equal(grad[i / reduce], got[i]);
        }
        using (var mean = backend.AllocateBuffer(fill))
        {
            backend.ReduceMeanBackward(g, mean, outer, reduce);
            var got = backend.DownloadBuffer(mean);
            // The device divide may round differently in the last place (not IEEE-exact division).
            for (int i = 0; i < n; i++)
            {
                float expected = grad[i / reduce] / reduce;
                Assert.True(System.Math.Abs(expected - got[i]) <= 1e-6f * System.Math.Abs(expected) + 1e-12f,
                    $"mean[{i}]: expected {expected}, got {got[i]}");
            }
        }
    }

    [SkippableFact]
    public void ReduceSumBackward_RejectsNegativeSizesAndShortBuffers()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        using var g = backend.AllocateBuffer(new float[4]);
        using var o = backend.AllocateBuffer(new float[8]);
        // Two negatives multiply to a positive count; each is rejected on its own.
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.ReduceSumBackward(g, o, -2, -4));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.ReduceMeanBackward(g, o, 2, -4));
        Assert.Throws<ArgumentException>(() => backend.ReduceSumBackward(g, o, 5, 1));    // gradOutput holds 4 < 5
        Assert.Throws<ArgumentException>(() => backend.ReduceSumBackward(g, o, 3, 3));    // gradInput holds 8 < 9
        backend.ReduceSumBackward(g, o, 0, 7);                                            // empty: no launch
    }
}
