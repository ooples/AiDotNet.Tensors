#if NET6_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
#endif
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1157: SumAxis on OpenCL, HIP, Metal and WebGPU gave each row to one thread, so a full reduction ran on a single
/// thread. Long rows now take a work-group each, and few long rows are split across work-groups into partials that are
/// reduced again (the CUDA #1150 split). Values are small integers, so every float partial sum is exact and the sums
/// must equal the reference exactly; repeated runs are bit-identical. Each backend runs where it is available.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class SumAxisRowSplitBackendTests
{
    private static float[] Values(int seed, int n)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)rng.Next(-7, 9)).ToArray();
    }

    private static float[] SumAxis(IDirectGpuBackend backend, float[] data, int outer, int reduce)
    {
        using var input = backend.AllocateBuffer(data);
        using var output = backend.AllocateBuffer(outer);
        backend.SumAxis(input, output, outer, reduce);
        return backend.DownloadBuffer(output);
    }

    private static readonly (int Outer, int Reduce)[] Shapes =
    {
        (1, 1_000_003),   // full reduction: split, chunk length not dividing the row
        (1, 65_536),      // exactly the split threshold
        (3, 200_001),     // few long rows: split
        (257, 300),       // a work-group per row
        (4096, 32),       // the row threshold
        (5000, 3),        // short rows: one thread per row
        (1, 1),
    };

    private static void CheckAll(IDirectGpuBackend backend)
    {
        foreach (var (outer, reduce) in Shapes)
        {
            var data = Values(outer * 31 + reduce, outer * reduce);
            var got = SumAxis(backend, data, outer, reduce);
            for (int r = 0; r < outer; r++)
            {
                double expected = 0;
                for (int c = 0; c < reduce; c++) expected += data[(long)r * reduce + c];
                Assert.True(expected == got[r], $"{backend.BackendName} row {r} of [{outer}, {reduce}]: expected {expected}, got {got[r]}");
            }
        }
        var rng = new Random(9);
        var noisy = Enumerable.Range(0, 2_500_000).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float first = SumAxis(backend, noisy, 1, noisy.Length)[0];
        for (int i = 0; i < 3; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(first), BitConverter.SingleToInt32Bits(SumAxis(backend, noisy, 1, noisy.Length)[0]));
    }

    [SkippableFact]
    public void OpenCl_SumAxis_ExactAndDeterministic_AndSplitsFewLongRows()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        long before = backend.SumAxisSplitLaunches;
        CheckAll(backend);
        Assert.True(backend.SumAxisSplitLaunches > before, "no few-long-rows sum took the split path");
        // A wide-but-short shape must not split.
        long mid = backend.SumAxisSplitLaunches;
        SumAxis(backend, Values(1, 257 * 300), 257, 300);
        Assert.Equal(mid, backend.SumAxisSplitLaunches);
    }

    [SkippableFact]
    public void Hip_SumAxis_ExactAndDeterministic()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Metal_SumAxis_ExactAndDeterministic()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        CheckAll(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_SumAxis_ExactAndDeterministic()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();   // the real backend, executing on Dawn; shared, so not disposed
        CheckAll(backend);
    }
#endif
}
#endif
