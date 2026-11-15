using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The shapes that made two CUDA kernels serial: max over long rows (a thread per row read each 49k-wide row alone) and
/// the sorted scatter's run starts for few items into many destinations (a thread per item filled every empty run up to
/// its key). Both must match a CPU reference exactly.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaLongRowKernelTests
{
    private static (DirectGpuTensorEngine Gpu, CudaBackend Cuda) Cuda()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        if (gpu is null || !gpu.IsGpuAvailable || gpu.GetBackend() is not CudaBackend cuda)
        {
            gpu?.Dispose();
            Skip.If(true, "CUDA backend did not resolve.");
            throw new InvalidOperationException("Unreachable after Skip.If.");
        }
        return (gpu, cuda);
    }

    [SkippableTheory]
    [InlineData(600, 49152)]
    [InlineData(9, 300)]
    [InlineData(7, 256)]
    [InlineData(1000, 17)]   // short rows: the thread-per-row kernel
    public void MaxAxis_MatchesTheCpu_IncludingNaNAndInfinities(int rows, int width)
    {
        var (gpu, cuda) = Cuda();
        using (gpu)
        {
            var rng = new Random(rows * 31 + width);
            var data = new float[rows * width];
            for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 200 - 100);
            // The CPU ReduceMax contract (ReduceMaxSeed): scan from -inf, take a value only when strictly greater, so a
            // NaN never wins wherever it sits, an all-NaN row gives -inf, and of equal maxima the first is kept.
            data[0] = float.NaN;                                          // row 0: a leading NaN is skipped
            if (rows > 1) data[width + width / 2] = float.NaN;            // row 1: so is a later one
            if (rows > 2) for (int i = 0; i < width; i++) data[2 * width + i] = float.NegativeInfinity;
            if (rows > 3) data[3 * width + width - 1] = float.PositiveInfinity;
            if (rows > 4) for (int i = 0; i < width; i++) data[4 * width + i] = float.NaN;   // all NaN: -inf
            if (rows > 5)
            {
                // A -0/+0 tie for the maximum, -0 first: -0 is kept, its bits included.
                for (int i = 0; i < width; i++) data[5 * width + i] = -1f - i;
                data[5 * width + 1] = -0f;
                data[5 * width + width - 1] = 0f;
            }
            if (rows > 6)
            {
                // Subnormals: never flushed to zero under --use_fast_math.
                for (int i = 0; i < width; i++) data[6 * width + i] = -float.Epsilon;
                data[6 * width + width - 1] = float.Epsilon;
            }

            var expected = new CpuEngine()
                .ReduceMax(new Tensor<float>(data, new[] { rows, width }), new[] { 1 }, keepDims: false)
                .ToArray();

            using var input = cuda.AllocateBuffer(data);
            using var output = cuda.AllocateBuffer(rows);
            cuda.MaxAxis(input, output, rows, width);
            var actual = cuda.DownloadBuffer(output);
            for (int r = 0; r < rows; r++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected[r]) == TestHelpers.MathCompat.SingleToInt32Bits(actual[r]),
                    $"row {r}: gpu {actual[r]} cpu {expected[r]}");
        }
    }

    [SkippableTheory]
    [InlineData(600, 49152, 8)]     // few items, many destinations: long empty runs
    [InlineData(5000, 64, 3)]       // many items per destination
    [InlineData(1, 100000, 2)]      // one item
    public void SortedScatterAdd_ManyDestinations_MatchesTheCpu(int count, int destinations, int features)
    {
        var (gpu, cuda) = Cuda();
        using (gpu)
        {
            var rng = new Random(count + destinations);
            var indices = Enumerable.Range(0, count).Select(i => i % 7 == 3 ? -1 : rng.Next(destinations + 2)).ToArray();
            var source = Enumerable.Range(0, count * features).Select(i => (float)Math.Sin(i * 0.37)).ToArray();
            var expected = new float[destinations * features];
            for (int i = 0; i < count; i++)
            {
                int d = indices[i];
                if (d < 0 || d >= destinations) continue;
                for (int f = 0; f < features; f++) expected[d * features + f] += source[i * features + f];
            }

            using var src = cuda.AllocateBuffer(source);
            using var idx = cuda.AllocateIntBuffer(indices);
            using var output = cuda.AllocateBuffer(destinations * features);
            cuda.SortedScatterSwitch = true;
            cuda.SortedScatterIgnoresCostModel = true;   // the one-item case is one the cost model never sends here
            bool ran = cuda.TrySortedScatterAdd(src, idx, output, count, destinations, features, accumulate: false);
            Assert.True(ran, "the sorted scatter did not run (kernels missing?)");
            var actual = cuda.DownloadBuffer(output);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                    $"output[{i / features}, {i % features}]: gpu {actual[i]} cpu {expected[i]}");
        }
    }
}
