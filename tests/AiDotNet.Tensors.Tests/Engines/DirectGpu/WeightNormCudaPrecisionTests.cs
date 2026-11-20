#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class WeightNormCudaPrecisionTests
{
    [SkippableFact]
    public void RaggedVectorRowsAndLargeStreamingRowsMatchTheNormalization()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA unavailable.");
        var native = Assert.IsAssignableFrom<IGpuBatchExecution>(backend);
        foreach (int columns in new[] { 1, 31, 96, 128, 192, 256, 513, 2816, 4096, 4097, 16387 })
        {
            const int rows = 3;
            var direction = new float[rows * columns];
            var gain = new[] { .75f, -1.125f, 0f };
            for (int i = 0; i < direction.Length; i++) direction[i] = (float)(Math.Sin(i * .37 + .11) * .03125);
            using var v = backend.AllocateBuffer(direction);
            using var g = backend.AllocateBuffer(gain);
            using var output = backend.AllocateBuffer(direction.Length);
            native.WeightNorm(v, g, output, rows, columns, 1e-24f);
            var actual = backend.DownloadBuffer(output);
            for (int row = 0; row < rows; row++)
            {
                double square = 0;
                for (int j = 0; j < columns; j++) { double x = direction[row * columns + j]; square += x * x; }
                double scale = gain[row] / Math.Sqrt(square + 1e-24f);
                for (int j = 0; j < columns; j++)
                {
                    int i = row * columns + j;
                    double expected = direction[i] * scale;
                    Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected) <= 7e-7 * Math.Abs(expected) + 1e-12,
                        $"columns={columns}, index={i}: {actual[i]:G9} != {expected:G17}");
                }
            }
        }
    }

    [SkippableFact]
    public void SubnormalDirectionsRemainNonzeroWhenEpsilonKeepsTheNormFinite()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA unavailable.");
        var native = Assert.IsAssignableFrom<IGpuBatchExecution>(backend);
        var direction = new[] { 1e-39f, -2e-39f, 3e-39f, -4e-39f, 5e-39f, -6e-39f, 7e-39f };
        using var v = backend.AllocateBuffer(direction);
        using var g = backend.AllocateBuffer(new[] { -.75f });
        using var output = backend.AllocateBuffer(direction.Length);
        native.WeightNorm(v, g, output, 1, direction.Length, 1e-24f);
        var actual = backend.DownloadBuffer(output);
        for (int i = 0; i < direction.Length; i++)
        {
            double expected = direction[i] * (-.75 / Math.Sqrt(1e-24f));
            Assert.NotEqual(0f, actual[i]);
            Assert.True(Math.Abs(actual[i] - expected) <= 7e-7 * Math.Abs(expected), $"index={i}: {actual[i]:G9} != {expected:G17}");
        }
    }

    [SkippableFact]
    public void ZeroDirectionAndZeroEpsilonKeepTheComposedNaNConvention()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA unavailable.");
        var native = Assert.IsAssignableFrom<IGpuBatchExecution>(backend);
        using var direction = backend.AllocateBuffer(new float[7]);
        using var gain = backend.AllocateBuffer(new[] { 1f });
        using var output = backend.AllocateBuffer(7);
        native.WeightNorm(direction, gain, output, 1, 7, 0f);
        Assert.All(backend.DownloadBuffer(output), value => Assert.True(float.IsNaN(value)));
    }
}
#endif
