using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CudaWeightNormPrecisionTests
{
    // A subnormal direction survives the precise forward; the backward must see the same value. With fast math
    // the stats kernel flushes 1e-39 to zero and the gain gradient collapses from ~1 to 0.
    [SkippableFact]
    public void Cuda_backward_keeps_subnormal_direction_the_precise_forward_preserves()
    {
        using var backend = new CudaBackend();
        CudaTestDevice.Require(backend);
        const float direction = 1e-39f, epsilon = 1e-24f, upstream = 1e27f;
        using var v = backend.AllocateBuffer(new[] { direction });
        using var g = backend.AllocateBuffer(new[] { 1f });
        using var h = backend.AllocateBuffer(new[] { upstream });
        using var output = backend.AllocateBuffer(new float[1]);
        using var stats = backend.AllocateBuffer(new float[2]);
        using var dg = backend.AllocateBuffer(new float[1]);

        backend.WeightNorm(v, g, output, 1, 1, epsilon);
        Assert.NotEqual(0f, backend.DownloadBuffer(output)[0]);

        backend.WeightNormStats(h, v, stats, 1, 1, epsilon);
        backend.WeightNormGainGradient(stats, dg, 1, accumulate: false);
        // dg = (h * v) / sqrt(v^2 + eps) = 1e-12 / 1e-12.
        Assert.InRange(backend.DownloadBuffer(dg)[0], 0.999f, 1.001f);
    }
}
