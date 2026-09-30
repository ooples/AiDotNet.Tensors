using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A whole-vector view shares its source's device binding (device-owned storage). Disposing the view used to clear that
/// shared binding, so the source - a GPU result still in use - lost its device copy and its next GPU use downloaded the
/// data only to upload it again (AbcScanGpuParityTests: every view of a deferred result took a host round trip).
/// </summary>
[Collection("DirectGpuSerial")]
public class DisposedViewSharedBindingTests
{
    [SkippableFact]
    public void Disposing_a_view_keeps_the_sources_device_copy()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend.");
        var x = new Tensor<float>(new[] { 8, 16 });
        var w = new Tensor<float>(new[] { 16, 8 });
        for (int i = 0; i < x.Length; i++) x[i] = (i % 7 - 3) * 0.125f;
        for (int i = 0; i < w.Length; i++) w[i] = (i % 5 - 2) * 0.25f;
        var result = gpu.FusedLinear(x, w, null, FusedActivationType.None);   // device-only result
        var buffer = result._gpuBuffer;
        Assert.NotNull(buffer);

        var view = result.Reshape(result.Length);
        view.Dispose();

        Assert.Same(buffer, result._gpuBuffer);
        bool saved = GpuLaunchProbe.CaptureReadbackSites;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            GpuLaunchProbe.Reset();
            var doubled = gpu.TensorMultiplyScalar(result, 2f);
            Assert.True(GpuLaunchProbe.Readbacks == 0,
                $"using the source after disposing its view read back: {string.Join("; ", GpuLaunchProbe.ReadbackSites)}");
            var expected = result.ToArray();
            var actual = doubled.ToArray();
            for (int i = 0; i < expected.Length; i++) Assert.Equal(2f * expected[i], actual[i], 4);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = saved;
        }
    }
}
