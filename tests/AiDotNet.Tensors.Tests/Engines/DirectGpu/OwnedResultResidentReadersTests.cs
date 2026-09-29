using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A GPU op result owns its device buffer (tensor._gpuBuffer) and is not in the activation cache. Readers that looked
/// only in the cache treated it as host-only: the dtype cast bridge took the host Tensor.Cast loop, and gradient routing
/// declined once the result's host backing had been materialized.
/// </summary>
[Collection("DirectGpuSerial")]
public class OwnedResultResidentReadersTests
{
    // FusedLinear's result is a DeferTensorResult: owned by the tensor, not entered in the activation cache.
    private static Tensor<float> OwnedResult(DirectGpuTensorEngine gpu)
    {
        var x = new Tensor<float>(new[] { 16, 32 });
        var w = new Tensor<float>(new[] { 32, 16 });
        for (int i = 0; i < x.Length; i++) x[i] = (i % 7 - 3) * 0.125f;
        for (int i = 0; i < w.Length; i++) w[i] = (i % 5 - 2) * 0.25f;
        var r = gpu.FusedLinear(x, w, null, FusedActivationType.None);
        Assert.NotNull(r._gpuBuffer);
        return r;
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_dtype_cast_of_an_owned_result_stays_on_the_device(bool hostReadFirst)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend.");
        var r = OwnedResult(gpu);
        int n = r.Length;
        var expected = OwnedResult(gpu).ToArray();
        if (hostReadFirst) r.ToArray();   // materializes the host backing; the owned buffer stays current

        var half = gpu.CastResidentDtype<float, Half>(r);

        Assert.True(half.IsGpuResident || half.HasPendingGpuData, "the cast of an owned result ran on the host");
        var values = half.ToArray();
        for (int i = 0; i < n; i++) Assert.Equal(expected[i], (float)values[i], 2);
    }

    [SkippableFact]
    public void Gradient_routing_uses_an_owned_result_after_its_host_backing_materialized()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend.");
        var g = OwnedResult(gpu);
        int n = g.Length;
        var expected = g.ToArray();
        var dest = new Tensor<float>(new[] { n });
        IGpuBuffer? stable = null;

        Assert.True(gpu.TryRouteGradResident(g, dest, ref stable), "routing declined an owned, current gradient");
        var routed = gpu.GetBackend()!.DownloadBuffer(stable!);
        for (int i = 0; i < n; i++) Assert.Equal(expected[i], routed[i]);
        stable!.Dispose();
    }
}
