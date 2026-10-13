#if NETCOREAPP
// GpuCpuCorrectnessFixture (GpuCpuCorrectnessTests.cs) is compiled only off .NET Framework.
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Storage adoption (set_) on the GPU engine leaves the source's shared device binding intact.</summary>
[Collection("DirectGpuSerial")]
public sealed class AdoptFromGpuBindingTests : IClassFixture<GpuCpuCorrectnessFixture>
{
    private readonly GpuCpuCorrectnessFixture _fixture;

    public AdoptFromGpuBindingTests(GpuCpuCorrectnessFixture fixture) => _fixture = fixture;

    private void SkipWithoutGpu() => Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available.");

    [SkippableFact]
    public void SetInPlace_KeepsTheSourcesDeviceBuffer()
    {
        SkipWithoutGpu();
        var gpu = _fixture.Gpu;
        // A device-only result covers its whole vector, so its device buffer lives on the vector and is shared with
        // adopters.
        var x = new Tensor<float>(new[] { 8, 16 });
        var w = new Tensor<float>(new[] { 16, 8 });
        for (int i = 0; i < x.Length; i++) x[i] = (i % 7 - 3) * 0.125f;
        for (int i = 0; i < w.Length; i++) w[i] = (i % 5 - 2) * 0.25f;
        var source = gpu.FusedLinear(x, w, null, FusedActivationType.None);
        Assert.True(source._gpuBuffer is not null, "precondition: FusedLinear leaves its result device-resident");
        var buffer = source._gpuBuffer;
        var target = new Tensor<float>(new[] { 3 });
        gpu.TensorSetInPlace(target, source);
        // Adoption used to clear the binding through the shared vector: the source lost its buffer.
        Assert.Same(buffer, source._gpuBuffer);
        var expected = source.ToArray();
        Assert.Equal(expected, target.ToArray());
    }
}
#endif
