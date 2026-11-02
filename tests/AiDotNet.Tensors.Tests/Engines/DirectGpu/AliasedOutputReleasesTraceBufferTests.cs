using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A compiled step writes each node's result by aliasing the plan's output tensor to the result's buffer. The tensor's
/// own device copy from the trace stayed in the activation cache, which lives as long as the tensor's storage: the
/// plan's lifetime. DenseNet-121 training held 1.2 GB of such trace-time results.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class AliasedOutputReleasesTraceBufferTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public AliasedOutputReleasesTraceBufferTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void AliasingAResidentStepOutput_FreesItsTraceTimeBuffer()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.PlacementBackend;
        Skip.If(backend is null, "No placement backend.");

        int[] shape = { 4, 64 };
        var x = new float[256];
        var fresh = new float[256];
        for (int i = 0; i < 256; i++) { x[i] = i; fresh[i] = -2f * i; }

        // The trace: a node realized into dest on the device, its buffer cached under dest's host array.
        var dest = new Tensor<float>(shape);
        ((IEngine)gpu).ReLUInto(dest, new Tensor<float>(x, shape));
        Assert.Equal(255f, dest.ToArray()[255]);
        var destArray = dest.GetBackingArrayForCacheLookupUnsafe();
        Assert.NotNull(destArray);
        Skip.IfNot(HasActivationEntry(gpu, destArray), "This backend keeps no activation entry for the result.");

        var src = Tensor<float>.FromGpuBuffer(backend, backend.AllocateBuffer(fresh), shape);
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
                DirectGpuTensorEngine.CopyResultInto(gpu, src, dest);
        }
        finally
        {
            gpu.ResumeActivationEviction();
        }

        Assert.Same(src._gpuBuffer, dest._gpuBuffer);
        Assert.False(HasActivationEntry(gpu, destArray));
        Assert.Equal(fresh, dest.ToArray());
    }

    private static bool HasActivationEntry(DirectGpuTensorEngine gpu, object key)
    {
        var cache = typeof(DirectGpuTensorEngine).GetField("_activationCache", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(gpu);
        Assert.NotNull(cache);
        var containsKey = cache.GetType().GetMethod("ContainsKey");
        Assert.NotNull(containsKey);
        return containsKey.Invoke(cache, new[] { key }) is true;
    }
}
