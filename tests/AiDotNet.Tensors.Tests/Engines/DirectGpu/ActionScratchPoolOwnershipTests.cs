using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Inside a compiled action, TryResidentZeros (Sign's zero gradient) and the device ReduceSum take their buffer from the
/// per-action scratch pool, whose captured graphs bake its address. ReleaseDeadDeviceStorage, which frees a gradient
/// backward dropped, disposed that buffer under the pool, so the next step's rental found it freed.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ActionScratchPoolOwnershipTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public ActionScratchPoolOwnershipTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void ReleasingADroppedPooledTensor_LeavesThePoolBufferLive()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");

        gpu.SuspendActivationEviction();   // with the capture path: a compiled step's resident scope, which pools
        using (gpu.EnterCompiledCapturePath())
        {
            try
            {
                Assert.True(gpu.ResidentStepActive);
                gpu.SetCurrentScratchAction(0);
                Assert.True(gpu.TryResidentZeros<float>(new[] { 64 }, out var zeros));
                var pooled = zeros._gpuBuffer;
                Assert.NotNull(pooled);

                gpu.ReleaseDeadDeviceStorage(zeros);

                Assert.NotEqual(IntPtr.Zero, pooled.Handle);
                // A freed buffer goes back to the backend's caching allocator, so the next allocation of the same size
                // would be handed the memory the pool still uses: the two would alias.
                var backend = gpu.GetBackend();
                Assert.NotNull(backend);
                using (var other = backend.AllocateBuffer(64))
                    Assert.NotEqual(pooled.Handle, other.Handle);
                gpu.SetCurrentScratchAction(0);   // the next step's same action and slot
                Assert.True(gpu.TryResidentZeros<float>(new[] { 64 }, out var again));
                Assert.Same(pooled, again._gpuBuffer);
                Assert.Equal(new float[64], again.ToArray());
            }
            finally
            {
                gpu.SetCurrentScratchAction(-1);
                gpu.ClearActionScratchPool();
                gpu.ResumeActivationEviction();
            }
        }
    }
}
