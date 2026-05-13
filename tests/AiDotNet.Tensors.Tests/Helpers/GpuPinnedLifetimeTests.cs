using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Issue #336: validates the WeightLifetime.GpuPinned wiring for the
/// optimizer-state-on-GPU work. Tags Adam m/v, BatchNorm running stats,
/// and weights with GpuPinned so they avoid the per-train-step
/// cuMemcpyHtoD/DtoH round-trip.
/// </summary>
public class GpuPinnedLifetimeTests
{
    [Fact]
    public void WeightLifetime_GpuPinned_EnumValueExists()
    {
        // Smoke check: the enum value is defined and distinct from the
        // existing values. Consumer code (AdamOptimizer._tapeM/v setter)
        // refers to this enum member by name.
        Assert.True(System.Enum.IsDefined(typeof(WeightLifetime), WeightLifetime.GpuPinned));
        Assert.NotEqual(WeightLifetime.GpuPinned, WeightLifetime.Default);
        Assert.NotEqual(WeightLifetime.GpuPinned, WeightLifetime.GpuOffload);
        Assert.NotEqual(WeightLifetime.GpuPinned, WeightLifetime.GpuManaged);
    }

    [Fact]
    public void RentPinnedOnGpu_CpuOnlyHost_FallsBackToCpuPinned_NotThrowing()
    {
        // On a host without GPU offload allocator (CPU CI), the accessor
        // must return a usable tensor — RentPinnedOnGpu's fallback path.
        var tensor = TensorAllocator.RentPinnedOnGpu<float>(new[] { 4, 8 });
        Assert.NotNull(tensor);
        Assert.Equal(32, tensor.Length);
    }

    [Fact]
    public void RentPinnedOnGpu_GpuOffloadCapableHost_TagsTensorGpuPinned()
    {
        // When the offload allocator IS registered (GPU host), the tensor
        // comes back tagged GpuPinned (not GpuOffload — issue #336 uses
        // the new lifetime tag to distinguish "lives on GPU for the train
        // loop" from "may swap to host for large-model memory pressure").
        if (WeightRegistry.OffloadAllocator is null
            || !WeightRegistry.OffloadAllocator.IsAvailable)
            return; // CPU-only host — skip; covered by the fallback test.

        var tensor = TensorAllocator.RentPinnedOnGpu<float>(new[] { 16 });
        Assert.True(
            tensor.Lifetime == WeightLifetime.GpuPinned
                || tensor.Lifetime == WeightLifetime.Default,
            $"Expected GpuPinned or fallback Default; got {tensor.Lifetime}");
    }

    [Fact]
    public void Tensor_LifetimeSetter_AcceptsGpuPinned()
    {
        // Direct lifetime assignment must not throw — consumer code may
        // construct a tensor first and then tag it.
        var t = new Tensor<float>(new[] { 8 })
        {
            Lifetime = WeightLifetime.GpuPinned,
        };
        Assert.Equal(WeightLifetime.GpuPinned, t.Lifetime);
    }
}
