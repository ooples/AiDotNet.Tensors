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

        // PR #345 review: tighten assertion to require GpuPinned exactly.
        // Permitting `Default` masked regressions in the new tagging
        // behavior — the whole point of issue #336 is that pinned-on-GPU
        // tensors are explicitly tagged as such. Also unregister the
        // tensor afterward so each test run doesn't accumulate
        // process-wide offload state.
        var tensor = TensorAllocator.RentPinnedOnGpu<float>(new[] { 16 });
        try
        {
            Assert.Equal(WeightLifetime.GpuPinned, tensor.Lifetime);
        }
        finally
        {
            WeightRegistry.UnregisterWeight(tensor);
        }
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

    [Fact]
    public void TryGetGpuBuffer_CpuOnlyHost_ReturnsNull()
    {
        var t = new Tensor<float>(new[] { 16 });
        Assert.Null(t.TryGetGpuBuffer());
    }

    [Fact]
    public void TryGetGpuBuffer_GpuPinnedTagWithoutPointer_ReturnsNull()
    {
        var t = new Tensor<float>(new[] { 16 })
        {
            Lifetime = WeightLifetime.GpuPinned,
        };
        Assert.Null(t.TryGetGpuBuffer());
    }

    [Fact]
    public void GpuOptimizer_TryAdamStep_CpuEngine_ReturnsFalse()
    {
        var priorEngine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current = new AiDotNet.Tensors.Engines.CpuEngine();
        try
        {
            var param = new Tensor<float>(new[] { 4 });
            var grad = new Tensor<float>(new[] { 4 });
            var m = new Tensor<float>(new[] { 4 });
            var v = new Tensor<float>(new[] { 4 });
            var ran = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamStep(
                param, grad, m, v,
                learningRate: 0.01f, beta1: 0.9f, beta2: 0.999f,
                epsilon: 1e-8f, weightDecay: 0f, step: 1);
            Assert.False(ran);
        }
        finally { AiDotNet.Tensors.Engines.AiDotNetEngine.Current = priorEngine; }
    }

    [Fact]
    public void GpuOptimizer_TrySgdStep_CpuEngine_ReturnsFalse()
    {
        var priorEngine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current = new AiDotNet.Tensors.Engines.CpuEngine();
        try
        {
            var param = new Tensor<float>(new[] { 4 });
            var grad = new Tensor<float>(new[] { 4 });
            var ran = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TrySgdStep(param, grad, 0.01f);
            Assert.False(ran);
        }
        finally { AiDotNet.Tensors.Engines.AiDotNetEngine.Current = priorEngine; }
    }

    [Fact]
    public void GpuOptimizer_TryAdamStep_NullArg_Throws()
    {
        var t = new Tensor<float>(new[] { 4 });
        Assert.Throws<System.ArgumentNullException>(() =>
            AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamStep(
                null!, t, t, t, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1));
    }
}
