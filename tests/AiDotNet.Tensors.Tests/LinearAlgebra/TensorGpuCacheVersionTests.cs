using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

public sealed class TensorGpuCacheVersionTests
{
    [Fact]
    public void SharedViews_AdvanceOneCacheEpochWithoutChangingOtherAutogradVersions()
    {
        var source = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        int epoch = source.GpuCacheVersion;
        int version = source.Version;
        var view = source.Reshape(new[] { 1, 2 });
        view[0] = 9f;
        Assert.True(source.GpuCacheVersion > epoch);
        Assert.Equal(source.GpuCacheVersion, view.GpuCacheVersion);
        Assert.Equal(version, source.Version);
    }

    [Fact]
    public void InferenceMutation_AdvancesCacheEpochButNotAutogradVersion()
    {
        var tensor = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        int epoch = tensor.GpuCacheVersion;
        int version = tensor.Version;
        using (new InferenceModeScope<double>()) tensor[0] = 2.0;
        Assert.True(tensor.GpuCacheVersion > epoch);
        Assert.Equal(version, tensor.Version);
    }

    [Fact]
    public void CopyOnWrite_DetachmentPreservesTrackingWithoutInvalidatingUntouchedSibling()
    {
        var source = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        int epoch = source.GpuCacheVersion;
        var clone = (Tensor<float>)source.CloneShared();
        using (new InferenceModeScope<float>()) clone[0] = 9f;
        Assert.Equal(epoch, source.GpuCacheVersion);
        Assert.True(clone.GpuCacheVersion > epoch);
        Assert.Equal(new[] { 1f, 2f }, source.ToArray());
        Assert.Equal(new[] { 9f, 2f }, clone.ToArray());
    }

    [Fact]
    public void CpuOnlyWrites_DoNotPayForGpuEpochUntilStorageParticipatesInCaching()
    {
        var tensor = new Tensor<float>(new[] { 1 });
        using (new InferenceModeScope<float>())
            for (int i = 0; i < 100; i++) tensor[0] = i;
        Assert.Equal(0, tensor.GpuCacheVersion);
        using (new InferenceModeScope<float>()) tensor[0] = 101;
        Assert.Equal(1, tensor.GpuCacheVersion);
    }

    [Fact]
    public void ExclusiveClaim_DoesNotInvokeFallibleResourceDisposalBeforeStorageSwap()
    {
        var storage = new TensorStorage<float>(new Vector<float>(2));
        var resource = new ThrowingOwner();
        storage.AttachGpuBufferOwner(resource);
        Assert.True(storage.TryClaimExclusive());
        Assert.Equal(0, resource.Attempts);
        Assert.Throws<InvalidOperationException>(() => storage.DisposeClaimedOwners());
        Assert.Equal(1, resource.Attempts);
        storage.DisposeClaimedOwners();
        Assert.Equal(1, resource.Attempts);
    }

    private sealed class ThrowingOwner : IDisposable
    {
        internal int Attempts { get; private set; }
        public void Dispose()
        {
            Attempts++;
            throw new InvalidOperationException("Injected resource cleanup failure.");
        }
    }
}
