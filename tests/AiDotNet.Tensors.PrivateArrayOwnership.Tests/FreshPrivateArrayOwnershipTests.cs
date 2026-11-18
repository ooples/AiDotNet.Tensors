using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// A separate test assembly guarantees a fresh CLR registry. Do not merge this fixture into the large test process:
// ordinary result-recycling tests may legitimately populate that process's unrelated recyclable-array registry.
public sealed class FreshPrivateArrayOwnershipTests
{
    [Fact]
    public void Memory_and_raw_array_wrappers_find_private_owner_when_recyclable_registry_is_empty()
    {
        Assert.Equal(0, ResultBufferTracker.TrackedCount);
        Assert.Equal(0, ResultBufferTracker.TrackedBytes);
        var array = new float[6];
        var owner = Assert.IsType<ResultBufferOwner>(ResultBufferTracker.TrackFreshPrivateArray(array, 0));
        Assert.Equal(0, owner.State.Wrappers);
        using var memoryWrapper = Tensor<float>.FromMemory(array.AsMemory(), new[] { 2, 3 });
        using var rawArrayWrapper = new Tensor<float>(array, new[] { 2, 3 });
        Assert.Same(array, memoryWrapper.GraphBackingArrayIdentity);
        Assert.Same(array, rawArrayWrapper.GraphBackingArrayIdentity);
        Assert.Same(owner, ((Vector<float>)memoryWrapper.GraphVectorIdentity).ResultOwner);
        Assert.Same(owner, ((Vector<float>)rawArrayWrapper.GraphVectorIdentity).ResultOwner);
        Assert.NotSame(memoryWrapper.GraphVectorIdentity, rawArrayWrapper.GraphVectorIdentity);
        Assert.NotSame(memoryWrapper.StorageIdentity, rawArrayWrapper.StorageIdentity);
        Assert.Equal(2, owner.State.Wrappers);
        Assert.Equal(0, ResultBufferTracker.TrackedCount);
        Assert.Equal(0, ResultBufferTracker.TrackedBytes);
        _ = memoryWrapper.Data.Span[0];
        Assert.True(owner.State.Escaped);
        Assert.Equal(0, ResultBufferTracker.TrackedCount);
        Assert.Equal(0, ResultBufferTracker.TrackedBytes);
        GC.KeepAlive(owner);
    }
}
