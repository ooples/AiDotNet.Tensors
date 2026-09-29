using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// AutoTensorCache is process-wide; keep these off the parallel runner so another test's rentals cannot interleave.
[Collection("DeferredMaterializerRegistrySerial")]
public class AutoTensorCacheDeviceStateTests
{
    // A shape nothing else in the suite rents, so a hit can only be the tensor this test returned.
    private static readonly int[] Shape = { 7, 13, 3 };

    private static Tensor<float> RentFresh()
    {
        // Drain anything a previous run left in the pool for this shape.
        for (int i = 0; i < 64; i++) AutoTensorCache.RentOrAllocate<float>(Shape);
        return new Tensor<float>(Shape);
    }

    [Fact]
    public void A_host_tensor_is_reused()
    {
        var host = RentFresh();
        AutoTensorCache.Return(host);
        Assert.Same(host, AutoTensorCache.RentOrAllocate<float>(Shape));
    }

    /// <summary>
    /// A tensor bound to a device buffer must not come back out of the pool: the next renter would resolve it through
    /// the stale _gpuBuffer fast path and read memory it does not own. Measured: a captured training step reported a
    /// negative mean-squared loss late in a long test process.
    /// </summary>
    [Fact]
    public void A_device_bound_tensor_is_never_handed_out_again()
    {
        var bound = RentFresh();
        bound._gpuBuffer = new MockGpuBuffer(new float[bound.Length]);
        AutoTensorCache.Return(bound);
        Assert.NotSame(bound, AutoTensorCache.RentOrAllocate<float>(Shape));
    }

    [Fact]
    public void A_tensor_with_a_pending_device_download_is_never_handed_out_again()
    {
        var pending = RentFresh();
        var backing = pending.GetBackingArrayForCacheLookupUnsafe()!;
        DeferredArrayMaterializer.Register(backing, _ => { });
        try
        {
            AutoTensorCache.Return(pending);
            Assert.NotSame(pending, AutoTensorCache.RentOrAllocate<float>(Shape));
        }
        finally
        {
            DeferredArrayMaterializer.Remove(backing);
        }
    }
}
