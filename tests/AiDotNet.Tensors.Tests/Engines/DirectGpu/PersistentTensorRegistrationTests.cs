using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Registering, unregistering and invalidating a persistent tensor must not copy it.
/// </summary>
/// <remarks>
/// <para>
/// Registration used to convert the tensor to float and allocate a GPU buffer immediately, which
/// made cloning quadratic in memory: a cloned layer re-registers every parameter, so each clone
/// re-uploaded a whole model. On a 31.5M-parameter decoder that was ~120 MB of float conversion per
/// clone, and it exhausted memory across ten large models in one test process.
/// </para>
/// <para>
/// Cleanup had the same shape of defect: unregistering or invalidating a copy-on-write alias went
/// through the writable <c>GetDataArray</c> escape purely to obtain a cache key, which detached the
/// alias and copied the full tensor. BLIP and VideoCLIP reproduced it under a 1 GiB managed-heap cap.
/// </para>
/// <para>
/// These tests state the invariants rather than leaving them to be re-broken. Two kinds of evidence
/// are used: storage identity (a COW alias must still share its source's backing array afterwards),
/// which holds on every target framework, and managed allocation across the call, which needs the
/// per-thread allocation counter that net471 does not have. On net471 every scenario still runs and
/// asserts identity and values; only the byte count is unavailable.
/// </para>
/// </remarks>
public class PersistentTensorRegistrationTests
{
    private const int Side = 1024;
    private const int Elements = Side * Side;   // 4 MiB of float: a detach is unmistakable in the allocation figure

    /// <summary>
    /// A [1, Side] input whose only non-zero is a 1 in column 0, so a linear pass through a
    /// [Side, Side] weight returns the weight's first ROW verbatim: output[j] == weight[0, j].
    /// </summary>
    private static Tensor<float> UnitRowProbe()
    {
        var probe = new Tensor<float>(new[] { 1, Side });
        probe[0] = 1.0f;
        return probe;
    }

    /// <summary>
    /// Runs <paramref name="action"/> and returns the managed bytes it allocated on this thread, or
    /// <see langword="null"/> where the runtime has no per-thread allocation counter (net471).
    /// </summary>
    /// <remarks>
    /// THREAD-LOCAL, not process-wide: GetTotalAllocatedBytes counts every thread, so an unrelated
    /// test running concurrently could push a measurement over its threshold and fail it.
    /// </remarks>
    private static long? MeasureAllocatedBytes(Action action)
    {
#if NET6_0_OR_GREATER   // GetAllocatedBytesForCurrentThread is .NET Core 3.0+; this project also targets net471.
        long before = System.GC.GetAllocatedBytesForCurrentThread();
        action();
        return System.GC.GetAllocatedBytesForCurrentThread() - before;
#else
        action();
        return null;
#endif
    }

    private static void AssertDidNotCopyTheTensor(long? allocated, string operation, Tensor<float> tensor)
    {
        if (allocated is not long bytes)
            return;   // net471: the identity and value assertions carry the test

        Assert.True(
            bytes < 256 * 1024,
            $"{operation} allocated {bytes:N0} bytes; it must not detach the "
                + $"{(long)tensor.Length * sizeof(float):N0}-byte tensor.");
    }

    private static void AssertStillSharesStorage(Tensor<float> source, Tensor<float> alias)
    {
        float[]? sourceArray = source.GetBackingArrayForCacheLookupUnsafe();
        float[]? aliasArray = alias.GetBackingArrayForCacheLookupUnsafe();
        Assert.NotNull(sourceArray);
        Assert.True(
            ReferenceEquals(sourceArray, aliasArray),
            "the copy-on-write alias was detached from its source: cache maintenance requested writable storage.");
    }

    [Fact]
    public void UnregisterPersistentTensor_DoesNotDetachCopyOnWriteAlias()
    {
        using var engine = new DirectGpuTensorEngine();
        using var source = new Tensor<float>(new[] { Elements });
        source[0] = 17.25f;
        using var clone = (Tensor<float>)source.CloneShared();
        AssertStillSharesStorage(source, clone);

        engine.RegisterPersistentTensor(source, PersistentTensorRole.Weights);
        engine.RegisterPersistentTensor(clone, PersistentTensorRole.Weights);

        long? allocated = MeasureAllocatedBytes(() => engine.UnregisterPersistentTensor(clone));

        AssertDidNotCopyTheTensor(allocated, "Unregistering a COW alias", clone);
        AssertStillSharesStorage(source, clone);
        Assert.Equal(17.25f, source[0]);
        Assert.Equal(17.25f, clone[0]);

        engine.UnregisterPersistentTensor(source);
    }

    [Fact]
    public void InvalidatePersistentTensor_DoesNotDetachCopyOnWriteAlias()
    {
        using var engine = new DirectGpuTensorEngine();
        if (!engine.IsGpuAvailable)
            return;   // without a backend the invalidation path returns before it touches the tensor

        using var source = new Tensor<float>(new[] { Side, Side });
        source[0] = -8.5f;
        using var clone = (Tensor<float>)source.CloneShared();

        engine.RegisterPersistentTensor(source, PersistentTensorRole.Weights);
        engine.RegisterPersistentTensor(clone, PersistentTensorRole.Weights);

        // Registration no longer uploads, so an invalidation of a never-read tensor finds no cache
        // entry and returns early -- it would pass here without exercising anything. Read the alias
        // as a WEIGHT through a GPU op first so the persistent cache holds a buffer keyed by the
        // SHARED array.
        float[]? shared = clone.GetBackingArrayForCacheLookupUnsafe();
        Assert.NotNull(shared);
        using var probe = UnitRowProbe();
        using (var warm = engine.FusedLinear(probe, clone, null, FusedActivationType.None))
            Assert.Equal(-8.5f, warm[0]);   // row 0 of the weight, i.e. element [0,0]
        Assert.NotNull(engine.TryGetCachedBuffer(shared));

        long? allocated = MeasureAllocatedBytes(() => engine.InvalidatePersistentTensor(clone));

        AssertDidNotCopyTheTensor(allocated, "Invalidating a COW alias", clone);
        AssertStillSharesStorage(source, clone);
        Assert.Equal(-8.5f, source[0]);
        Assert.Equal(-8.5f, clone[0]);
        Assert.NotNull(engine.TryGetCachedBuffer(shared));   // re-uploaded under the same identity, not evicted

        engine.UnregisterPersistentTensor(clone);
        engine.UnregisterPersistentTensor(source);
    }

    /// <summary>
    /// The cache key and the upload source are different questions. The identity accessor used for
    /// the key never triggers a pending deferred GPU-to-CPU download, so if invalidation uploaded
    /// from it, a tensor whose newest value still lived on the device would push its STALE host
    /// bytes back to the GPU. The upload must read through an accessor that materialises first.
    /// </summary>
    [Fact]
    public void InvalidatePersistentTensor_UploadsTheMaterialisedValue_WhenADownloadIsPending()
    {
        using var engine = new DirectGpuTensorEngine();
        if (!engine.IsGpuAvailable)
            return;

        using var weight = new Tensor<float>(new[] { Side, Side });
        for (int i = 0; i < Elements; i++) weight[i] = 1.0f;
        float[]? array = weight.GetBackingArrayForCacheLookupUnsafe();
        Assert.NotNull(array);

        engine.RegisterPersistentTensor(weight, PersistentTensorRole.Weights);
        using var probe = UnitRowProbe();
        using (var warm = engine.FusedLinear(probe, weight, null, FusedActivationType.None))
            Assert.Equal(1.0f, warm[0]);
        Assert.NotNull(engine.TryGetCachedBuffer(array));

        // Stand in for FinishGpuOp: the host array is stale and a download that would fill it with the
        // device's current value (3.0) is registered but has not run yet.
        DeferredArrayMaterializer.Register(array, pending =>
        {
            var target = (float[])pending;
            for (int i = 0; i < target.Length; i++) target[i] = 3.0f;
        });
        try
        {
            Assert.True(DeferredArrayMaterializer.IsPending(array));

            engine.InvalidatePersistentTensor(weight);

            Assert.False(
                DeferredArrayMaterializer.IsPending(array),
                "invalidation uploaded without materialising the pending download; the GPU now holds stale data.");

            // Read the weight back THROUGH the cache: the weight path serves the buffer invalidation
            // just uploaded (its host Version stamp was refreshed), so this is the device-side value.
            using var readback = engine.FusedLinear(probe, weight, null, FusedActivationType.None);
            Assert.Equal(3.0f, readback[0]);
            Assert.Equal(3.0f, readback[Side - 1]);
        }
        finally
        {
            DeferredArrayMaterializer.Remove(array);
            engine.UnregisterPersistentTensor(weight);
        }
    }

    [Fact]
    public void RegisterPersistentTensor_DoesNotMaterialiseTheTensor()
    {
        // The engine under test must be the GPU one. AiDotNetEngine.Current starts as CpuEngine and
        // stays there when GPU detection is off or unavailable, and CpuEngine.RegisterPersistentTensor
        // is ALREADY a no-op -- so this would have passed without ever running the changed override.
        using var engine = new DirectGpuTensorEngine();
        if (!engine.IsGpuAvailable)
        {
            // No backend means TryGetBackend early-outs and there is nothing to assert about uploads.
            return;
        }

        var tensor = new Tensor<double>(new[] { 512, 512 });   // 2 MB of doubles
        for (int i = 0; i < tensor.Length; i += 512) tensor[i] = i * 0.5;

        long payload = (long)tensor.Length * sizeof(double);

        // Warm any one-time engine state so the measurement is of registration alone.
        engine.RegisterPersistentTensor(tensor, PersistentTensorRole.Weights);
        engine.UnregisterPersistentTensor(tensor);

        long? allocated = MeasureAllocatedBytes(() => engine.RegisterPersistentTensor(tensor, PersistentTensorRole.Weights));

        engine.UnregisterPersistentTensor(tensor);

        if (allocated is not long bytes)
            return;   // net471: no per-thread counter; the registration round-trip above still ran

        // A float copy of the payload would be half its size; anything approaching that means the
        // eager upload is back. Bookkeeping is allowed to allocate a little.
        long floatCopy = payload / 2;
        Assert.True(
            bytes < floatCopy / 4,
            $"registration allocated {bytes:N0} bytes for a {payload:N0}-byte tensor. A float "
                + $"copy would be about {floatCopy:N0}; the upload belongs on the read path, which "
                + "allocates on a cache miss and re-uploads when the host Version moves.");
    }
}
