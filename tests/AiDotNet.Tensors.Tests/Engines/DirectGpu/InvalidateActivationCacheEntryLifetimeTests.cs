using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Regression guard for the #226-class race re-surfaced via #1468:
/// <see cref="DirectGpuTensorEngine"/>'s stale-buffer re-upload paths
/// (<c>UploadTensor</c>) call <c>InvalidateActivationCacheEntry</c>, which disposed
/// the GPU buffer WITHOUT coordinating with a still-pending deferred-download
/// materializer. The buffer was freed, then the next CPU access (a CNN backward
/// pass reading an activation in <c>ComputeGradients</c>) fired the materializer
/// against the freed buffer → "buffer released before materialization"
/// (OpenCL -5 / CL_INVALID_MEM_OBJECT), crashing <c>AiModelBuilder.BuildAsync</c>
/// for every prebuilt CNN/RNN model.
///
/// The fix centralizes the guard inside <c>InvalidateActivationCacheEntry</c>:
/// because <see cref="Helpers.DeferredArrayMaterializer.Register"/> is
/// first-write-wins, the array key is permanently bound to exactly one buffer, so
/// that buffer's contents are the array's ONLY defined CPU value. The entry is
/// therefore materialized (downloaded) to CPU BEFORE the buffer is disposed.
///
/// This reflection-based test pins the materialize-before-dispose ordering on any
/// CI host without requiring a real GPU runtime — mirroring
/// <see cref="ActivationCacheEvictionLifetimeTests"/>.
/// </summary>
public class InvalidateActivationCacheEntryLifetimeTests
{
    /// <summary>Fake GPU buffer — tracks Dispose so the test can assert on lifetime.</summary>
    private sealed class TrackingGpuBuffer : IGpuBuffer
    {
        public int DisposeCount;
        public int Size { get; }
        public long SizeInBytes => Size * sizeof(float);
        public IntPtr Handle => IntPtr.Zero;

        public TrackingGpuBuffer(int size) { Size = size; }
        public void Dispose() => DisposeCount++;
    }

    private static ConstructorInfo GetActivationCacheEntryCtor(Type activationCacheEntryType)
    {
        var ctor = activationCacheEntryType.GetConstructor(new[]
        {
            typeof(IGpuBuffer),
            typeof(int[]),
            typeof(long),
            typeof(IDirectGpuBackend)
        });
        Assert.NotNull(ctor);
        return ctor!;
    }

    [Fact]
    public void InvalidateActivationCacheEntry_MaterializesPendingDownloadBeforeDisposingBuffer()
    {
        using var engine = new DirectGpuTensorEngine();

        var engineType = typeof(DirectGpuTensorEngine);
        var activationCacheField = engineType.GetField(
            "_activationCache", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var invalidateMethod = engineType.GetMethod(
            "InvalidateActivationCacheEntry", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var activationCacheEntryType = engineType.Assembly.GetType(
            "AiDotNet.Tensors.Engines.ActivationCacheEntry")!;
        var entryCtor = GetActivationCacheEntryCtor(activationCacheEntryType);

        object activationCache = activationCacheField.GetValue(engine)!;
        var tryAdd = activationCache.GetType().GetMethod("TryAdd")!;

        // The cache key and the buffer it backs. The materializer's job is to copy
        // this buffer's contents into the key array — its only defined CPU value.
        var key = new float[4];
        var buffer = new TrackingGpuBuffer(size: 4);
        object entry = entryCtor.Invoke(new object?[] { buffer, new[] { 4 }, 1L, null });
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { (object)key, entry })!);

        // Register a pending materializer. It records the buffer's DisposeCount at the
        // moment it runs — which MUST be 0, proving the buffer was still alive (not
        // freed) when the deferred download executed.
        int disposeCountWhenMaterialized = -1;
        bool materializerRan = false;
        AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.Register(key, _ =>
        {
            materializerRan = true;
            disposeCountWhenMaterialized = buffer.DisposeCount;
        });

        try
        {
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(key));

            invalidateMethod.Invoke(engine, new object[] { key });

            // 1. The pending download was materialized (not silently dropped).
            Assert.True(materializerRan,
                "InvalidateActivationCacheEntry must materialize a pending deferred download " +
                "instead of disposing its buffer out from under it (#226 / #1468).");

            // 2. Ordering: the buffer was still alive when the download ran.
            Assert.Equal(0, disposeCountWhenMaterialized);

            // 3. No longer pending — a subsequent read won't touch a freed buffer.
            Assert.False(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(key));

            // 4. The buffer was disposed afterward (no leak).
            Assert.Equal(1, buffer.DisposeCount);
        }
        finally
        {
            AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.Remove(key);
        }
    }
}
